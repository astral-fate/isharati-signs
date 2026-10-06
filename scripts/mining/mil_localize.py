"""From the spotter's candidates (train.py spot) to signs.

For each word, the candidate stretches (one per top clip) are compared with each other; the medoid (nearest to the
others) is the sign, and the mean distance from it to the other candidates is the agreement (low = the clips agree on
one motion). Evaluation on the held-out words (no lexicon example was used in training): the rank of the true lexicon
sign among all lexicon signs for the medoid, as in motion_spot.py eval, against a random stretch of the same
clips. Missing words: the medoid is saved as a pending-review sign.

  .venv/Scripts/python scripts/mining/mil_localize.py
Output: results/spotter_eval.json; data/asl/lexicon/youtube_asl/{signs/*.npy, entries.jsonl, review.json}
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from isharati.pose.dtw import resample  # noqa: E402
from motion_spot import LEX, YT, lexicon, sign_vec  # noqa: E402

MIL = YT / "mil"
OUT = LEX / "youtube_asl"
SRC_FPS, FPS = 30, 25


def main():
    cands = json.loads((MIL / "candidates.json").read_text(encoding="utf-8"))
    meta = json.loads((MIL / "meta.json").read_text(encoding="utf-8"))
    part = {it["id"]: it for it in meta["items"]}
    z = np.load(YT / "poses" / "raw_keypoints_1.npz")
    glosses, V = lexicon()
    rng = random.Random(0)

    def rank(vec, w):
        dist = np.sqrt(((V - vec) ** 2).sum(1))
        true = min(dist[i] for i, g in enumerate(glosses) if g == w)
        return int((dist < true).sum()) + 1

    results, review = {}, []
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries = []
    for w, c in cands.items():
        segs = []
        for k in c["candidates"]:
            p = z[k["clip"]].astype(np.float32)
            segs.append((k, p[k["start"]: min(len(p), k["end"])], p))
        segs = [s for s in segs if len(s[1]) >= 6]
        if len(segs) < 2:
            continue
        vecs = np.stack([sign_vec(s[1]) for s in segs])
        D = np.sqrt(((vecs[:, None] - vecs[None]) ** 2).sum(-1))
        med = int(D.sum(1).argmin())
        agree = float(D[med][np.arange(len(segs)) != med].mean())
        k, seg, full = segs[med]
        r = {"n_clips": c["n_clips"], "agreement": round(agree, 3), "clip": k["clip"], "start": k["start"],
             "end": k["end"], "peak": round(k["peak"], 3)}
        if c["group"] == "held_out":
            r["rank"] = rank(vecs[med], w)
            # chance: a random stretch of the same length from a random candidate clip
            kk, _, pp = segs[rng.randrange(len(segs))]
            L = len(seg)
            s0 = rng.randrange(0, max(1, len(pp) - L))
            r["random_rank"] = rank(sign_vec(pp[s0:s0 + L]), w)
        else:
            sid = f"yt_{w}"
            np.save(OUT / "signs" / f"{sid}.npy", resample(seg, max(6, round(len(seg) * FPS / SRC_FPS))))
            entries.append({"gloss": w.upper(), "sign_id": sid, "dataset": "youtube_asl",
                            "keypoints_path": f"youtube_asl/signs/{sid}.npy", "review_status": "pending",
                            "is_religious": False, "is_letter": False})
            vid, span = k["clip"].rsplit(".", 1)
            t0 = int(span.split("-")[0]) + k["start"]
            review.append({"word": w, "agreement": r["agreement"], "clip": k["clip"],
                           "youtube": f"https://www.youtube.com/watch?v={vid}&t={max(0, t0 // SRC_FPS - 1)}s",
                           "others": [x["clip"] for x in c["candidates"] if x["clip"] != k["clip"]][:3]})
        results[w] = r
    held = {w: r for w, r in results.items() if "rank" in r}
    ranks = np.array([r["rank"] for r in held.values()])
    rand = np.array([r["random_rank"] for r in held.values()])
    agree = np.array([r["agreement"] for r in held.values()])

    def summary(x):
        return {"top1": float((x <= 1).mean()), "top10": float((x <= 10).mean()), "top50": float((x <= 50).mean()),
                "median_rank": float(np.median(x))}
    from scipy.stats import spearmanr
    rho = spearmanr(agree, ranks)
    ev = {"held_out_words": len(held), "lexicon_signs": len(glosses), "spotter": summary(ranks),
          "random_stretch": summary(rand), "agreement_vs_rank_spearman": [float(rho.statistic), float(rho.pvalue)],
          "by_agreement": {}, "per_word": held}
    for q in (25, 50):
        t = float(np.percentile(agree, q))
        ev["by_agreement"][f"most_agreeing_{q}pct"] = {"threshold": t, **summary(ranks[agree <= t])}
    (ROOT / "results" / "spotter_eval.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    (OUT / "review.json").write_text(json.dumps(sorted(review, key=lambda r: r["agreement"]), indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in ev.items() if k != "per_word"}, indent=1))
    print(len(entries), "pending-review signs ->", OUT)


if __name__ == "__main__":
    main()
