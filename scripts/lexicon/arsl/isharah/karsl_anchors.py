"""Find KArSL signs inside Isharah sentences: real per-sign boundaries for the glosses both datasets share.

CTC alignment turned out to give no usable sign timing (alignment_agreement.md). Here the boundaries come
from matching clips instead: for every Isharah sentence that contains a gloss KArSL also has, the KArSL
clips of that sign (canonical + held-out signer) are searched in the sentence with subsequence DTW, which
finds the stretch of signing that looks most like the isolated sign.

Features per frame: both wrists relative to the neck, and each hand's 21 points relative to its own wrist
(hand shape), all in shoulder widths (the Isharati pose contract is already neck-centred and scaled).

Checks that need no hand labels:
  discrimination  the sign that is in the sentence should match better than KArSL signs that are not
                  (share of distractors it beats; 50% = chance)
  order           two anchored glosses of one sentence should be found in the sentence's gloss order
                  (50% = chance)
  per gloss       discrimination per sign, so only reliable signs become anchors

Run with the models venv (numpy + numba):
  D:/islam/models/.venv/Scripts/python scripts/lexicon/arsl/isharah/karsl_anchors.py [--limit N]

Outputs in data/lexicon_v2/isharah/:
  karsl_anchors.jsonl   {sid, gloss, sign_id, start, end (frames at 25 fps), cost, beats}
  karsl_anchors.md      the checks, overall and per gloss
"""
import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from gloss_clips import ISHARAH, PKL, to_contract  # noqa: E402
from isharati.text.arabic import normalize_ar  # noqa: E402

LEX = DATA / "lexicon"
OUT = DATA / "lexicon_v2" / "isharah"
N_DISTRACTORS = 8
MIN_BEATS = 0.75      # a gloss becomes an anchor sign if it beats this share of distractors on average


def features(p):
    """[T,50,3] contract pose -> [T, 88]: wrists (2x2) + both hands' shapes (2 x 21 x 2)."""
    wr = p[:, [7, 4], :2]                                  # left, right wrist (neck-centred)
    lh = p[:, 8:29, :2] - p[:, 8:9, :2]
    rh = p[:, 29:50, :2] - p[:, 29:30, :2]
    return np.concatenate([wr.reshape(len(p), -1), lh.reshape(len(p), -1), rh.reshape(len(p), -1)],
                          axis=1).astype(np.float32)


@njit(cache=True)
def subsequence_dtw(q, s):
    """Best match of query q (m,d) anywhere in sequence s (n,d): open start and end in s.
    Returns (start, end exclusive, cost per matched query frame)."""
    m, n = q.shape[0], s.shape[0]
    D = np.full((m + 1, n + 1), np.inf)
    start = np.zeros((m + 1, n + 1), np.int64)
    for j in range(n + 1):
        D[0, j] = 0.0
        start[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = 0.0
            for k in range(q.shape[1]):
                diff = q[i - 1, k] - s[j - 1, k]
                c += diff * diff
            c = np.sqrt(c)
            best, src = D[i - 1, j - 1], 0
            if D[i - 1, j] < best:
                best, src = D[i - 1, j], 1
            if D[i, j - 1] < best:
                best, src = D[i, j - 1], 2
            D[i, j] = c + best
            if src == 0:
                start[i, j] = start[i - 1, j - 1]
            elif src == 1:
                start[i, j] = start[i - 1, j]
            else:
                start[i, j] = start[i, j - 1]
    end = 1
    for j in range(1, n + 1):
        if D[m, j] < D[m, end]:
            end = j
    return start[m, end], end, D[m, end] / m


def karsl_queries():
    """normalised gloss -> (sign_id, [feature arrays of its KArSL clips])."""
    out = {}
    for fname, folder in (("lexicon.jsonl", "signs"), ("templates.jsonl", "templates")):
        for line in (LEX / fname).read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["is_letter"]:
                continue
            key = normalize_ar(e["gloss"])
            clip = np.load(LEX / e["keypoints_path"])
            sid, clips = out.setdefault(key, (e["sign_id"], []))
            clips.append(features(clip))
    return out


def best_match(queries, feats):
    """Lowest-cost match over a sign's KArSL clips."""
    return min((subsequence_dtw(q, feats) for q in queries), key=lambda r: r[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    rng = random.Random(0)
    queries = karsl_queries()
    keys = list(queries)
    sentences = {}
    for split in ("train.csv", "dev.csv"):
        for line in (ISHARAH / split).read_text(encoding="utf-8").splitlines()[1:]:
            if "|" in line:
                sid, g = line.split("|", 1)
                sentences[sid] = [t.replace("_", " ") for t in g.split()]
    print("loading", PKL.name, flush=True)
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    ids = [s for s in sentences if s in data]
    if args.limit:
        ids = ids[:args.limit]

    anchors, per_gloss, order_ok, order_n = [], defaultdict(list), 0, 0
    for n, sid in enumerate(ids):
        entry = data.pop(sid)
        shared = [(i, normalize_ar(g)) for i, g in enumerate(sentences[sid]) if normalize_ar(g) in queries]
        if not shared:
            continue
        try:
            feats = features(to_contract(entry["keypoints"]))
        except ValueError:
            continue
        present = {k for _, k in shared}
        found = []
        for pos, key in shared:
            s, e, cost = best_match(queries[key][1], feats)
            others = [k for k in rng.sample(keys, N_DISTRACTORS + len(present)) if k not in present][:N_DISTRACTORS]
            beats = float(np.mean([cost < best_match(queries[k][1], feats)[2] for k in others]))
            per_gloss[key].append(beats)
            found.append((pos, (s + e) / 2))
            anchors.append({"sid": sid, "gloss": sentences[sid][pos], "sign_id": queries[key][0],
                            "position": pos, "start": int(s), "end": int(e), "cost": round(float(cost), 4),
                            "beats": round(beats, 3)})
        for (p1, c1), (p2, c2) in ((a, b) for i, a in enumerate(found) for b in found[i + 1:]):
            if p1 != p2:
                order_n += 1
                order_ok += (c1 < c2) == (p1 < p2)
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1} sentences, {len(anchors)} anchors", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    reliable = {k for k, v in per_gloss.items() if len(v) >= 5 and np.mean(v) >= MIN_BEATS}
    with open(OUT / "karsl_anchors.jsonl", "w", encoding="utf-8") as f:
        for a in anchors:
            a["reliable_sign"] = normalize_ar(a["gloss"]) in reliable
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    disc = np.mean([a["beats"] for a in anchors]) if anchors else float("nan")
    rows = sorted(per_gloss.items(), key=lambda kv: -np.mean(kv[1]))
    report = [
        "# KArSL signs found in Isharah sentences (subsequence DTW)", "",
        f"- Sentences searched: {len(ids):,}; shared glosses: {len(per_gloss)} of {len(queries)} KArSL signs",
        f"- Anchors (sentence x gloss): {len(anchors):,}", "",
        "| Check | Result | Chance |", "|---|---|---|",
        f"| discrimination: true sign beats distractor signs | {disc:.1%} | 50% |",
        f"| order: two anchors found in gloss order ({order_n:,} pairs) | {order_ok / max(order_n, 1):.1%} | 50% |",
        f"| signs reliable as anchors (≥ 5 uses, beats ≥ {MIN_BEATS:.0%}) | {len(reliable)} | |", "",
        "## Per gloss", "", "| Gloss | uses | beats distractors |", "|---|---|---|",
    ] + [f"| {k} | {len(v)} | {np.mean(v):.0%} |" for k, v in rows]
    (OUT / "karsl_anchors.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
