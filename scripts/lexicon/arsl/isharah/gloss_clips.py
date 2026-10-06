"""Isharah sentences -> one canonical pose clip per gloss, in the Isharati pose contract [T,50,3].

Isharah (Saudi SL, continuous) gives each sentence a gloss sequence but no per-gloss timing. This script:
  1. converts every sample to the Isharati contract: MediaPipe pose 0/11-16 (+ neck = mid-shoulders),
     left hand = points 21-41, right hand = points 0-20 (verified against the body's wrists), pixels
     scaled to 0..1 of the frame like MediaPipe's own output, resampled to 25 fps, then
     isharati.pose.keypoints.normalize (neck-centred, shoulder-scaled);
  2. trims the resting frames at both ends and cuts the signing into N segments at the N-1 deepest
     pauses in hand motion, N = number of glosses in the sentence;
  3. per gloss, takes the DTW medoid of its segments (drawn from as many signers as possible) as the
     canonical clip, and its mean distance to the others as a consistency score;
  4. validates the cutting: same-gloss segments should be closer than different-gloss ones, and closer
     than with random cut points (the control).

Isharah is CC BY-NC-ND: these clips are for internal use and never published (spec §4).

Outputs in data/lexicon_v2/isharah/:
  signs/ish_NNNN.npy     canonical clip per gloss
  entries.jsonl          SignEntry rows (dataset "isharah", review_status "pending")
  glosses.csv            gloss -> occurrences, signers, consistency, clip
  report.md              validation numbers
"""
import csv
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.text.arabic import normalize_ar  # noqa: E402
from isharati.pose.dtw import dtw_distance  # noqa: E402
from isharati.pose.keypoints import interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS, N_JOINTS  # noqa: E402

ISHARAH = Path(r"D:\islam\ishara")
PKL = ISHARAH / "pose_data_isharah2000_hands_lips_body_phase1_SI.pkl"
OUT = DATA / "lexicon_v2" / "isharah"
SRC_FPS = 30.0          # smartphone recordings; Isharah does not state its frame rate
BODY = 61               # Isharah points 61-85 are MediaPipe pose landmarks 0-24
MIN_OCC = 3
MAX_CANDIDATES = 12
# Isharati slot -> Isharah point
BODY_MAP = {0: BODY + 0, 2: BODY + 12, 3: BODY + 14, 4: BODY + 16, 5: BODY + 11, 6: BODY + 13, 7: BODY + 15}


def frame_size(x):
    """Frame (W, H) in pixels. Landscape 1920x1080 unless a point lies below y=1080 (portrait)."""
    return (1080.0, 1920.0) if np.nanmax(x[..., 1]) > 1085 else (1920.0, 1080.0)


def to_contract(x):
    """Isharah (T,86,2) pixels -> Isharati [T',50,3] at 25 fps, normalised."""
    x = np.asarray(x, np.float32)
    x[(x == 0).all(axis=-1)] = np.nan              # (0,0) = not detected
    w, h = frame_size(x)
    x = x / np.array([w, h], np.float32)
    t = x.shape[0]
    out = np.full((t, N_JOINTS, 3), np.nan, np.float32)
    for slot, src in BODY_MAP.items():
        out[:, slot, :2] = x[:, src]
    out[:, 1, :2] = (out[:, 2, :2] + out[:, 5, :2]) / 2
    out[:, 8:29, :2] = x[:, 21:42]
    out[:, 29:50, :2] = x[:, 0:21]
    out[..., 2] = np.where(np.isnan(out[..., 0]), np.nan, 0.0)
    n = max(1, int(round(t * FPS / SRC_FPS)))
    out = out[np.linspace(0, t - 1, n).round().astype(int)]
    filled, _ = interpolate_missing(out)
    return normalize(filled)


def hand_speed(p):
    wrists = p[:, [4, 7], :2]
    hands = np.concatenate([p[:, 8:29, :2], p[:, 29:50, :2]], axis=1)
    v = np.linalg.norm(np.diff(np.concatenate([wrists, hands], axis=1), axis=0), axis=-1).mean(axis=1)
    return np.convolve(np.concatenate([[v[0]], v]), np.ones(3) / 3, "same")


def cut(p, n, rng=None):
    """Split a sentence into n gloss segments: trim rest at both ends, then cut at the n-1 deepest
    pauses (local minima of hand speed), at least a few frames apart. rng -> random cuts (control)."""
    s = hand_speed(p)
    active = np.where(s > 0.2 * s.max())[0]
    a, b = (int(active[0]), int(active[-1]) + 1) if len(active) else (0, len(p))
    if b - a < 2 * n:
        return None
    if rng is not None:
        cuts = sorted(rng.sample(range(a + 2, b - 2), n - 1)) if n > 1 else []
    else:
        seg = s[a:b]
        minima = [i for i in range(1, len(seg) - 1) if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]]
        min_gap = max(2, (b - a) // (3 * n))
        cuts = []
        for i in sorted(minima, key=lambda i: seg[i]):
            if all(abs(i - c) >= min_gap for c in cuts) and min_gap <= i <= len(seg) - min_gap:
                cuts.append(i)
            if len(cuts) == n - 1:
                break
        if len(cuts) < n - 1:  # not enough pauses: fall back to equal parts
            cuts = [round((b - a) * k / n) for k in range(1, n)]
        cuts = sorted(a + c for c in cuts)
    bounds = [a] + cuts + [b]
    return [(bounds[k], bounds[k + 1]) for k in range(n)]


def load():
    glosses = {}
    for split in ("train.csv", "dev.csv"):
        for line in (ISHARAH / split).read_text(encoding="utf-8").splitlines()[1:]:
            if "|" in line:
                sid, g = line.split("|", 1)
                glosses[sid] = [t.replace("_", " ") for t in g.split()]
    print("loading", PKL.name, flush=True)
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    return glosses, data


def collect(glosses, data, rng=None):
    segs = defaultdict(list)
    for sid, entry in data.items():
        g = glosses.get(sid)
        if not g:
            continue
        try:
            p = to_contract(entry["keypoints"])
        except ValueError:
            continue  # no shoulders detected
        parts = cut(p, len(g), rng)
        if not parts:
            continue
        for gloss, (a, b) in zip(g, parts):
            segs[normalize_ar(gloss)].append({"sid": sid, "signer": sid.split("_")[0], "surface": gloss,
                                              "clip": p[a:b]})
    return segs


def collect_aligned(data, path, free=False):
    """Segments from CTC forced alignment (ctc_align.py): spans are in source frames.
    Clips are kept as float16 copies; free=True drops each sentence's raw keypoints once converted."""
    segs = defaultdict(list)
    scale = FPS / SRC_FPS
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        entry = data.pop(row["sid"], None) if free else data.get(row["sid"])
        if entry is None:
            continue
        try:
            p = to_contract(entry["keypoints"])
        except ValueError:
            continue
        for gloss, (a, b) in zip(row["glosses"], row["spans"]):
            a, b = int(round(a * scale)), max(int(round(b * scale)), int(round(a * scale)) + 2)
            if b <= len(p):
                gloss = gloss.replace("_", " ")
                segs[normalize_ar(gloss)].append({"sid": row["sid"], "signer": row["sid"].split("_")[0],
                                                  "surface": gloss, "clip": p[a:b].astype(np.float16)})
    return segs


def separation(segs, pairs=500, seed=0):
    rng = random.Random(seed)
    multi = [v for v in segs.values() if len(v) >= 2]
    flat = [s for v in segs.values() for s in v]
    same, diff = [], []
    for _ in range(pairs):
        a, b = rng.sample(rng.choice(multi), 2)
        c = rng.choice(flat)
        if normalize_ar(c["surface"]) != normalize_ar(a["surface"]):
            f32 = lambda s: s["clip"].astype(np.float32)
            same.append(dtw_distance(f32(a), f32(b)))
            diff.append(dtw_distance(f32(a), f32(c)))
    return float(np.mean(same)), float(np.mean(diff))


def pick(occ):
    """Candidates spread over signers, then the DTW medoid."""
    by_signer = defaultdict(list)
    for o in occ:
        by_signer[o["signer"]].append(o)
    cand, k = [], 0
    while len(cand) < MAX_CANDIDATES and any(len(v) > k for v in by_signer.values()):
        cand += [v[k] for v in by_signer.values() if len(v) > k]
        k += 1
    cand = [{**c, "clip": c["clip"].astype(np.float32)} for c in cand[:MAX_CANDIDATES]]
    d = np.zeros((len(cand), len(cand)))
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            d[i, j] = d[j, i] = dtw_distance(cand[i]["clip"], cand[j]["clip"])
    m = int(d.sum(1).argmin())
    return cand[m], float(d[np.triu_indices(len(cand), 1)].mean()), len(by_signer)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="use only the first N samples (quick test)")
    ap.add_argument("--alignments", type=Path, help="alignments.jsonl from ctc_align.py")
    args = ap.parse_args()
    glosses, data = load()
    if args.limit:
        data = dict(list(data.items())[:args.limit])
    n_sentences = len(data)
    # validation on a sample of sentences, so the three ways of cutting never sit in memory at full size
    sample = dict(random.Random(2).sample(sorted(data.items()), min(2000, len(data))))
    pauses = collect(glosses, sample)
    control = collect(glosses, sample, rng=random.Random(1))
    same_p, diff_p = separation(pauses)
    same_c, diff_c = separation(control)
    if args.alignments:
        same, diff = separation(collect_aligned(sample, args.alignments))
        del pauses, control, sample
        segs = collect_aligned(data, args.alignments, free=True)
    else:
        same, diff, segs = same_p, diff_p, pauses

    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries, table = [], []
    for n, (key, occ) in enumerate(sorted(segs.items(), key=lambda kv: -len(kv[1]))):
        if len(occ) < MIN_OCC:
            continue
        best, spread, n_signers = pick(occ)
        sign_id = f"ish_{n:04d}"
        np.save(OUT / "signs" / f"{sign_id}.npy", best["clip"])
        entries.append({"gloss": best["surface"], "sign_id": sign_id, "dataset": "isharah",
                        "keypoints_path": f"signs/{sign_id}.npy", "review_status": "pending",
                        "is_religious": False, "is_letter": False})
        table.append({"gloss": key, "surface": best["surface"], "sign_id": sign_id, "occurrences": len(occ),
                      "signers": n_signers, "mean_dtw": round(spread, 4), "medoid_sample": best["sid"]})
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    with open(OUT / "glosses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    report = [
        "# Isharah per-gloss clips", "",
        f"- Sentences used: {n_sentences:,}; glosses segmented: {sum(len(v) for v in segs.values()):,}",
        "- Validation rows below are measured on a random sample of 2,000 sentences.",
        f"- Distinct glosses: {len(segs)}; with ≥ {MIN_OCC} occurrences (lexicon entries): {len(entries)}", "",
        "| Cutting | same-gloss DTW | different-gloss DTW | ratio (lower = better) |", "|---|---|---|---|",
        *([f"| CTC forced alignment (used) | {same:.3f} | {diff:.3f} | {same / diff:.3f} |"] if args.alignments else []),
        f"| at hand-motion pauses | {same_p:.3f} | {diff_p:.3f} | {same_p / diff_p:.3f} |",
        f"| random cut points (control) | {same_c:.3f} | {diff_c:.3f} | {same_c / diff_c:.3f} |",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
