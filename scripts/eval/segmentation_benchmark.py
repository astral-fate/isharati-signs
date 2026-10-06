"""Continuous-signing segmentation benchmark (paper: results.tex, Table tab:segmentation_benchmark).

Every number in that table comes from this script and results/segmentation_benchmark.json.

A. Synthetic continuous streams with exact ground truth. Isolated ArSL dictionary clips (KArSL, the unified and
   the Kuwaiti dictionary, from data/lexicon_v2/lexicon_qa.jsonl) are joined with the app's own transition
   (isharati.pose.stitch._transition), 4-12 frames, with time-warp and coordinate noise, in two conditions:
     direct  the hands move straight from the end of one sign to the start of the next (no rest; closest to
             coarticulation), as the app's StitchPoser does;
     rest    the hands drop to a rest pose and hold it between signs.
   Each stream also starts and ends at rest. Metrics: boundary F1 (+-2/4/6 frames), segment F1 (IoU >= 0.5),
   stroke clipping, fragmentation, merging, count ratio, durations, with 95% bootstrap CIs over streams.
B. Real continuous corpora without timestamps: Isharah and ArabSign (gloss list per sentence): count agreement and
   same-gloss DTW consistency (with a random-cut control on the same sentences); EGCBT (Deaf Bible, silent) and
   YouTube-ASL: durations and segments per minute only (no ground truth).
C. Sanity check: the wrapper's old decoder (labels 1 and 2 taken as signing; 1 is O = rest) against the fixed one.

Methods: velocity minima (paper thresholds), uniform 1.4 s windows, the RoPE segmenter through the app wrapper
(isharati.pose.segmentation.segment_pose_array) and through the package's own preprocessing and decoder
(sign_language_segmentation.bin + metrics), on the same model weights.

Needs the project venv (torch, pytorch_lightning, sign_language_segmentation; mediapipe for prepare-egcbt):
  set PYTHONPATH=src & set OPENBLAS_NUM_THREADS=1
  D:/islam/isharati/.venv/Scripts/python scripts/eval/segmentation_benchmark.py prepare-isharah   # once, ~6 GB RAM
  D:/islam/isharati/.venv/Scripts/python scripts/eval/segmentation_benchmark.py prepare-egcbt     # once, MediaPipe
  D:/islam/isharati/.venv/Scripts/python scripts/eval/segmentation_benchmark.py run
Output: results/segmentation_benchmark.json, docs/paper/data/segmentation_benchmark.csv
"""
import argparse
import csv
import json
import platform
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lexicon" / "arsl" / "isharah"))
from isharati.config import DATA  # noqa: E402
from isharati.pose import segmentation as seg  # noqa: E402
from isharati.pose.dtw import dtw_distance, resample  # noqa: E402
from isharati.pose.keypoints import L_EL, L_SH, L_WR, LH, NECK, R_EL, R_SH, R_WR, RH  # noqa: E402
from isharati.pose.stitch import _transition  # noqa: E402

SEED = 20261005
FPS = 25
CACHE = DATA / "cache" / "segmentation_benchmark"
OUT_JSON = ROOT / "results" / "segmentation_benchmark.json"
OUT_CSV = ROOT / "docs" / "paper" / "data" / "segmentation_benchmark.csv"
ISHARAH = Path(r"D:\islam\ishara")
ARABSIGN_ZIP = Path(r"D:\islam\ArabSign\drive-download-20260930T221317Z-1-001.zip")
EGCBT_DIR = Path(r"E:\bible")
YTASL = DATA / "youtube_asl" / "poses" / "raw_keypoints_1.npz"
PKG_NORM = Path(r"D:\islam\isharati\.venv\Lib\site-packages\pose_anonymization\data\pose_normalization.json")

# --- synthetic setup
GROUPS = {"karsl": ["karsl"], "unified": ["unified_dictionary"], "kuwaiti": ["kuwaiti_dictionary"]}
STREAMS_PER_GROUP = 120          # x 3 groups x 2 conditions = 720 streams
CLIP_FRAMES = (15, 100)          # isolated clips kept: 0.6-4 s
TOLERANCES = (2, 4, 6)
NOISE_SD = 0.01                  # coordinate noise, shoulder widths
N_BOOT = 1000

# --- methods
VEL_RATIO, VEL_MIN_RUN = 0.15, 8     # paper: cut where speed < 0.15 max for >= 8 frames (0.32 s)
UNIFORM_S = 1.4                      # paper: fixed windows of 1.25-1.5 s
MIN_FRAMES = 4                       # shortest segment any method keeps (the wrapper's default)


# ============================================================ segmenters
def wrist_speed(pose):
    """Dominant wrist speed per frame, max of the two wrists (image-plane xy, shoulder widths per frame)."""
    v = np.linalg.norm(np.diff(pose[:, [R_WR, L_WR], :2], axis=0), axis=-1).max(axis=1)
    return np.concatenate([v[:1], v]) if len(v) else np.zeros(len(pose))


def runs(mask):
    """Maximal runs of True as (start, end) inclusive."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def seg_velocity(pose):
    """Velocity minima as the paper describes it: a pause is >= 8 frames below 0.15 x the peak speed; the signs are
    the stretches between pauses."""
    v = wrist_speed(pose)
    low = v < VEL_RATIO * v.max()
    pause = np.zeros(len(pose), bool)
    for a, b in runs(low):
        if b - a + 1 >= VEL_MIN_RUN:
            pause[a:b + 1] = True
    return [(a, b) for a, b in runs(~pause) if b - a + 1 >= MIN_FRAMES]


def seg_uniform(pose):
    w = int(round(UNIFORM_S * FPS))
    return [(a, min(a + w, len(pose)) - 1) for a in range(0, len(pose), w) if min(a + w, len(pose)) - a >= MIN_FRAMES]


def _spans(segs):
    return [(int(s["start"]), int(s["end"])) for s in segs]


def decode_old(log_probs):
    """The wrapper's decoder before 5 October 2026: labels 1 and 2 taken as signing. In the package 1 is O (rest)
    and 2 is B, so rest frames joined the signs and real signs (mostly I = 3) were dropped."""
    preds = np.asarray(log_probs).argmax(axis=1)
    return [{"start": a, "end": b} for a, b in runs(np.isin(preds, (1, 2)))]


_PKG = None


def _pkg_tables():
    """Per-joint mean/std of the package's input normalisation (pose_anonymization), in the order the model reads:
    pose_format reduce_holistic body points, then the left and right hand."""
    global _PKG
    if _PKG is None:
        norm = json.loads(PKG_NORM.read_text(encoding="utf-8"))
        body = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
                "LEFT_HIP", "RIGHT_HIP"]
        mean = [norm["POSE_LANDMARKS"][n]["mean"] for n in body]
        std = [norm["POSE_LANDMARKS"][n]["std"] for n in body]
        for comp in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            mean += [v["mean"] for v in norm[comp].values()]
            std += [v["std"] for v in norm[comp].values()]
        std = np.array(std, np.float32)
        std[std == 0] = 1
        _PKG = np.array(mean, np.float32), std
    return _PKG


def to_package_input(pose):
    """Isharati [T,50,3] -> the [T,50,6] input sign_language_segmentation.bin builds from a .pose file
    (utils.pose.preprocess_pose: reduce_holistic, hide legs, pose_anonymization normalize_mean_std, velocity).
    Steps: reorder joints; body wrist := hand wrist (correct_wrists); centre on the mean shoulder midpoint and scale
    by the mean shoulder width (Pose.normalize); hands relative to their wrist (shift_hands); per-joint z-score;
    hips (hidden legs, masked) = 0. Depth is set to the training mean (z-score 0): the Isharati contract's z is
    not on the scale of pose_format's z, and Isharah/YouTube-ASL have none."""
    mean, std = _pkg_tables()
    x = np.asarray(pose, np.float32)
    T = len(x)
    out = np.zeros((T, 50, 3), np.float32)
    out[:, 0:6] = x[:, [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]]
    out[:, 8:29] = x[:, LH]
    out[:, 29:50] = x[:, RH]
    out[:, 4] = out[:, 8]           # LEFT_WRIST := left-hand WRIST
    out[:, 5] = out[:, 29]          # RIGHT_WRIST := right-hand WRIST
    centre = ((out[:, 0] + out[:, 1]) / 2).mean(axis=0)
    width = np.linalg.norm(out[:, 0] - out[:, 1], axis=-1).mean()
    out = (out - centre) / max(float(width), 1e-6)
    out[:, 8:29] -= out[:, 8:9]
    out[:, 29:50] -= out[:, 29:30]
    out = (out - mean) / std
    out[:, 6:8] = 0.0
    out[..., 2] = 0.0
    t = np.arange(T, dtype=np.float32) / FPS
    vel = np.concatenate([np.zeros_like(out[:1]), np.diff(out, axis=0) / np.diff(t)[:, None, None]]) if T > 1 \
        else np.zeros_like(out)
    return np.concatenate([out, vel], axis=-1).astype(np.float32)


def run_all(pose, timing=None):
    """All segmenters on one [T,50,3] stream -> {method: [(start, end) inclusive]}."""
    from sign_language_segmentation.metrics import filter_segments as pkg_filter
    from sign_language_segmentation.metrics import likeliest_probs_to_segments as pkg_decode
    import torch

    pose = np.asarray(pose, np.float32)
    out = {"velocity": seg_velocity(pose), "uniform": seg_uniform(pose)}
    t0 = time.perf_counter()
    app = seg.segment_pose_array(pose)["signs"]
    dt_app = time.perf_counter() - t0
    out["rope_app"] = [(s["start_frame"], s["end_frame"]) for s in app]
    lp_app = seg.pose_log_probs(pose)["sign"]
    out["rope_app_old_decoder"] = _spans(seg.filter_segments(decode_old(lp_app), min_len=MIN_FRAMES, gap=1))
    t0 = time.perf_counter()
    x = to_package_input(pose)
    lp_pkg = seg.pose_log_probs(x)["sign"]
    dt_pkg = time.perf_counter() - t0
    # sign_language_segmentation.bin defaults: min_frames=3, merge_gap=0
    out["rope_pkg"] = _spans(pkg_filter(pkg_decode(torch.from_numpy(lp_pkg)), min_frames=3, merge_gap=0))
    out["rope_pkg_wrapper_decoder"] = _spans(seg.filter_segments(seg.likeliest_probs_to_segments(lp_pkg),
                                                                 min_len=MIN_FRAMES, gap=1))
    if timing is not None:
        timing["frames"] += len(pose)
        timing["rope_app_s"] += dt_app
        timing["rope_pkg_s"] += dt_pkg
    return out


METHODS = ["velocity", "uniform", "rope_app", "rope_pkg", "rope_pkg_wrapper_decoder", "rope_app_old_decoder"]
LABELS = {"velocity": "Velocity minima (paper thresholds)", "uniform": f"Uniform {UNIFORM_S:.2f} s windows",
          "rope_app": "RoPE, app wrapper (fixed decoder)", "rope_pkg": "RoPE, package preprocessing + decoder",
          "rope_pkg_wrapper_decoder": "RoPE, package preprocessing + wrapper decoder",
          "rope_app_old_decoder": "RoPE, app wrapper, OLD decoder (bug)",
          "pause_cut_known_n": "Pause cut, N = #glosses (oracle count)", "random_known_n": "Random cut, N = #glosses"}


# ============================================================ A. synthetic streams
def load_pool():
    pool = defaultdict(list)
    for line in (DATA / "lexicon_v2" / "lexicon_qa.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        group = next((g for g, ds in GROUPS.items() if e["dataset"] in ds), None)
        if group is None:
            continue
        path = next((p for p in (DATA / "lexicon_v2" / e["keypoints_path"], DATA / "lexicon" / e["keypoints_path"])
                     if p.exists()), None)
        if path is None:
            continue
        p = np.load(path).astype(np.float32)
        if not (CLIP_FRAMES[0] <= len(p) <= CLIP_FRAMES[1]) or not np.isfinite(p).all():
            continue
        spread = [float(np.ptp(p[:, h, :2], axis=1).max()) for h in (LH, RH)]
        travel = float(np.linalg.norm(np.diff(p[:, [R_WR, L_WR], :2], axis=0), axis=-1).sum(0).max())
        if max(spread) < 0.05 or travel < 0.2:   # a hand never seen, or no movement
            continue
        pool[group].append({"id": e["sign_id"], "gloss": e["gloss"], "pose": p})
    return pool


def rest_pose(frame):
    """Hands lowered to the waist below the shoulders, hand shapes kept."""
    r = frame.copy()
    for sh, el, wr, hand in ((R_SH, R_EL, R_WR, RH), (L_SH, L_EL, L_WR, LH)):
        sx = frame[sh, 0]
        wrist = np.array([0.8 * sx, 1.4, frame[wr, 2]], np.float32)
        r[el] = np.array([1.1 * sx, 0.7, frame[el, 2]], np.float32)
        r[hand] = frame[hand] - frame[hand][0] + wrist
        r[wr] = wrist
    return r


def make_stream(clips, condition, rng):
    """Join clips with the app's transition; returns pose [T,50,3] and gold sign spans (inclusive)."""
    parts, spans, t = [], [], 0

    def add(x):
        nonlocal t
        parts.append(x.astype(np.float32))
        t += len(x)

    warped = []
    for c in clips:
        n = max(CLIP_FRAMES[0] // 2, int(round(len(c) * rng.uniform(0.85, 1.15))))
        warped.append(resample(c, n))
    rest0 = rest_pose(warped[0][0])
    add(np.repeat(rest0[None], rng.integers(8, 16), axis=0))
    add(_transition(rest0, warped[0][0], 8))
    for i, c in enumerate(warped):
        if i > 0:
            prev = parts[-1][-1]
            if condition == "direct":
                add(_transition(prev, c[0], int(rng.integers(4, 13))))
            else:
                r = rest_pose(prev)
                add(_transition(prev, r, int(rng.integers(4, 13))))
                add(np.repeat(r[None], rng.integers(3, 11), axis=0))
                add(_transition(r, c[0], int(rng.integers(4, 13))))
        spans.append((t, t + len(c) - 1))
        add(c)
    r = rest_pose(parts[-1][-1])
    add(_transition(parts[-1][-1], r, 8))
    add(np.repeat(r[None], rng.integers(8, 16), axis=0))
    pose = np.concatenate(parts)
    pose = pose + rng.normal(0, NOISE_SD, pose.shape).astype(np.float32)
    return pose, spans


def build_streams(pool):
    rng = np.random.default_rng(SEED)
    streams = []
    for condition in ("direct", "rest"):
        for group in GROUPS:
            items = pool[group]
            for k in range(STREAMS_PER_GROUP):
                n = int(rng.integers(3, 9))
                pick = rng.choice(len(items), size=n, replace=False)
                pose, spans = make_stream([items[j]["pose"] for j in pick], condition, rng)
                streams.append({"condition": condition, "group": group, "pose": pose, "gold": spans,
                                "signs": [items[j]["id"] for j in pick]})
    return streams


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def greedy_match(scores):
    """scores: list of (score, i, j) for admissible pairs, higher = better -> number of one-to-one matches."""
    used_i, used_j, n = set(), set(), 0
    for _, i, j in sorted(scores, reverse=True):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            n += 1
    return n


def stream_stats(gold, pred):
    """Counts for one stream (summed over streams for micro-averaged metrics)."""
    st = {"n_gold": len(gold), "n_pred": len(pred)}
    gb = sorted({b for s, e in gold for b in (s, e + 1)})
    pb = sorted({b for s, e in pred for b in (s, e + 1)})
    st["n_gold_b"], st["n_pred_b"] = len(gb), len(pb)
    for tol in TOLERANCES:
        pairs = [(-abs(g - p), i, j) for i, g in enumerate(gb) for j, p in enumerate(pb) if abs(g - p) <= tol]
        st[f"tp_b{tol}"] = greedy_match(pairs)
    pairs = [(iou(g, p), i, j) for i, g in enumerate(gold) for j, p in enumerate(pred) if iou(g, p) >= 0.5]
    st["tp_seg"] = greedy_match(pairs)
    clipped = frag = merged = 0
    for k, (s, e) in enumerate(gold):
        L = e - s + 1
        cs, ce = s + int(0.2 * L), e - int(0.2 * L)
        if not any(a <= cs and b >= ce for a, b in pred):
            clipped += 1
        ov = [min(e, b) - max(s, a) + 1 for a, b in pred]
        if sum(o >= max(2, 0.2 * L) for o in ov) >= 2:
            frag += 1
        if ov and max(ov) > 0:
            a, b = pred[int(np.argmax(ov))]
            for k2, (s2, e2) in enumerate(gold):
                if k2 != k and min(e2, b) - max(s2, a) + 1 >= 0.5 * (e2 - s2 + 1):
                    merged += 1
                    break
    st.update(clipped=clipped, fragmented=frag, merged=merged)
    st["dur"] = [(b - a + 1) / FPS for a, b in pred]
    return st


def summarise(stats):
    """Micro-averaged metrics from a list of per-stream stats."""
    S = lambda k: sum(s[k] for s in stats)
    out = {}
    for tol in TOLERANCES:
        p = S(f"tp_b{tol}") / max(S("n_pred_b"), 1)
        r = S(f"tp_b{tol}") / max(S("n_gold_b"), 1)
        out[f"boundary_f1_{tol}"] = 2 * p * r / (p + r) if p + r else 0.0
        out[f"boundary_p_{tol}"], out[f"boundary_r_{tol}"] = p, r
    p, r = S("tp_seg") / max(S("n_pred"), 1), S("tp_seg") / max(S("n_gold"), 1)
    out["segment_f1_iou50"] = 2 * p * r / (p + r) if p + r else 0.0
    ng = max(S("n_gold"), 1)
    out["clipping_rate"] = S("clipped") / ng
    out["fragmentation_rate"] = S("fragmented") / ng
    out["merge_rate"] = S("merged") / ng
    ratios = [s["n_pred"] / s["n_gold"] for s in stats]
    out["count_ratio_mean"] = float(np.mean(ratios))
    out["over_segmented_streams"] = float(np.mean([s["n_pred"] > s["n_gold"] for s in stats]))
    out["under_segmented_streams"] = float(np.mean([s["n_pred"] < s["n_gold"] for s in stats]))
    d = [x for s in stats for x in s["dur"]]
    out["duration_mean_s"] = float(np.mean(d)) if d else 0.0
    out["duration_sd_s"] = float(np.std(d)) if d else 0.0
    out["share_over_4s"] = float(np.mean(np.array(d) > 4.0)) if d else 0.0
    return out


def bootstrap(stats, seed):
    rng = np.random.default_rng(seed)
    point = summarise(stats)
    boots = defaultdict(list)
    for _ in range(N_BOOT):
        sample = [stats[i] for i in rng.integers(0, len(stats), len(stats))]
        for k, v in summarise(sample).items():
            boots[k].append(v)
    return {k: {"value": v, "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]}
            for k, v in point.items()}


def part_a(timing):
    pool = load_pool()
    print("pool:", {g: len(v) for g, v in pool.items()}, flush=True)
    streams = build_streams(pool)
    per = defaultdict(list)            # (condition, method) -> stats
    gold_dur = defaultdict(list)
    for n, s in enumerate(streams):
        preds = run_all(s["pose"], timing)
        for m in METHODS:
            st = stream_stats(s["gold"], preds[m])
            st["group"] = s["group"]
            per[(s["condition"], m)].append(st)
        gold_dur[s["condition"]] += [(e - b + 1) / FPS for b, e in s["gold"]]
        if n % 60 == 0:
            print(f"  stream {n}/{len(streams)}", flush=True)
    res = {"setup": {
        "pool_clips": {g: len(v) for g, v in pool.items()}, "streams": len(streams),
        "streams_per_condition": len(streams) // 2, "signs_per_stream": [3, 8],
        "transition_frames": [4, 12], "rest_hold_frames": [3, 10], "time_warp": [0.85, 1.15],
        "noise_sd": NOISE_SD, "clip_frames": list(CLIP_FRAMES), "n_boot": N_BOOT, "seed": SEED,
        "gold_signs": {c: len(v) for c, v in gold_dur.items()},
        "gold_duration_s": {c: [float(np.mean(v)), float(np.std(v))] for c, v in gold_dur.items()},
        "stream_seconds": {c: [float(np.mean([len(s["pose"]) / FPS for s in streams if s["condition"] == c]))]
                           for c in ("direct", "rest")}}}
    for (cond, m), stats in per.items():
        res.setdefault(cond, {})[m] = bootstrap(stats, SEED)
        res[cond][m]["by_group"] = {g: {k: v for k, v in summarise([s for s in stats if s["group"] == g]).items()
                                        if k in ("boundary_f1_4", "segment_f1_iou50")} for g in GROUPS}
    return res


# ============================================================ B. real corpora
def prepare_isharah(n=500):
    import pickle
    from gloss_clips import to_contract
    glosses = {}
    for split in ("train.csv", "dev.csv"):
        for line in (ISHARAH / split).read_text(encoding="utf-8").splitlines()[1:]:
            if "|" in line:
                sid, g = line.split("|", 1)
                glosses[sid] = g.split()
    pkl = ISHARAH / "pose_data_isharah2000_hands_lips_body_phase1_SI.pkl"
    print("loading", pkl.name, flush=True)
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    sids = sorted(s for s in data if s in glosses)
    pick = random.Random(SEED).sample(sids, n)
    out = {}
    for sid in pick:
        try:
            out[sid] = to_contract(data[sid]["keypoints"]).astype(np.float16)
        except ValueError:
            pass
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "isharah_sample.npz", **out)
    (CACHE / "isharah_sample_glosses.json").write_text(
        json.dumps({s: glosses[s] for s in out}, ensure_ascii=False), encoding="utf-8")
    print("isharah sample:", len(out), "of", len(sids), flush=True)


def load_isharah():
    d = np.load(CACHE / "isharah_sample.npz")
    g = json.loads((CACHE / "isharah_sample_glosses.json").read_text(encoding="utf-8"))
    return [{"id": s, "signer": s.split("_")[0], "pose": d[s].astype(np.float32), "glosses": g[s]} for s in d.files]


def load_arabsign(per_cell=1):
    text = zipfile.ZipFile(ARABSIGN_ZIP).read("ArabSignGroundTruth.txt").decode("utf-16")
    ref = {}
    for line in text.splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 3 and p[2].strip():
            ref[p[0].strip()] = p[2].split()       # whitespace separates signs; '-' joins a multi-word sign
    rows = list(csv.DictReader(open(DATA / "ar_arabsign" / "index.csv", encoding="utf-8")))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["signer"], r["sentence"])].append(r)
    rng = random.Random(SEED)
    out = []
    for key in sorted(cells):
        if key[1] not in ref:
            continue
        for r in rng.sample(cells[key], min(per_cell, len(cells[key]))):
            p = np.load(DATA / "ar_arabsign" / r["path"]).astype(np.float32)
            out.append({"id": r["clip"], "signer": r["signer"], "pose": p, "glosses": ref[key[1]]})
    return out


def prepare_egcbt(n_videos=6, window_s=60.0):
    """Pose for a 60 s window (from 30% of the running time) of n_videos EGCBT videos, MediaPipe Holistic as the
    app extracts (isharati.pose.keypoints), resampled to 25 fps."""
    import cv2
    from isharati.pose.keypoints import Holistic, frame_from_holistic, interpolate_missing, normalize
    vids = []
    for f in sorted(EGCBT_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(f))
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and n:
            vids.append((f, fps, n / fps))
    step = max(1, len(vids) // n_videos)
    CACHE.mkdir(parents=True, exist_ok=True)
    out, meta = {}, {}
    for f, fps, dur in vids[::step][:n_videos]:
        t0 = 0.3 * dur
        cap = cv2.VideoCapture(str(f))
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        raw, i, kept = [], 0, -1
        with Holistic() as holo:
            while i < window_s * fps:
                ok = cap.grab()
                if not ok:
                    break
                k = int(i * FPS / fps)
                if k > kept:
                    kept = k
                    _, img = cap.retrieve()
                    img = cv2.resize(img, (960, 540))
                    raw.append(frame_from_holistic(holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), i * 1000 / fps)))
                i += 1
        cap.release()
        raw = np.stack(raw)
        filled, missing = interpolate_missing(raw)
        out[f.stem] = normalize(filled).astype(np.float16)
        meta[f.stem] = {"start_s": round(t0, 1), "frames": len(raw), "dominant_hand_missing": round(missing, 3)}
        print(f.stem, meta[f.stem], flush=True)
    np.savez_compressed(CACHE / "egcbt_sample.npz", **out)
    (CACHE / "egcbt_sample.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load_youtube_asl(n=300, src_fps=30.0):
    rng = random.Random(SEED)
    d = np.load(YTASL)
    keys = sorted(d.files)
    out = []
    from isharati.pose.keypoints import interpolate_missing
    for k in rng.sample(keys, n * 3):
        p = d[k].astype(np.float32)
        m = max(1, int(round(len(p) * FPS / src_fps)))
        p = p[np.linspace(0, len(p) - 1, m).round().astype(int)]
        if np.isnan(p[:, R_WR]).all() and np.isnan(p[:, L_WR]).all():
            continue
        p, missing = interpolate_missing(p)      # undetected hands are NaN, as at extraction
        if len(p) >= 2 * FPS and missing <= 0.2:
            out.append({"id": k, "pose": p})
        if len(out) == n:
            break
    return out


def known_n_cuts(pose, n, rng=None):
    from gloss_clips import cut
    parts = cut(pose, n, rng)
    return [(a, b - 1) for a, b in parts] if parts else None


def separation(segs, pairs, seed):
    """Same-gloss vs different-gloss DTW distance over segments {gloss, sid, clip}; pairs drawn from different
    sentences. Returns ratio with a bootstrap CI over pairs."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for s in segs:
        by[s["gloss"]].append(s)
    multi = [v for v in by.values() if len({s["sid"] for s in v}) >= 2]
    if not multi:
        return None
    same, diff = [], []
    for _ in range(pairs * 3):
        if len(same) >= pairs:
            break
        v = rng.choice(multi)
        a, b = rng.sample(v, 2)
        c = rng.choice(segs)
        if a["sid"] == b["sid"] or c["gloss"] == a["gloss"] or c["sid"] == a["sid"]:
            continue
        same.append(dtw_distance(a["clip"], b["clip"]))
        diff.append(dtw_distance(a["clip"], c["clip"]))
    same, diff = np.array(same), np.array(diff)
    brng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        i = brng.integers(0, len(same), len(same))
        boots.append(same[i].mean() / diff[i].mean())
    return {"ratio": float(same.mean() / diff.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                                 float(np.percentile(boots, 97.5))],
            "same": float(same.mean()), "diff": float(diff.mean()), "pairs": int(len(same))}


def part_b_gloss(name, sents, timing, dtw_pairs=400):
    preds = {}
    for n, s in enumerate(sents):
        p = run_all(s["pose"], timing)
        p["pause_cut_known_n"] = known_n_cuts(s["pose"], len(s["glosses"])) or []
        preds[s["id"]] = p
        if n % 100 == 0:
            print(f"  {name} {n}/{len(sents)}", flush=True)
    res = {"sentences": len(sents), "signers": len({s["signer"] for s in sents}),
           "glosses_per_sentence": [float(np.mean([len(s["glosses"]) for s in sents])),
                                    float(np.std([len(s["glosses"]) for s in sents]))],
           "sentence_s": [float(np.mean([len(s["pose"]) / FPS for s in sents])),
                          float(np.std([len(s["pose"]) / FPS for s in sents]))], "methods": {}}
    for m in METHODS + ["pause_cut_known_n"]:
        diffs = np.array([len(preds[s["id"]][m]) - len(s["glosses"]) for s in sents])
        dur = [(b - a + 1) / FPS for s in sents for a, b in preds[s["id"]][m]]
        matched = [s for s in sents if len(preds[s["id"]][m]) == len(s["glosses"])]
        r = {"abs_count_error_mean": float(np.abs(diffs).mean()), "count_diff_mean": float(diffs.mean()),
             "exact_count_rate": float((diffs == 0).mean()), "segments_per_sentence": float(
                 np.mean([len(preds[s["id"]][m]) for s in sents])),
             "duration_mean_s": float(np.mean(dur)) if dur else 0.0, "duration_sd_s": float(np.std(dur)) if dur else 0.0,
             "duration_median_s": float(np.median(dur)) if dur else 0.0,
             "duration_p90_s": float(np.percentile(dur, 90)) if dur else 0.0,
             "share_over_4s": float(np.mean(np.array(dur) > 4.0)) if dur else 0.0,
             "count_matched_sentences": len(matched)}
        dm = [(b - a + 1) / FPS for s in matched for a, b in preds[s["id"]][m]]
        r["duration_mean_s_count_matched"] = float(np.mean(dm)) if dm else 0.0
        if m != "rope_app_old_decoder" and len(matched) >= 20:
            segs = [{"gloss": g, "sid": s["id"], "clip": s["pose"][a:b + 1]}
                    for s in matched for g, (a, b) in zip(s["glosses"], preds[s["id"]][m])]
            r["dtw"] = separation(segs, dtw_pairs, SEED)
            rng = random.Random(SEED)
            ctrl = []
            for s in matched:
                cuts = known_n_cuts(s["pose"], len(s["glosses"]), rng)
                if cuts:
                    ctrl += [{"gloss": g, "sid": s["id"], "clip": s["pose"][a:b + 1]}
                             for g, (a, b) in zip(s["glosses"], cuts)]
            r["dtw_random_control_same_sentences"] = separation(ctrl, dtw_pairs, SEED)
        res["methods"][m] = r
    return res


def part_b_free(name, items, timing):
    preds = []
    for n, s in enumerate(items):
        preds.append((len(s["pose"]), run_all(s["pose"], timing)))
        if n % 100 == 0:
            print(f"  {name} {n}/{len(items)}", flush=True)
    minutes = sum(T for T, _ in preds) / FPS / 60
    res = {"items": len(items), "minutes": minutes, "methods": {}}
    for m in METHODS:
        dur = [(b - a + 1) / FPS for _, p in preds for a, b in p[m]]
        covered = sum(sum(b - a + 1 for a, b in p[m]) for _, p in preds) / sum(T for T, _ in preds)
        res["methods"][m] = {"segments_per_min": len(dur) / minutes, "duration_mean_s": float(np.mean(dur)) if dur
                             else 0.0, "duration_sd_s": float(np.std(dur)) if dur else 0.0,
                             "duration_median_s": float(np.median(dur)) if dur else 0.0,
                             "duration_p90_s": float(np.percentile(dur, 90)) if dur else 0.0,
                             "share_over_4s": float(np.mean(np.array(dur) > 4.0)) if dur else 0.0,
                             "frames_covered": covered}
    return res


# ============================================================ output
def write_csv(res):
    rows = []
    for cond in ("direct", "rest"):
        for m in METHODS:
            r = res["synthetic"][cond][m]
            f = lambda k: f"{r[k]['value']:.3f}"
            ci = lambda k: f"{r[k]['ci95'][0]:.3f}-{r[k]['ci95'][1]:.3f}"
            rows.append({"part": "A_synthetic", "corpus": f"synthetic_{cond}", "method": m,
                         "boundary_f1_2": f("boundary_f1_2"), "boundary_f1_4": f("boundary_f1_4"),
                         "boundary_f1_4_ci": ci("boundary_f1_4"), "boundary_f1_6": f("boundary_f1_6"),
                         "segment_f1": f("segment_f1_iou50"), "segment_f1_ci": ci("segment_f1_iou50"),
                         "clipping": f("clipping_rate"), "fragmentation": f("fragmentation_rate"),
                         "merge": f("merge_rate"), "count_ratio": f("count_ratio_mean"),
                         "duration_mean_s": f("duration_mean_s"), "duration_sd_s": f("duration_sd_s")})
    for corpus in ("isharah", "arabsign"):
        for m, r in res["real"][corpus]["methods"].items():
            rows.append({"part": "B_real_glosses", "corpus": corpus, "method": m,
                         "abs_count_error": f"{r['abs_count_error_mean']:.2f}",
                         "exact_count_rate": f"{r['exact_count_rate']:.3f}",
                         "dtw_ratio": f"{r['dtw']['ratio']:.3f}" if r.get("dtw") else "",
                         "dtw_random_control": f"{r['dtw_random_control_same_sentences']['ratio']:.3f}"
                         if r.get("dtw_random_control_same_sentences") else "",
                         "duration_mean_s": f"{r['duration_mean_s']:.3f}", "duration_sd_s": f"{r['duration_sd_s']:.3f}"})
    for corpus in ("egcbt", "youtube_asl"):
        if corpus not in res["real"]:
            continue
        for m, r in res["real"][corpus]["methods"].items():
            rows.append({"part": "B_no_ground_truth", "corpus": corpus, "method": m,
                         "segments_per_min": f"{r['segments_per_min']:.1f}",
                         "duration_mean_s": f"{r['duration_mean_s']:.3f}", "duration_sd_s": f"{r['duration_sd_s']:.3f}"})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def audio_volume(path, start_s, dur_s=60.0):
    """ffmpeg volumedetect over the window: (mean dB, max dB), or None without ffmpeg."""
    import re
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return None
    err = subprocess.run(["ffmpeg", "-hide_banner", "-ss", str(start_s), "-t", str(dur_s), "-i", str(path), "-vn",
                          "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
    mean, mx = re.search(r"mean_volume: (-?[\d.]+)", err), re.search(r"max_volume: (-?[\d.]+)", err)
    return [float(mean.group(1)), float(mx.group(1))] if mean and mx else None


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    t_start = time.time()
    timing = {"frames": 0, "rope_app_s": 0.0, "rope_pkg_s": 0.0}
    seg.segment_pose_array(np.zeros((30, 50, 3), np.float32))       # load the model outside the timing
    res = {"script": "scripts/eval/segmentation_benchmark.py", "seed": SEED, "fps": FPS,
           "methods": {m: LABELS[m] for m in METHODS + ["pause_cut_known_n", "random_known_n"]},
           "params": {"velocity_ratio": VEL_RATIO, "velocity_min_pause_frames": VEL_MIN_RUN, "uniform_s": UNIFORM_S,
                      "min_frames": MIN_FRAMES, "rope_app": "segment_pose_array(min_frames=4, merge_gap=1)",
                      "rope_pkg": "bin defaults: min_frames=3, merge_gap=0", "model_dir": str(seg._MODEL_DIR)}}
    print("A. synthetic", flush=True)
    res["synthetic"] = part_a(timing)
    print("B. real corpora", flush=True)
    res["real"] = {"isharah": part_b_gloss("isharah", load_isharah(), timing),
                   "arabsign": part_b_gloss("arabsign", load_arabsign(), timing)}
    if (CACHE / "egcbt_sample.npz").exists():
        d = np.load(CACHE / "egcbt_sample.npz")
        items = [{"id": k, "pose": d[k].astype(np.float32)} for k in d.files]
        res["real"]["egcbt"] = part_b_free("egcbt", items, timing)
        windows = json.loads((CACHE / "egcbt_sample.json").read_text(encoding="utf-8"))
        for stem, w in windows.items():
            w["audio_mean_max_db"] = audio_volume(EGCBT_DIR / f"{stem}.mp4", w["start_s"])
        res["real"]["egcbt"]["windows"] = windows
        res["real"]["egcbt"]["note"] = "no ground truth: silent Deaf Bible videos without timestamps or gloss lists"
    if YTASL.exists():
        res["real"]["youtube_asl"] = part_b_free("youtube_asl", load_youtube_asl(), timing)
        res["real"]["youtube_asl"]["note"] = ("no ground truth: caption-level clips (English captions, no glosses), "
                                              "part 1, assumed 30 fps resampled to 25")
    res["throughput"] = {"cpu": cpu_name(), "torch_threads": args.threads, "frames": timing["frames"],
                         "rope_app_fps": timing["frames"] / timing["rope_app_s"],
                         "rope_pkg_fps": timing["frames"] / timing["rope_pkg_s"],
                         "note": "model inference + wrapper pre/post-processing per stream, batch size 1, CPU; "
                                 "other jobs shared the machine"}
    res["runtime_s"] = round(time.time() - t_start, 1)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    write_csv(res)
    print("wrote", OUT_JSON, OUT_CSV, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare-isharah", "prepare-egcbt", "run"])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "prepare-isharah":
        prepare_isharah()
    elif args.cmd == "prepare-egcbt":
        prepare_egcbt()
    else:
        run(args)


if __name__ == "__main__":
    main()
