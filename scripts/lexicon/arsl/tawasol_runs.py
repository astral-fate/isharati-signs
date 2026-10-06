"""Tawasol vocabulary lessons -> Arabic signs, one signing stretch at a time.

The unit is the signing stretch (a hand raised above waist level, scripts/lexicon/asl/islamic_terms.signing_runs), which in
these lessons is one term: the signer rests with clasped hands between terms. Steps:
  extract  MediaPipe Holistic on each lesson (cached as data/lexicon_v2/tawasol/raw_<video>.npz)
  sheet    one numbered tile per stretch (its middle frame, caption visible) -> tawasol/runs/<video>.jpg
  cut      read tawasol/runs/<video>.json, a list aligned with the tiles: a term, "" to skip the stretch (title,
           greeting), or "+" to join it to the previous term (one sign split by a short dip) -> signs + entries
  .venv/Scripts/python scripts/lexicon/arsl/tawasol_runs.py extract|sheet|cut <video> ...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "lexicon" / "asl"))
from islamic_terms import signing_runs  # noqa: E402

from isharati.pose.keypoints import frame_from_holistic, interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

SRC = Path(r"D:\islam\gathring data\tawasol_videos")
OUT = DATA / "lexicon_v2" / "tawasol"
RUNS = OUT / "runs"
MIN_FRAMES = 12  # 0.5 s: shorter is a partly detected sign, not used

# the general vocabulary lessons (one of each where the channel uploaded a lesson twice)
EXTRA = ["008", "110", "112", "119", "124", "130", "132", "133", "139", "075", "077", "152"]  # second takes, Hajj, uncaptioned
LESSONS = ["002", "003", "004", "005", "006", "007", "009", "083", "084", "085", "086", "087", "088", "089", "101",
           "102", "103", "108", "111", "113", "114", "115", "116", "117", "118", "120", "121", "123", "125", "131",
           "134", "136", "137", "138", "140", "147", "151", "152", "153"]


def video_path(num):
    return next(SRC.glob(f"{num} - */*.mp4"))


def extract(num):
    cache = OUT / f"raw_{num}.npz"
    if cache.exists():
        return
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video_path(num)))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = set(np.linspace(0, n_src - 1, int(round(n_src * FPS / src_fps))).round().astype(int).tolist())
    raw, times, i = [], [], 0
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holo:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in picks:
                raw.append(frame_from_holistic(holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                times.append(i / src_fps)
            i += 1
    cap.release()
    np.savez_compressed(cache, raw=np.stack(raw), t=np.array(times))


def load(num):
    z = np.load(OUT / f"raw_{num}.npz")
    return z["raw"], z["t"]


def sheet(num):
    raw, t = load(num)
    runs = signing_runs(raw)
    cap, tiles = cv2.VideoCapture(str(video_path(num))), []
    for n, (s, e) in enumerate(runs):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t[(s + e) // 2]) * 1000)
        ok, img = cap.read()
        tile = cv2.resize(img, (480, 270)) if ok else np.zeros((270, 480, 3), np.uint8)
        cv2.rectangle(tile, (0, 0), (70, 44), (0, 0, 0), -1)
        cv2.putText(tile, str(n), (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, f"{t[s]:.1f}s", (380, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cap.release()
    RUNS.mkdir(parents=True, exist_ok=True)
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(RUNS / f"{num}.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    print(num, len(runs), "stretches")


def cut(num):
    raw, t = load(num)
    runs = signing_runs(raw)
    labels = json.loads((RUNS / f"{num}.json").read_text(encoding="utf-8"))
    if len(labels) != len(runs):
        raise SystemExit(f"{num}: {len(labels)} labels for {len(runs)} stretches")
    spans = []  # (gloss, start, end)
    for lab, (s, e) in zip(labels, runs):
        if lab == "+" and spans:
            g, s0, _ = spans[-1]
            spans[-1] = (g, s0, e)
        elif lab and lab != "+":
            spans.append((lab, s, e))
    entries_path = OUT / "entries.jsonl"
    entries = {e["sign_id"]: e for e in (json.loads(l) for l in entries_path.read_text(encoding="utf-8").splitlines()
                                         if l.strip())} if entries_path.exists() else {}
    for k in [k for k in entries if k.startswith(f"taw{num}_")]:
        del entries[k]
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    kept = 0
    for n, (gloss, s, e) in enumerate(spans):
        if e - s < MIN_FRAMES:
            print(f"  {num} {gloss}: too short ({e - s} frames), not used")
            continue
        sid = f"taw{num}_{n:02d}"
        filled, _ = interpolate_missing(raw[s:e])
        np.save(OUT / "signs" / f"{sid}.npy", normalize(filled))
        entries[sid] = {"gloss": gloss, "sign_id": sid, "dataset": "tawasol", "keypoints_path":
                        f"tawasol/signs/{sid}.npy", "review_status": "pending", "is_religious": False,
                        "is_letter": num == "134"}
        kept += 1
    entries_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries.values()), encoding="utf-8")
    print(f"{num}: {kept} signs")


if __name__ == "__main__":
    mode, nums = sys.argv[1], sys.argv[2:] or LESSONS
    for num in nums:
        {"extract": extract, "sheet": sheet, "cut": cut}[mode](num)
        if mode == "extract":
            print(num, "extracted", flush=True)
