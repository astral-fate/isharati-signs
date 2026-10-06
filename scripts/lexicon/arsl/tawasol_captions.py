"""Step 1 of cutting a Tawasol vocabulary lesson: find its caption intervals and tile one numbered crop per caption,
so the terms can be read in order and typed into data/lexicon_v2/tawasol/terms/<video>.json (a list of strings,
one per numbered crop, "" for a crop to skip such as a title card).

Captions are white text on a green background. The signer's white thobe is white in almost every frame, so a pixel
counts as caption only if it is white now but not white in most frames (the persistent mask). The white-caption
share over time is smoothed and thresholded; intervals shorter than 0.4 s are dropped, and so is anything after the
signer leaves (the end card).
  .venv/Scripts/python scripts/lexicon/arsl/tawasol_captions.py 102 103 ...
Output: data/lexicon_v2/tawasol/captions/<video>.json (intervals) and <video>.jpg (numbered crops)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
SRC = Path(r"D:\islam\gathring data\tawasol_videos")
OUT = DATA / "lexicon_v2" / "tawasol" / "captions"
STEP_FPS = 5


def video_path(num):
    return next(SRC.glob(f"{num} - */*.mp4"))


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(fps / STEP_FPS)))
    frames, times, i = [], [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(cv2.resize(img, (640, 360)))
            times.append(i / fps)
        i += 1
    cap.release()
    return frames, np.array(times)


def intervals(frames, times):
    white = np.stack([f.min(axis=2) > 190 for f in frames])            # T,H,W
    greenish = np.stack([(f[..., 1] > f[..., 2] + 30) & (f[..., 1] > f[..., 0] + 30) for f in frames])
    scene = greenish.mean(axis=(1, 2)) > 0.25                             # the lesson set, not the logo cards
    persistent = cv2.dilate((white[scene].mean(axis=0) > 0.35).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    # text strokes are thin; a sleeve or hand moving out of the thobe area is a thick white blob: remove those
    k7, k5 = np.ones((7, 7), np.uint8), np.ones((5, 5), np.uint8)
    thick = np.stack([cv2.dilate(cv2.morphologyEx(w.astype(np.uint8), cv2.MORPH_OPEN, k7), k5) > 0 for w in white])
    cap = white & ~persistent[None] & ~thick
    share = cap.mean(axis=(1, 2)) * scene
    share = np.convolve(share, np.ones(3) / 3, "same")
    thr = max(0.0015, 0.35 * np.quantile(share[scene], 0.9)) if scene.any() else 1.0
    on = share > thr
    # one caption replacing another with no gap: split where the caption pixels stop overlapping the previous ones
    change = np.zeros(len(on), bool)
    for k in range(1, len(on)):
        if on[k] and on[k - 1]:
            a, b = cv2.dilate(cap[k].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, cap[k - 1]
            inter = (a & b).sum()
            change[k] = inter < 0.35 * max(1, min(cap[k].sum(), b.sum()))
    out, k = [], 0
    while k < len(on):
        if on[k]:
            j = k + 1
            while j < len(on) and on[j] and not change[j]:
                j += 1
            if times[j - 1] - times[k] >= 0.4:
                out.append((float(times[k]), float(times[j - 1])))
            k = j
        else:
            k += 1
    # a caption fading out and back in is one caption: merge neighbours whose crops overlap strongly
    return out, cap


def crop_sheet(frames, times, spans, cap, path):
    tiles = []
    for n, (a, b) in enumerate(spans):
        idx = int(np.argmin(np.abs(times - (a + b) / 2)))
        ys, xs = np.nonzero(cap[idx])
        f = frames[idx]
        if len(xs):
            x0, x1 = max(0, xs.min() - 8), min(f.shape[1], xs.max() + 8)
            y0, y1 = max(0, ys.min() - 6), min(f.shape[0], ys.max() + 6)
            c = f[y0:y1, x0:x1]
        else:
            c = f
        c = cv2.resize(c, (360, max(24, min(120, int(c.shape[0] * 360 / max(1, c.shape[1]))))))
        tile = np.full((130, 420, 3), 40, np.uint8)
        tile[:c.shape[0], 60:60 + c.shape[1]] = c
        cv2.putText(tile, str(n), (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 40))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for num in sys.argv[1:]:
        frames, times = read_frames(video_path(num))
        spans, cap = intervals(frames, times)
        (OUT / f"{num}.json").write_text(json.dumps(spans), encoding="utf-8")
        if spans:
            crop_sheet(frames, times, spans, cap, OUT / f"{num}.jpg")
        print(num, len(spans), "captions")


if __name__ == "__main__":
    main()
