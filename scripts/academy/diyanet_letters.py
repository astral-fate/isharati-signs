"""The Arabic alphabet in sign for the Academy, from Diyanet's «İşaret Dili ile Kur'an», episode 30
(«Arap İşaret Dili Parmak Alfabesi», https://youtu.be/1lTFjbvRnO4).

The episode shows one letter at a time on screen, with its handshape drawn beside it, while the signer signs it. The
letter panel's changes (sampled at 5 fps) split the episode into 28 spans in alphabetical order; inside each span the
sign is the stretch where a hand is raised above the resting hands (with a few frames of margin). The video is tracked
with the Holistic task (scripts/face/track.py), so each letter's face comes from the same frames.

  py scripts/academy/diyanet_letters.py
Output: <hub_build>/app_data_staging/diyanet_letters.npz, published by scripts/hub/publish_app_data.py — per letter l01..l28: <id>/pose [T,50,3] float16 (normalised
as every lexicon clip), <id>/blend [T,52], <id>/face [T,160,3] float16 (face contour points in the same frame).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "face"), str(ROOT / "scripts" / "lexicon" / "arsl")]
from attach import fifty, window_frame  # noqa: E402
from isharati.pose import proportions  # noqa: E402
from isharati.pose.face import POINTS  # noqa: E402
from reviewed_recordings import rest_undetected_hands  # noqa: E402

VIDEO = Path(r"F:\tid_quran_videos\lessons\30_1lTFjbvRnO4.mp4")
TRACK = VIDEO.with_suffix(".track.npz")
# staged for the isharati-app-data dataset (scripts/hub/publish_app_data.py --letters <this file>); the app reads it from the Hub
OUT = Path(r"D:\islam\hub_build") / "app_data_staging" / "diyanet_letters.npz"
FPS = 25
LETTERS = "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي".split()


def letter_spans():
    """[(start_s, end_s)] of the 28 letter panels, from the text-panel changes."""
    c = cv2.VideoCapture(str(VIDEO))
    n, fps = int(c.get(7)), c.get(5)
    masks, times = [], []
    for f in range(0, n, 5):
        c.set(1, f)
        ok, im = c.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(im, (1280, 720))[90:560, 700:1220], cv2.COLOR_BGR2GRAY)
        masks.append(cv2.resize((g > 170).astype(np.uint8), (130, 118)))
        times.append(f / fps)
    segs, start, ref = [], None, None
    for m, t in zip(masks, times):
        if m.mean() < 0.003:
            if start is not None:
                segs.append((start, t))
                start = None
            continue
        if start is None:
            start, ref = t, m
        elif (m != ref).mean() > 0.02:
            segs.append((start, t))
            start, ref = t, m
    segs = [s for s in segs if s[1] - s[0] >= 1.0]
    # the letters are the run of long panels (3-9 s each) between the title cards and the closing credits
    run = [s for s in segs if 3.0 <= s[1] - s[0] <= 9.0 and 50 < s[0] < 215]
    if len(run) != len(LETTERS):
        raise SystemExit(f"expected {len(LETTERS)} letter panels, found {len(run)}: {run}")
    return run


def active(clip, margin=4):
    """The stretch of a span where a hand is up: wrist above its resting height by a third of a shoulder width."""
    wy = np.nanmin(clip[:, [4, 7], 1], axis=1)              # the higher wrist (image y points down)
    rest = np.nanpercentile(wy, 90)
    up = np.flatnonzero(wy < rest - 0.33)
    if up.size == 0:
        return 0, len(clip)
    return max(0, up[0] - margin), min(len(clip), up[-1] + 1 + margin)


def main():
    track = dict(np.load(TRACK, allow_pickle=True))
    raw50 = rest_undetected_hands(fifty(track))
    out, index = {}, []
    for k, ((a, b), letter) in enumerate(zip(letter_spans(), LETTERS), 1):
        s, e = int(round(a * FPS)), int(round(b * FPS))
        clip, _, _ = window_frame(raw50[s:e])
        i0, i1 = active(clip)
        s, e = s + i0, s + i1
        clip, neck, scale = window_frame(raw50[s:e])
        sid = f"l{k:02d}"
        face = ((track["face"][s:e].astype(np.float32) - neck) / scale)[:, POINTS]
        ratio = proportions.ratio(clip)
        if np.isfinite(ratio) and ratio > 0:
            face[..., 1] *= proportions.REF / ratio            # as face.load scales a lexicon face
        out[f"{sid}/pose"] = clip.astype(np.float16)
        out[f"{sid}/blend"] = track["blend"][s:e].astype(np.float32)
        out[f"{sid}/face"] = face.astype(np.float16)
        index.append({"id": sid, "letter": letter, "start_s": round(s / FPS, 2), "end_s": round(e / FPS, 2)})
        print(f"{sid} {letter}: {e - s} frames ({s / FPS:.1f}-{e / FPS:.1f} s)")
    out["blend_names"] = np.array([str(n) for n in track["meta"].item()["blend_names"]]) \
        if track["meta"].dtype == object else np.array(json.loads(str(track["meta"]))["blend_names"])
    out["index"] = np.array(json.dumps(index, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"{len(index)} letters -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
