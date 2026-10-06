"""Frames around one signing stretch of a Tawasol lesson, for reading a caption missing from its middle frame.
  .venv/Scripts/python scripts/lexicon/arsl/tawasol_peek.py <video> <stretch> [<stretch> ...]  -> tawasol/runs/peek_<video>.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tawasol_runs import RUNS, load, signing_runs, video_path  # noqa: E402

num, idx = sys.argv[1], [int(i) for i in sys.argv[2:]]
raw, t = load(num)
runs = signing_runs(raw)
cap, rows = cv2.VideoCapture(str(video_path(num))), []
for i in idx:
    s, e = runs[i]
    row = []
    for sec in np.linspace(float(t[s]) - 1.2, float(t[e - 1]) + 0.4, 6):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sec) * 1000)
        ok, img = cap.read()
        tile = cv2.resize(img, (320, 180)) if ok else np.zeros((180, 320, 3), np.uint8)
        cv2.putText(tile, f"#{i} {sec:.1f}s", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        row.append(tile)
    rows.append(np.hstack(row))
cv2.imwrite(str(RUNS / f"peek_{num}.jpg"), np.vstack(rows))
