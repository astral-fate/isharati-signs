"""Readable label sheet for a Saudi dictionary video: for each overlay interval (saudi_dictionary/labels/<id>.json),
the top band and the left band of the frame at full resolution, numbered like the intervals.
  .venv/Scripts/python scripts/lexicon/arsl/saudi_dictionary_sheet.py <video-id> ...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from saudi_dictionary import LAB, by_id  # noqa: E402

for i in sys.argv[1:]:
    spans = json.loads((LAB / f"{i}.json").read_text(encoding="utf-8"))
    cap, tiles = cv2.VideoCapture(str(by_id(i))), []
    for n, (a, b) in enumerate(spans):
        cap.set(cv2.CAP_PROP_POS_MSEC, (a + min(1.0, (b - a) / 2)) * 1000)
        ok, img = cap.read()
        if not ok:
            img = np.zeros((720, 1280, 3), np.uint8)
        h, w = img.shape[:2]
        top = cv2.resize(img[: int(h * 0.26), : int(w * 0.5)], (560, int(560 * 0.26 * h / (0.5 * w))))
        side = cv2.resize(img[int(h * 0.15): int(h * 0.7), : int(w * 0.3)], (200, int(200 * 0.55 * h / (0.3 * w))))
        right = cv2.resize(img[: int(h * 0.7), int(w * 0.7):], (200, int(200 * 0.7 * h / (0.3 * w))))
        H = max(top.shape[0], side.shape[0], right.shape[0])
        pad = lambda x: np.vstack([x, np.full((H - x.shape[0], x.shape[1], 3), 30, np.uint8)])
        tile = np.hstack([np.full((H, 60, 3), 30, np.uint8), pad(top), pad(side), pad(right)])
        cv2.putText(tile, str(n), (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        tiles.append(tile)
    cap.release()
    for part in range(0, len(tiles), 12):
        cv2.imwrite(str(LAB / f"{i}.read{part // 12}.jpg"), np.vstack(tiles[part:part + 12]))
    print(i, len(tiles), "tiles")
