"""Arabic signs cut from a video at known times -> lexicon entries.

For single signs found inside longer videos: a song, a lesson, a story. The time of each sign comes from the speech
or singing (faster-whisper word timestamps, the word sung as it is signed), checked against the frames, and is kept
only when the signer makes the same sign at every time the word occurs. One entry per word (its first clean take);
the other takes are listed in terms.csv as evidence.

Output: data/lexicon_v2/timed_signs/{signs/*.npy, entries.jsonl, terms.csv}
  .venv/Scripts/python scripts/lexicon/arsl/timed_signs.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.pose.keypoints import Holistic, frame_from_holistic, interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

SRC = Path(r"D:\islam\gathring data\arabic_dictionaries\extra")
OUT = DATA / "lexicon_v2" / "timed_signs"
DATASET = "timed_signs"

SIGNS = [  # (gloss, video id, start s, end s, source, how the time was found)
    ("الدنيا", "hpbp2V1IGgA", 0.2, 1.3, "Heba Abdelrahman, «الدنيا بحالها مشاكل» in sign language (YouTube Short)",
     "sung at 0.0-1.2 s and 15.6-17.0 s (Whisper); the same two-handed sign at both, first take kept"),
]
TAKES = {"الدنيا": [(0.2, 1.3), (15.8, 16.7)]}


def extract(path, a, b):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    raw = []
    with Holistic() as holo:
        for t in np.arange(a, b, 1.0 / FPS):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if ok:
                raw.append(frame_from_holistic(holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), t * 1000)))
    cap.release()
    return np.stack(raw)


def main():
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries, rows = [], []
    for n, (gloss, vid, a, b, source, how) in enumerate(SIGNS):
        raw = extract(SRC / f"{vid}.mp4", a, b)
        filled, missing = interpolate_missing(raw)
        sid = f"ts_{vid}_{n:02d}"
        np.save(OUT / "signs" / f"{sid}.npy", normalize(filled))
        entries.append({"gloss": gloss, "sign_id": sid, "dataset": DATASET, "keypoints_path": f"timed_signs/signs/{sid}.npy",
                        "review_status": "pending", "is_religious": False, "is_letter": False})
        rows.append({"gloss": gloss, "sign_id": sid, "video": vid, "start_s": a, "end_s": b, "frames": len(raw),
                     "missing_hand": f"{missing:.0%}", "source": source, "timing": how,
                     "takes": "; ".join(f"{x}-{y}" for x, y in TAKES.get(gloss, [])),
                     "url": f"https://www.youtube.com/watch?v={vid}&t={int(a)}s"})
        print(gloss, sid, len(raw), "frames, missing hand", f"{missing:.0%}")
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
                                       encoding="utf-8")
    with open(OUT / "terms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
