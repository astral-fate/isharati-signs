"""Arabic signs from the «إشارتي هي لغتي» channel (Ayman Abbas) vocabulary videos -> Arabic lexicon entries.

One signer on a plain light background. Each term is captioned in dark text at the top left of the frame while it is
signed (fading in and out), with the hands down between terms. The caption windows were read from a 1 fps timeline of
the caption area (timeline_*.jpg next to the video), so each term's sign is the signing inside its own window:
  1. MediaPipe Holistic on the full frame, at 25 fps;
  2. signing frames = a hand raised above waist level (scripts/lexicon/asl/islamic_terms.signing_runs);
  3. a term takes the signing runs that overlap its caption window (padded by PAD seconds).

Terms are written as captioned; a phrase caption («رأس المال») becomes one phrase sign. A term signed twice (توكيل)
keeps its first take.
Output: data/lexicon_v2/ayman_abbas/{signs/*.npy, entries.jsonl, terms.csv, contact_<video>.jpg}
  .venv/Scripts/python scripts/lexicon/arsl/ayman_abbas.py [video_id ...]
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
sys.path.insert(0, str(ROOT / "scripts" / "lexicon" / "asl"))
from islamic_terms import signing_runs  # noqa: E402

from isharati.pose.keypoints import Holistic, frame_from_holistic, interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

SRC = Path(r"D:\islam\gathring data\arabic_dictionaries\ayman_abbas")
OUT = DATA / "lexicon_v2" / "ayman_abbas"
DATASET = "ayman_abbas"
PAD = 0.6         # seconds around a caption window: the sign can start as the caption fades in
MIN_FRAMES = 12   # 0.5 s at 25 fps

VIDEOS = {  # YouTube id -> (title, [(term, caption from s, caption to s)])
    "mPUd4lyqfTE": ("تعرف على مصطلحات التجارة والأعمال بلغة الإشارة", [
        ("السلام عليكم ورحمة الله وبركاته", 1, 4), ("تجارة وأعمال", 7, 10), ("مشروع", 13, 17),
        ("ميزانية", 18, 21), ("إفلاس", 23, 26), ("حسابات", 28, 30), ("تجارة", 32, 33), ("يشتري", 34, 37),
        ("يبيع", 38, 40), ("يتسوق", 41, 44), ("استغلال", 46, 47), ("رأس المال", 49, 52), ("رخيص", 53, 55),
        ("غالي", 57, 59), ("اعتماد مالي", 61, 64), ("ينتخب", 66, 69), ("انتخابات", 71, 74), ("مدين", 75, 77),
        ("دائن", 78, 80), ("توكيل", 81, 84), ("توكيل عام", 85, 88), ("توكيل خاص", 90, 93), ("نقود", 100, 101),
        ("بنك", 103, 106), ("دكان", 108, 110), ("شيك", 111, 114), ("جدول أعمال", 116, 118), ("تأمين", 120, 122),
        ("صرف", 124, 126)]),
}


def extract(path):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = set(np.linspace(0, n - 1, int(round(n * FPS / src_fps))).round().astype(int).tolist())
    raw, times, i = [], [], 0
    with Holistic() as holo:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in picks:
                raw.append(frame_from_holistic(holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), i * 1000 / src_fps)))
                times.append(i / src_fps)
            i += 1
    cap.release()
    return np.stack(raw), np.array(times)


def main():
    vids = sys.argv[1:] or list(VIDEOS)
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries_path = OUT / "entries.jsonl"
    entries = {e["sign_id"]: e for e in (json.loads(l) for l in entries_path.read_text(encoding="utf-8").splitlines()
                                         if l.strip())} if entries_path.exists() else {}
    rows = []
    for vid in vids:
        _, terms = VIDEOS[vid]
        path = SRC / f"{vid}.mp4"
        cache = OUT / f"raw_{vid}.npz"
        if cache.exists():
            z = np.load(cache)
            raw, t = z["raw"], z["t"]
        else:
            raw, t = extract(path)
            np.savez_compressed(cache, raw=raw, t=t)
        runs = signing_runs(raw, min_len=4, max_gap=4)
        thumbs, cap = [], cv2.VideoCapture(str(path))
        for n, (gloss, a, b) in enumerate(terms):
            sid = f"aya_{vid}_{n:02d}"
            mine = [(s, e) for s, e in runs if t[s] < b + PAD and t[e - 1] > a - PAD]
            if not mine:
                entries.pop(sid, None)
                rows.append({"video": vid, "gloss": gloss, "sign_id": sid, "caption_s": f"{a}-{b}", "start_s": "",
                             "end_s": "", "status": "no signing in the caption window"})
                continue
            s = max(mine[0][0], int(np.searchsorted(t, a - PAD)))
            e = min(mine[-1][1], int(np.searchsorted(t, b + 1 + PAD)))
            status = None
            if e - s < MIN_FRAMES:
                status = "too short, not used"
            else:
                filled, missing = interpolate_missing(raw[s:e])
                np.save(OUT / "signs" / f"{sid}.npy", normalize(filled))
                entries[sid] = {"gloss": gloss, "sign_id": sid, "dataset": DATASET,
                                "keypoints_path": f"ayman_abbas/signs/{sid}.npy", "review_status": "pending",
                                "is_religious": False, "is_letter": False,
                                "source_url": f"https://www.youtube.com/watch?v={vid}&t={int(t[s])}s"}
                status = f"missing hand {missing:.0%}"
                for frac in (0.25, 0.5, 0.75):  # three frames per sign, to check it is the whole sign
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(t[s + int((e - s - 1) * frac)]) * 1000)
                    ok, img = cap.read()
                    if ok:
                        img = cv2.resize(img, (256, 144))
                        cv2.putText(img, f"{n:02d}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        thumbs.append(img)
            if status == "too short, not used":
                entries.pop(sid, None)
            rows.append({"video": vid, "gloss": gloss, "sign_id": sid, "caption_s": f"{a}-{b}",
                         "start_s": round(float(t[s]), 2), "end_s": round(float(t[e - 1]), 2), "status": status})
        cap.release()
        if thumbs:
            grid = np.vstack([np.hstack(thumbs[i:i + 6]) if len(thumbs[i:i + 6]) == 6 else
                              np.hstack(thumbs[i:i + 6] + [np.zeros_like(thumbs[0])] * (6 - len(thumbs[i:i + 6])))
                              for i in range(0, len(thumbs), 6)])
            cv2.imwrite(str(OUT / f"contact_{vid}.jpg"), grid)
    entries_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries.values()) + "\n",
                            encoding="utf-8")
    with open(OUT / "terms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["sign_id"], r["gloss"], r["caption_s"], r["start_s"], r["end_s"], r["status"])


if __name__ == "__main__":
    main()
