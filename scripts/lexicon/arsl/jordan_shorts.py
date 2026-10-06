"""Levantine (Jordanian/Palestinian) Arabic signs from one-word YouTube Shorts («لغة الإشارة <word> #...»).

Each Short teaches one sign: the signer repeats it several times between explanations. The sign is the motion
segment that recurs most: the signing is cut at pauses in hand speed into 0.5-2.5 s segments, and the segment with
the smallest mean DTW distance to its three nearest segments is kept (a one-off gesture has no near neighbours).
The gloss is the title's first word or phrase; the others after «/» or «-» become aliases. Tagged by dialect so
the demo uses them only where no Saudi sign exists.
  .venv/Scripts/python scripts/lexicon/arsl/jordan_shorts.py extract|cut
Output: data/lexicon_v2/jordan_shorts/{raw_*.npz, signs/*.npy, entries.jsonl}
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent / "isharah"))
from gloss_clips import hand_speed  # noqa: E402

from isharati.pose.dtw import dtw_distance  # noqa: E402
from isharati.pose.keypoints import Holistic, frame_from_holistic, interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

SRC = Path(r"D:\islam\gathring data\jordan_shorts")
OUT = DATA / "lexicon_v2" / "jordan_shorts"


def glosses(title):
    t = re.sub(r"#\S+", " ", title).replace("لغة الإشارة", " ")
    parts = [p.strip(" -–_/،,") for p in re.split(r"[/\-–_]", t)]
    return [re.sub(r"\s+", " ", p) for p in parts if p.strip(" -–_/،,")]


def extract(path):
    cache = OUT / f"raw_{path.stem.split(' - ')[-1]}.npz"
    if cache.exists():
        return
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = set(np.linspace(0, n_src - 1, int(round(n_src * FPS / src_fps))).round().astype(int).tolist())
    raw, i = [], 0
    with Holistic() as holo:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i in picks:
                _, img = cap.retrieve()
                raw.append(frame_from_holistic(holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), i * 1000 / src_fps)))
            i += 1
    cap.release()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, raw=np.stack(raw))


def segments(p):
    v = hand_speed(p)
    active = v > 0.25 * np.percentile(v, 90)
    minima = [k for k in range(1, len(v) - 1) if v[k] <= v[k - 1] and v[k] <= v[k + 1] and v[k] < 0.3 * np.percentile(v, 90)]
    cuts = sorted(set([0] + minima + [len(v)]))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if 12 <= b - a <= 62 and active[a:b].mean() > 0.6:
            out.append((a, b))
    return out


def repeated_sign(p):
    segs = segments(p)
    if len(segs) < 3:
        return None
    clips = [p[a:b] for a, b in segs]
    D = np.array([[dtw_distance(x, y) if i != j else np.inf for j, y in enumerate(clips)] for i, x in enumerate(clips)])
    score = np.sort(D, axis=1)[:, :3].mean(axis=1)
    k = int(score.argmin())
    return segs[k], float(score[k]), float(np.median(score))


def main():
    mode = sys.argv[1]
    vids = sorted(SRC.glob("*.mp4"))
    if mode in ("extract", "extract-rev"):  # a second worker can take the list from the other end
        for v in (vids[::-1] if mode == "extract-rev" else vids):
            extract(v)
            print(v.name, "extracted", flush=True)
        return
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries, kept = [], 0
    for v in vids:
        vid = v.stem.split(" - ")[-1]
        info = v.with_suffix(".info.json")
        cache = OUT / f"raw_{vid}.npz"
        if not info.exists() or not cache.exists():
            continue
        names = glosses(json.loads(info.read_text(encoding="utf-8"))["title"])
        if not names:
            continue
        filled, _ = interpolate_missing(np.load(cache)["raw"])
        p = normalize(filled)
        found = repeated_sign(p)
        if not found:
            print(f"  {vid} {names[0]}: no repeated sign found")
            continue
        (a, b), best, typical = found
        sid = f"jo_{vid}"
        np.save(OUT / "signs" / f"{sid}.npy", p[a:b])
        for g in names:
            entries.append({"gloss": g, "sign_id": sid, "dataset": "jordan_shorts", "dialect": "levantine",
                            "keypoints_path": f"jordan_shorts/signs/{sid}.npy", "review_status": "pending",
                            "is_religious": False, "is_letter": False, "repeat_dtw": round(best, 3)})
        kept += 1
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    print(f"{kept} signs, {len(entries)} glosses")


if __name__ == "__main__":
    main()
