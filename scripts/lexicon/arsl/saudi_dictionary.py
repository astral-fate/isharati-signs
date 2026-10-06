"""Signs from Mohammed Hussein's explanation of the Saudi sign language dictionary (YouTube, «شرح قاموس لغة الإشارة
السعودية»), one entry at a time.

Each dictionary entry is announced by a numbered green label at the top left («194.أم») that stays up while the
signer demonstrates and then explains the sign. He keeps signing through the explanation, so a hand-raised stretch
is not one sign. Instead:
  labels   green-label intervals (the label changes -> a new entry), and one numbered crop per interval
           -> saudi_dictionary/labels/<video>.{json,jpg}; the entries are typed into <video>.terms.json
           (a list aligned with the crops; "" skips one)
  extract  MediaPipe Holistic per video (cached)
  cut      the entry's sign = the first motion segment after the label appears: from the first frame the hands
           move after the onset to the first clear pause in hand speed (0.5-3 s); three frames per cut go to
           a contact sheet for checking
  .venv/Scripts/python scripts/lexicon/arsl/saudi_dictionary.py labels|extract|cut [video-id ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent / "isharah"))
from gloss_clips import hand_speed  # noqa: E402

from isharati.pose.keypoints import Holistic, frame_from_holistic, interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

SRC = Path(r"D:\islam\gathring data\saudi_dictionary")
OUT = DATA / "lexicon_v2" / "saudi_dictionary"
LAB = OUT / "labels"
STEP = 0.5  # seconds between label samples


def videos():
    return sorted(p for p in SRC.glob("*/*.mp4") if ".f" not in p.stem and ".temp" not in p.stem)


def vid(p):
    return p.stem.split(" - ")[-1]


def by_id(i):
    return next(p for p in videos() if vid(p) == i)


def green_box(img):
    """Mask of the label's green box in the top-left region, and the box's white text."""
    h, w = img.shape[:2]
    r = img[: int(h * 0.22), : int(w * 0.45)]
    b, g, rr = r[..., 0].astype(int), r[..., 1].astype(int), r[..., 2].astype(int)
    box = (g > 110) & (g > rr + 50) & (g > b + 50)
    return box, r


def labels(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(fps * STEP)))
    samples, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % every == 0:
            ok, img = cap.retrieve()
            box, region = green_box(img)
            on = box.mean() > 0.01
            sig = None
            if on:
                ys, xs = np.nonzero(box)
                crop = region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                sig = cv2.resize((crop.min(axis=2) > 180).astype(np.uint8) * 255, (96, 24))
            samples.append((i / fps, on, sig, region if on else None))
        i += 1
    cap.release()
    spans = []  # [start, end, crop]
    for t, on, sig, region in samples:
        if not on:
            if spans and spans[-1][3] is not None:
                spans[-1][3] = None  # the label went off: close it
            continue
        if spans and spans[-1][3] is not None and np.mean(np.abs(sig.astype(int) - spans[-1][3].astype(int))) < 40:
            spans[-1][1] = t
        else:
            spans.append([t, t, region, sig])
    spans = [s for s in spans if s[1] - s[0] >= 1.0]
    LAB.mkdir(parents=True, exist_ok=True)
    (LAB / f"{vid(path)}.json").write_text(json.dumps([[s[0], s[1]] for s in spans]), encoding="utf-8")
    tiles = []
    for n, (a, b, region, _) in enumerate(spans):
        box, _ = green_box(np.pad(region, ((0, 1), (0, 1), (0, 0))))
        ys, xs = np.nonzero(box)
        c = region[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5] if len(xs) else region
        c = cv2.resize(c, (300, max(20, int(c.shape[0] * 300 / max(1, c.shape[1])))))[:80]
        tile = np.full((90, 380, 3), 40, np.uint8)
        tile[:c.shape[0], 70:70 + c.shape[1]] = c
        cv2.putText(tile, str(n), (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(tile, f"{a:.0f}s", (4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 40))
        cv2.imwrite(str(LAB / f"{vid(path)}.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(vid(path), len(spans), "labels")


def overlays(path):
    """Any on-screen label, whatever its style (green or orange boxes, grey phrase boxes, big words with a dictionary
    picture, side captions): edges in the side and top bands that are not part of the static room, and that stay
    still for at least a second. Hands pass through those bands but do not stay still, so they are not labels."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(fps * STEP)))
    frames, times, i = [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % every == 0:
            _, img = cap.retrieve()
            frames.append(cv2.resize(img, (480, 270)))
            times.append(i / fps)
        i += 1
    cap.release()
    h, w = frames[0].shape[:2]
    band = np.zeros((h, w), bool)
    band[:, : int(w * 0.33)] = True
    band[:, int(w * 0.72):] = True
    band[: int(h * 0.16), :] = True
    edges = np.stack([(cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 80, 200) > 0) & band for f in frames])
    static = edges.mean(axis=0) > 0.5                                  # the room, the logo in the corner
    k = np.ones((3, 3), np.uint8)
    ov = np.stack([cv2.dilate((e & ~static).astype(np.uint8), k) > 0 for e in edges])
    count = ov.sum(axis=(1, 2))
    spans, cur = [], None                                                # [start, end, mask, frame index]
    for n in range(1, len(frames)):
        a, b = ov[n], ov[n - 1]
        inter, union = (a & b).sum(), (a | b).sum()
        stable = count[n] > 150 and union and inter / union > 0.6
        if stable and cur is not None and ((a & cur[2]).sum() / max(1, (a | cur[2]).sum())) > 0.5:
            cur[1] = times[n]
        elif stable:
            if cur is not None and cur[1] - cur[0] >= 1.0:
                spans.append(cur)
            cur = [times[n - 1], times[n], a.copy(), n]
        else:
            if cur is not None and times[n] - cur[1] > STEP * 1.5:
                if cur[1] - cur[0] >= 1.0:
                    spans.append(cur)
                cur = None
    if cur is not None and cur[1] - cur[0] >= 1.0:
        spans.append(cur)
    LAB.mkdir(parents=True, exist_ok=True)
    (LAB / f"{vid(path)}.json").write_text(json.dumps([[s[0], s[1]] for s in spans]), encoding="utf-8")
    tiles = []
    for n, (a, b, mask, idx) in enumerate(spans):
        ys, xs = np.nonzero(mask)
        f = frames[idx]
        c = f[max(0, ys.min() - 6):ys.max() + 7, max(0, xs.min() - 6):xs.max() + 7]
        scale = min(300 / c.shape[1], 110 / c.shape[0])
        c = cv2.resize(c, (max(1, int(c.shape[1] * scale)), max(1, int(c.shape[0] * scale))))
        tile = np.full((120, 380, 3), 40, np.uint8)
        tile[:c.shape[0], 70:70 + c.shape[1]] = c
        cv2.putText(tile, str(n), (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(tile, f"{a:.0f}s", (4, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 40))
        cv2.imwrite(str(LAB / f"{vid(path)}.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(vid(path), len(spans), "overlays")


def extract(path):
    cache = OUT / f"raw_{vid(path)}.npz"
    if cache.exists():
        return
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = set(np.linspace(0, n_src - 1, int(round(n_src * FPS / src_fps))).round().astype(int).tolist())
    raw, times, i = [], [], 0
    with Holistic() as holo:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i in picks:
                _, img = cap.retrieve()
                raw.append(frame_from_holistic(holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), i * 1000 / src_fps)))
                times.append(i / src_fps)
            i += 1
    cap.release()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, raw=np.stack(raw), t=np.array(times))
    print(vid(path), "extracted", flush=True)


def first_sign(raw, t, a, b):
    """[s, e) frames of the first motion segment after time a (before b): hands start moving, then the first
    clear pause in hand speed at least 0.5 s later, capped at 3 s."""
    s0, s1 = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
    if s1 - s0 < 12:
        return None
    filled, _ = interpolate_missing(raw[s0:s1])
    v = hand_speed(normalize(filled))
    moving = np.where(v > 0.3 * np.percentile(v, 90))[0]
    if not len(moving):
        return None
    start = int(moving[0])
    lo, hi = start + 12, min(len(v) - 1, start + 75)
    if lo >= hi:
        return None
    seg = v[lo:hi]
    minima = [k for k in range(1, len(seg) - 1) if seg[k] <= seg[k - 1] and seg[k] <= seg[k + 1]
              and seg[k] < 0.35 * v[start:hi].max()]
    end = lo + minima[0] if minima else hi
    return s0 + start, s0 + end


def cut(path):
    i = vid(path)
    spans = json.loads((LAB / f"{i}.json").read_text(encoding="utf-8"))
    terms = json.loads((LAB / f"{i}.terms.json").read_text(encoding="utf-8"))
    if len(terms) != len(spans):
        raise SystemExit(f"{i}: {len(terms)} terms for {len(spans)} labels")
    z = np.load(OUT / f"raw_{i}.npz")
    raw, t = z["raw"], z["t"]
    entries_path = OUT / "entries.jsonl"
    entries = {e["sign_id"]: e for e in (json.loads(l) for l in entries_path.read_text(encoding="utf-8").splitlines()
                                         if l.strip())} if entries_path.exists() else {}
    for k in [k for k in entries if k.startswith(f"sd_{i}_")]:
        del entries[k]
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    cap, rows = cv2.VideoCapture(str(path)), []
    for n, ((a, b), gloss) in enumerate(zip(spans, terms)):
        if not gloss:
            continue
        found = first_sign(raw, t, a - 0.5, b)
        if not found:
            continue
        s, e = found
        sid = f"sd_{i}_{n:02d}"
        filled, _ = interpolate_missing(raw[s:e])
        np.save(OUT / "signs" / f"{sid}.npy", normalize(filled))
        # part 2 of his series explains the unified Arabic dictionary, the rest the Saudi one
        dataset = "arabic_dictionary_explained" if "العربي" in path.parent.name else "saudi_dictionary_explained"
        entries[sid] = {"gloss": gloss, "sign_id": sid, "dataset": dataset,
                        "keypoints_path": f"saudi_dictionary/signs/{sid}.npy", "review_status": "pending",
                        "is_religious": False, "is_letter": False}
        row = []
        for frac in (0.2, 0.5, 0.8):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t[s + int((e - s - 1) * frac)]) * 1000)
            ok, img = cap.read()
            tile = cv2.resize(img, (256, 144)) if ok else np.zeros((144, 256, 3), np.uint8)
            cv2.putText(tile, f"{n}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            row.append(tile)
        rows.append(np.hstack(row))
    cap.release()
    entries_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries.values()), encoding="utf-8")
    if rows:
        cv2.imwrite(str(LAB / f"{i}.cuts.jpg"), np.vstack(rows))
    print(i, len(rows), "signs")


if __name__ == "__main__":
    mode, ids = sys.argv[1], sys.argv[2:]
    paths = [by_id(x) for x in ids] if ids else videos()
    for p in paths:
        {"labels": labels, "overlays": overlays, "extract": extract, "cut": cut}[mode](p)
