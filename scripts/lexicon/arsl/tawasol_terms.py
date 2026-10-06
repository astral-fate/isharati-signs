"""Arabic signs from Tawasol Center vocabulary videos (Saudi Deaf interpreters) -> Arabic lexicon entries.

Each video shows one signer on a green background. Every term is captioned in white on the right of the frame and
signed while its caption is up, with the hands clasped at rest between terms. There is no speech, so:
  1. MediaPipe Holistic on the full frame;
  2. signing stretches = a hand raised above waist level (scripts/lexicon/asl/islamic_terms.signing_runs);
  3. caption onsets = times the share of white pixels right of the signer rises above a threshold; with the
     terms listed in order below, onset k anchors term k, and islamic_terms.assign_in_order gives every term its
     own stretch in order.

Terms are written as captioned. A caption that is a phrase («صوم رمضان») becomes one phrase sign.
Output: data/lexicon_v2/tawasol/{signs/*.npy, entries.jsonl, terms.csv, contact_<video>.jpg}
  .venv/Scripts/python scripts/lexicon/arsl/tawasol_terms.py [video ...]
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
from islamic_terms import assign_in_order, signing_runs  # noqa: E402

from isharati.pose.keypoints import frame_from_holistic, interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

SRC = Path(r"D:\islam\gathring data\tawasol_videos")
OUT = DATA / "lexicon_v2" / "tawasol"
DATASET = "tawasol"

VIDEOS = {  # video number -> (caption x start as a share of the width, terms in caption order)
    "109": (0.5, ["محرم", "صفر", "ربيع الأول", "ربيع الثاني", "جمادى الأولى", "جمادى الآخرة", "رجب", "شعبان",
                  "رمضان", "شوال", "ذو القعدة", "ذو الحجة"]),
    "091": (0.55, ["أركان الإسلام الخمسة", "شهادة أن لا إله إلا الله وأن محمدا رسول الله", "إقامة الصلاة",
                   "إيتاء الزكاة", "صوم رمضان", "حج البيت لمن استطاع إليه سبيلا"]),
    "104": (0.55, ["مناسك العمرة", "الميقات", "الكعبة", "الحجر الأسود", "طواف", "مقام إبراهيم", "الصفا والمروة"]),
    "076": (0.5, ["رمي الجمرات", "الأضحية", "فك الإحرام", "الطواف", "طواف الوداع", "الصفا والمروة", "يسبح",
                  "جبل عرفات", "خطبة العيد", "منى", "مزدلفة", "إحرام", "مكة", "تمتع", "مفرد", "قارن", "الميقات",
                  "صلاة العيد", "الجمرات الكبرى", "الجمرات الوسطى", "الجمرات الصغرى"]),  # «قارن» is captioned «مقرن»
}
MIN_FRAMES = 12  # 0.5 s at 25 fps
RELIGIOUS_VIDEOS = {"091", "104", "076"}
RELIGIOUS_MONTHS = {"محرم", "رمضان", "شوال", "ذو الحجة"}


def video_path(num):
    return next((SRC.glob(f"{num} - */*.mp4")))


def extract(path):
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    picks = np.linspace(0, len(frames) - 1, int(round(len(frames) * FPS / src_fps))).round().astype(int)
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holo:
        raw = np.stack([frame_from_holistic(holo.process(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))) for i in picks])
    white = np.array([caption_share(frames[i], X0) for i in picks])
    return raw, picks / src_fps, white


def caption_share(img, x0):
    """Share of near-white pixels right of the signer, where the caption sits."""
    region = img[:, int(img.shape[1] * x0):]
    return float((region.min(axis=2) > 200).mean())


def caption_onsets(white, t, n_terms, last_sign_end, min_gap=1.0):
    """Start times of the n captions: rises of the white share above a threshold chosen so that exactly n
    separate caption-on intervals appear (captions fade, so a fixed threshold splits or merges some)."""
    best = None
    for thr in np.quantile(white, np.linspace(0.3, 0.95, 60)):
        on = white > thr
        starts, last_end = [], -1e9
        k = 0
        while k < len(on):
            if on[k]:
                j = k
                while j < len(on) and on[j]:
                    j += 1
                if t[j - 1] - t[k] >= 0.3 and t[k] < last_sign_end:  # >= 0.3 s up; not the end card
                    if t[k] - last_end < min_gap and starts:
                        pass  # the same caption flickering across a fade
                    else:
                        starts.append(float(t[k]))
                    last_end = float(t[j - 1])
                k = j
            else:
                k += 1
        if len(starts) == n_terms:
            return starts
        if best is None or abs(len(starts) - n_terms) < abs(len(best) - n_terms):
            best = starts
    return best


def main():
    global X0
    nums = sys.argv[1:] or list(VIDEOS)
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries_path = OUT / "entries.jsonl"
    entries = {e["sign_id"]: e for e in (json.loads(l) for l in entries_path.read_text(encoding="utf-8").splitlines()
                                         if l.strip())} if entries_path.exists() else {}
    rows = []
    for num in nums:
        X0, terms = VIDEOS[num]
        path = video_path(num)
        cache = OUT / f"raw_{num}.npz"
        if cache.exists():
            z = np.load(cache)
            raw, t, white = z["raw"], z["t"], z["white"]
        else:
            raw, t, white = extract(path)
            np.savez_compressed(cache, raw=raw, t=t, white=white)
        runs = signing_runs(raw)
        onsets = caption_onsets(white, t, len(terms), float(t[runs[-1][1] - 1]))
        starts = {g: s for g, s in zip(terms, onsets)} if len(onsets) == len(terms) else {}
        print(f"{num}: {len(runs)} signing stretches, {len(onsets)} caption onsets for {len(terms)} terms"
              + ("" if starts else "  (count differs: one stretch per term, in order)"))
        if starts:  # a phrase caption can span several stretches: a term takes every stretch starting in its window
            spans = {}
            for k, g in enumerate(terms):
                a = onsets[k] - 0.5
                b = onsets[k + 1] - 0.5 if k + 1 < len(terms) else float("inf")
                mine = [r for r, (s, e) in enumerate(runs) if a <= t[s] < b]
                if mine:
                    spans[g] = (runs[mine[0]][0], runs[mine[-1]][1])
        else:
            spans = {g: runs[r] for g, r in assign_in_order(terms, starts, runs, t).items()}
        thumbs, cap = [], cv2.VideoCapture(str(path))
        for n, gloss in enumerate(terms):
            sid = f"taw{num}_{n:02d}"
            if gloss not in spans:
                rows.append({"video": num, "gloss": gloss, "sign_id": sid, "start_s": "", "end_s": "",
                             "caption_s": starts.get(gloss, ""), "status": "not found"})
                continue
            s, e = spans[gloss]
            if e - s < MIN_FRAMES:  # a sign made low (near the waist) is only partly detected: not a whole sign
                entries.pop(sid, None)
                rows.append({"video": num, "gloss": gloss, "sign_id": sid, "start_s": round(float(t[s]), 2),
                             "end_s": round(float(t[e - 1]), 2), "caption_s": starts.get(gloss, ""),
                             "status": "too short, not used"})
                continue
            filled, missing = interpolate_missing(raw[s:e])
            np.save(OUT / "signs" / f"{sid}.npy", normalize(filled))
            entries[sid] = {"gloss": gloss, "sign_id": sid, "dataset": DATASET,
                            "keypoints_path": f"tawasol/signs/{sid}.npy", "review_status": "pending",
                            "is_religious": num in RELIGIOUS_VIDEOS or gloss in RELIGIOUS_MONTHS,
                            "is_letter": False}
            rows.append({"video": num, "gloss": gloss, "sign_id": sid, "start_s": round(float(t[s]), 2),
                         "end_s": round(float(t[e - 1]), 2), "caption_s": round(starts[gloss], 2) if gloss in starts else "",
                         "status": f"missing hand {missing:.0%}"})
            for frac in (0.25, 0.5, 0.75):  # three frames per sign, to check it is the whole sign
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t[s + int((e - s - 1) * frac)]) * 1000)
                ok, img = cap.read()
                if ok:
                    thumbs.append(cv2.resize(img, (256, 144)))
        cap.release()
        if thumbs:
            grid = np.vstack([np.hstack(thumbs[i:i + 3]) for i in range(0, len(thumbs), 3)])
            cv2.imwrite(str(OUT / f"contact_{num}.jpg"), grid)
    entries_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries.values()), encoding="utf-8")
    old = []
    if (OUT / "terms.csv").exists():
        old = [r for r in csv.DictReader(open(OUT / "terms.csv", encoding="utf-8")) if r["video"] not in nums]
    with open(OUT / "terms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(old + rows)
    for r in rows:
        print(f"  {r['video']} {r['gloss'][:28]:28s} {r['start_s']!s:>7} - {r['end_s']!s:<7} caption {r['caption_s']!s:<6} {r['status']}")


if __name__ == "__main__":
    main()
