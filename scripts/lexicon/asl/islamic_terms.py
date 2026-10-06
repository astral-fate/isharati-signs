"""Islamic ASL signs from "GDM's 40 Islamic signs" (Global Deaf Muslim) -> ASL lexicon entries.

The video shows one Deaf signer; each term is captioned, spoken in Arabic, then signed, with the hands resting at
the sides between terms. So:
  1. MediaPipe Holistic on the signer region, cropped to 4:3 like ASL Citizen's 640x480, so body proportions match
     when clips from both sources are stitched;
  2. signing stretches = frames where a hand is raised above waist level, merged over short gaps;
  3. each term is anchored by its spoken Arabic word (Whisper word timestamps): its sign is the stretch that
     overlaps most with [term start, next term start); terms with no audio (Hadith, Surah, Ayah) take the
     stretches inside their silent gap, in order.

Output: data/asl/lexicon/islamic/{signs/*.npy, entries.jsonl, terms.csv, contact_sheet.jpg}
  .venv/Scripts/python scripts/lexicon/asl/islamic_terms.py [gdm|noor]
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
from isharati.text.arabic import normalize_ar  # noqa: E402
from isharati.pose.keypoints import L_SH, L_WR, LH, R_SH, R_WR, RH, Holistic, frame_from_holistic, interpolate_missing, normalize  # noqa: E402
from isharati.types import FPS  # noqa: E402

SRC = DATA / "asl" / "islamic_terms"
VIDEO = SRC / "L-lWEt7OeJk.mp4"
OUT = DATA / "asl" / "lexicon" / "islamic"
CROP_X0 = 192  # 1152x720 -> x 192..1152 = 960x720 (4:3), the signer is centre-right

# (gloss, spoken Arabic cue or None, English aliases the glosser may use)
GDM_TERMS = [
    ("ASSALAMU ALAIKUM", "السلام عليكم", ["peace be upon you", "greeting"]),
    ("ISLAM", "اسلام", []), ("MUSLIM", "مسلم", ["muslims"]), ("MASJID", "مسجد", ["mosque"]),
    ("ALLAH", "الله", []), ("MUHAMMAD", "محمد صلي", ["muhammad pbuh"]),
    ("PROPHET MUHAMMAD", "محمد", ["prophet", "messenger"]), ("ARABIC", "عربي", []),
    ("SALAH", "صلاه", ["salat", "prayers"]), ("WUDU", "وضوء", ["ablution"]), ("KHUTBAH", "خطبه", ["sermon"]),
    ("JUMMAH", "جمعه", ["friday prayer"]), ("ADHAN", "اذان", ["call to prayer"]), ("IMAM", "امام", []),
    ("ALLAHU AKBAR", "الله اكبر", ["god is great"]), ("DUA", "دعاء", ["supplication"]), ("QURAN", "قران", ["koran"]),
    ("HADITH", None, []), ("SURAH", None, ["chapter"]), ("AYAH", None, ["verse", "ayat"]),
    ("SAWM", "صوم", ["fasting"]), ("SADAQAH", "صدقه", ["charity"]), ("ZAKAT", "زكاه", ["zakah", "almsgiving"]),
    ("KAABAH", "كعبه", ["kaaba"]), ("HAJJ", "حج", ["pilgrimage"]), ("UMRAH", "عمره", []),
    ("UMMAH", "امه", ["muslim community"]), ("SHAHADAH", "شهاده", ["testimony of faith"]), ("HALAL", "حلال", []),
    ("HARAM", "حرام", []), ("JANNAH", "جنه", ["paradise"]), ("JAHANNAM", "جهنم", ["hellfire"]),
    ("DAWAH", "دعوه", []), ("DEEN", "دين", ["way of life"]), ("DUNYA", "دنيا", ["worldly life"]),
    ("BISMILLAH", "بسم الله", []), ("ALHAMDULILLAH", "الحمد لله", ["praise be to allah"]),
    ("INSHAALLAH", "ان شاء الله", ["if allah wills"]), ("SUBHANALLAH", "سبحان الله", ["glory be to allah"]),
    ("MASHAALLAH", "ما شاء الله", []), ("WA ALAIKUM ASSALAM", "والسلام عليكم", []),
]

# "Lesson 2 Islamic words in SSL/ASL" (Noor For Sign): 7 Ramadan terms, one signer, caption top-left
NOOR_TERMS = [
    ("RAMADAN", "رمضان", []), ("EID AL-FITR", "الفطر", ["eid"]), ("SAWM", "صوم", ["fasting", "fast"]),
    ("IFTAR", "افطار", ["breaking the fast"]), ("SUHOOR", "سهور", ["suhur", "predawn meal"]),
    ("TARAWEEH", "طراوي", ["taraweeh prayer", "tarawih"]), ("LAYLAT AL-QADR", "ليله القدر", ["night of power"]),
]

CONFIGS = {  # name -> (source folder, video file, output folder, crop x0 for a 4:3 crop, terms, dataset, id prefix)
    "gdm": ("islamic_terms", "L-lWEt7OeJk.mp4", "islamic", 192, GDM_TERMS, "gdm_islamic_signs", "isl"),
    "noor": ("noor_lesson2", "VWzAH0j56f4.mp4", "noor", 160, NOOR_TERMS, "noor_for_sign", "noor"),
}
TERMS, DATASET, PREFIX = GDM_TERMS, "gdm_islamic_signs", "isl"


def extract():
    cap = cv2.VideoCapture(str(VIDEO))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img[:, CROP_X0:], cv2.COLOR_BGR2RGB))
    cap.release()
    picks = np.linspace(0, len(frames) - 1, int(round(len(frames) * FPS / src_fps))).round().astype(int)
    with Holistic() as holo:
        raw = np.stack([frame_from_holistic(holo.process(frames[i], i * 1000 / src_fps)) for i in picks])
    return raw, picks / src_fps


def signing_runs(raw, min_len=8, max_gap=6):
    """Frames with a hand raised above waist level, as [start, end) runs."""
    sh_y = (raw[:, R_SH, 1] + raw[:, L_SH, 1]) / 2
    sw = np.abs(raw[:, R_SH, 0] - raw[:, L_SH, 0])
    waist = sh_y + 1.3 * sw
    active = np.zeros(len(raw), bool)
    for wrist, hand in ((R_WR, RH), (L_WR, LH)):
        seen = ~np.isnan(raw[:, hand, 0]).all(axis=1)
        active |= seen & (raw[:, wrist, 1] < waist)
    runs, start, gap = [], None, 0
    for i, a in enumerate(active):
        if a:
            start = i if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(active)))
    return [(a, b) for a, b in runs if b - a >= min_len]


def caption_intervals(min_len=1.5, step_fps=5):
    """Each term has its own caption (name + definition) on the left of the frame. Times where that region
    changes sharply split the video into one interval per caption."""
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    every = max(1, int(round(fps / step_fps)))
    prev, diffs, times, i = None, [], [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % every == 0:
            region = cv2.cvtColor(img[380:680, 0:560], cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.abs(region - prev).mean()))
                times.append(i / fps)
            prev = region
        i += 1
    cap.release()
    d = np.array(diffs)
    thr = d.mean() + 2 * d.std()
    cuts = [0.0]
    for k in range(1, len(d) - 1):
        if d[k] > thr and d[k] >= d[k - 1] and d[k] >= d[k + 1] and times[k] - cuts[-1] >= min_len:
            cuts.append(times[k])
    cuts.append(times[-1] if times else 0.0)
    return list(zip(cuts[:-1], cuts[1:]))


def term_times(whisper):
    """Start time of each term's spoken cue, found in order in Whisper's word stream."""
    words = [(normalize_ar(w["word"]), w["start"]) for w in whisper["words"]]
    starts, pos = {}, 0
    for gloss, cue, _ in TERMS:
        if cue is None:
            continue
        toks = normalize_ar(cue).split()
        for i in range(pos, len(words) - len(toks) + 1):
            if all(t in words[i + k][0] for k, t in enumerate(toks)):
                starts[gloss] = words[i][1]
                pos = i + len(toks)
                break
    return starts


MAX_TERM_SECONDS = 6.0


def assign_in_order(order, starts, runs, t):
    """Terms and signing stretches both run in order. Every term gets its own stretch, later than the previous
    term's; heard terms are scored by overlap with [cue - 1 s, cue + 6 s] (never past the next cue), so one missed
    stretch cannot shift every term after it. Solved by dynamic programming over (term, stretch)."""
    heard = [g for g in order if g in starts]
    win = {}
    for k, g in enumerate(heard):
        nxt = starts[heard[k + 1]] if k + 1 < len(heard) else t[-1]
        win[g] = (starts[g] - 1.0, min(nxt, starts[g] + MAX_TERM_SECONDS))
    span = [(float(t[s]), float(t[e - 1])) for s, e in runs]

    def score(g, r):
        if g not in win:
            return 0.0  # a silent term costs nothing, it only has to fit in order
        a, b = win[g]
        return max(0.0, min(span[r][1], b) - max(span[r][0], a))

    n, m = len(order), len(runs)
    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG)
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 = term i took stretch j-1, 0 = stretch j-1 skipped
    dp[0, :] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            take = dp[i - 1, j - 1] + score(order[i - 1], j - 1)
            skip = dp[i, j - 1]
            dp[i, j], choice[i, j] = (take, 1) if take >= skip else (skip, 0)
    assigned, i, j = {}, n, m
    while i > 0 and j > 0:
        if choice[i, j] == 1:
            assigned[order[i - 1]] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return assigned


def configure(name):
    global SRC, VIDEO, OUT, CROP_X0, TERMS, DATASET, PREFIX
    src, video, out, x0, terms, dataset, prefix = CONFIGS[name]
    SRC = DATA / "asl" / src
    VIDEO, OUT, CROP_X0, TERMS, DATASET, PREFIX = SRC / video, DATA / "asl" / "lexicon" / out, x0, terms, dataset, prefix


def main():
    configure(sys.argv[1] if len(sys.argv) > 1 else "gdm")
    whisper = json.loads((SRC / "whisper.json").read_text(encoding="utf-8"))
    cache = SRC / "raw_pose.npz"
    if cache.exists():
        z = np.load(cache)
        raw, t = z["raw"], z["t"]
    else:
        raw, t = extract()
        np.savez_compressed(cache, raw=raw, t=t)
    runs = signing_runs(raw)
    starts = term_times(whisper)
    print(f"{len(runs)} signing stretches; {len(starts)} of {len(TERMS)} terms heard")

    order = [g for g, _, _ in TERMS]
    assigned = assign_in_order(order, starts, runs, t)

    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries, rows, thumbs = [], [], []
    cap = cv2.VideoCapture(str(VIDEO))
    for n, (gloss, cue, aliases) in enumerate(TERMS):
        if gloss not in assigned:
            rows.append({"gloss": gloss, "start_s": "", "end_s": "", "heard": gloss in starts, "status": "not found"})
            continue
        s, e = runs[assigned[gloss]]
        filled, missing = interpolate_missing(raw[s:e])
        clip = normalize(filled)
        sid = f"{PREFIX}_{n:03d}"
        np.save(OUT / "signs" / f"{sid}.npy", clip)
        for g in [gloss] + [a.upper() for a in aliases]:
            entries.append({"gloss": g, "sign_id": sid, "dataset": DATASET,
                            "keypoints_path": f"{OUT.name}/signs/{sid}.npy", "review_status": "pending",
                            "is_religious": True, "is_letter": False})
        rows.append({"gloss": gloss, "start_s": round(float(t[s]), 2), "end_s": round(float(t[e - 1]), 2),
                     "heard": gloss in starts, "status": f"missing hand {missing:.0%}"})
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t[(s + e) // 2]) * 1000)
        ok, img = cap.read()
        if ok:
            img = cv2.resize(img[:, CROP_X0:], (240, 180))
            cv2.putText(img, gloss[:18], (6, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(img)
    cap.release()
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    with open(OUT / "terms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if thumbs:
        while len(thumbs) % 8:
            thumbs.append(np.zeros_like(thumbs[0]))
        grid = np.vstack([np.hstack(thumbs[i:i + 8]) for i in range(0, len(thumbs), 8)])
        cv2.imwrite(str(OUT / "contact_sheet.jpg"), grid)
    found = sum(1 for r in rows if r["status"] != "not found")
    print(f"{found}/{len(TERMS)} terms cut into signs -> {OUT}")
    for r in rows:
        print(f"  {r['gloss']:20s} {r['start_s']!s:>7} - {r['end_s']!s:<7} heard={r['heard']!s:5s} {r['status']}")


if __name__ == "__main__":
    main()
