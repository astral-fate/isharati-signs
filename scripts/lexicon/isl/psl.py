"""Pakistan Sign Language (PSL) word signs from the FESF / Deaf Reach PSL dictionary (psl.org.pk) -> Urdu lexicon.

Most Urdu speakers are Pakistani, and PSL (same Indo-Pakistani family as ISL, much shared vocabulary) is their sign
language. The dictionary (Family Educational Services Foundation, Karachi) lists ~6,500 concepts in 80 categories,
each with an English title, the Urdu word (title_secondary) and a roman-Urdu spelling, and one or more videos of a
Deaf signer. Category pages carry the concepts, a word page its videos (JSON in the Next.js payload, the video files on
the site's public CloudFront bucket); no login, no robots.txt, no terms-of-use page. Footer: «© All rights reserved,
Pakistan Sign Language, FESF». Treat the signs like WSLP / ISLRTC: research and the demo only, ask FESF
(psl.org.pk/contact) before any public or commercial redistribution.

Every video has one layout: the signer on the left ~60 % of the frame, a picture and the caption on the right; the
sign is performed twice, first under the English caption, then (after the hands rest) under the Urdu caption. The
first signing run is kept (one performance), the second is used only as a consistency measure (repeat_dist). Videos
are 720p-1080p, up to 15 MB: each is downloaded, re-encoded to 960 px wide, 25 fps (~0.3 MB, kept on F:), then
tracked (scripts/face/track.py, MediaPipe Holistic, the signer's crop) and cut as scripts/lexicon/isl/islrtc.py cuts
ISLRTC videos (2 frames of padding, checks: 0.4-4 s, dominant hand >= 70 %, one signing run). The face of the same
frames goes to data/face/ur/psl_<video>.npz.

Order (catalog): Islamic terms the Urdu lexicon lacks, then corpus gap words by frequency (content words of the
Qur'an + hadith English corpus with no ISL sign, as scripts/eval/corpus_eda.py measures Urdu mode), then words new to
the lexicon, then Islamic terms it has, then the rest. Sentences and the alphabets are skipped. Several videos of one
concept: Islamic concepts get up to 3, other concepts their first video (the second when the first is rejected); the
main clip is the medoid of 3+, else the concept's own recording (video id == concept id), others are "variant_of".
The lexicon loads PSL after CISLR, WSLP and ISLRTC (isl.py), so a PSL sign never displaces an existing sign.

  PYTHONPATH=src py scripts/lexicon/isl/psl.py catalog            concepts x gap list -> F:/psl_new/pslorg/plan.csv
  "D:/islam/gathring data/.venv/Scripts/python" scripts/lexicon/isl/psl.py work <k> <n>   page, download, track, cut
  PYTHONPATH=src py scripts/lexicon/isl/psl.py entries|qa|coverage|faces
Output: data/lexicon_isl/psl/{signs/<video>.npy, entries.jsonl}, data/face/ur/psl_<video>.npz; videos, reduced tracks
(no face mesh), pages and rejects on F:/psl_new/pslorg/.
"""
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isharati.config import DATA  # noqa: E402

WORK = Path(os.environ.get("PSL_WORK", r"F:\psl_new\pslorg"))
VIDEOS, TRACKS, PAGES, TMP = WORK / "videos", WORK / "tracks", WORK / "pages", WORK / "tmp"
CONCEPTS, PLAN = WORK / "concepts.jsonl", WORK / "plan.csv"
OUT = DATA / "lexicon_isl" / "psl"
FACE = DATA / "face" / "ur"
DATASET = "psl_fesf"
SITE = "https://psl.org.pk/dictionary"
LICENCE = ("FESF / Deaf Reach PSL dictionary (psl.org.pk); (c) FESF, all rights reserved; no terms of use or robots.txt; "
           "research and demo use only, ask FESF before redistribution")
UA = {"User-Agent": "Mozilla/5.0 (research; Isharati sign-language project)"}
BUDGET = 3.6 * 1024 ** 3      # everything on F:/psl_new (videos, tracks, pages)
CROP = 0.62                   # the signer's part of the frame (left)
PAD, MIN_S, MAX_S, MIN_HAND, MAX_MISSING = 2, 0.4, 4.0, 0.7, 0.45
UNTIL = 7.0                   # seconds tracked first (the first performance is over by then in most videos)
SKIP_CATS = {7, 8, 101, 65, 81, 21}   # alphabets, sentences, fast-food menus (sentences)

# the dictionary's English titles for Islamic concepts -> the English words of the Qur'an translations, then roman Urdu
ISLAMIC_ALIASES = {
    "ablution": ["WUDU", "WUZU"], "dry ablution": ["TAYAMMUM"], "fast": ["FASTING", "ROZA", "SAWM"],
    "missed fast": ["QAZA ROZA"], "zakat": ["ZAKAH", "ALMS"], "quraan": ["QURAN", "KORAN", "QUR'AN"],
    "quranic verse": ["VERSE", "AYAH", "AYAT"], "surah": ["SURA", "CHAPTER"], "mosque": ["MASJID"],
    "heaven": ["PARADISE", "JANNAH", "JANNAT"], "hell": ["HELLFIRE", "JAHANNAM", "JAHANNUM"],
    "angel": ["FARISHTA"], "hadith": ["HADEES"], "sunnah": ["SUNNA"], "worship": ["IBADAH", "IBADAT"],
    "nabi": ["PROPHET"], "infidel": ["DISBELIEVER", "KAFIR", "UNBELIEVER"], "true believer": ["BELIEVER", "MOMIN"],
    "martyr": ["SHAHEED", "SHAHID"], "sermon": ["KHUTBAH", "KHUTBA"], "doomsday": ["QIYAMAH", "DAY OF JUDGMENT"],
    "life after death": ["HEREAFTER", "AKHIRAH", "AAKHIRAT"], "if allah wills": ["INSHALLAH", "IN SHA ALLAH"],
    "all praises to almighty allah": ["ALHAMDULILLAH"], "may allah reward you": ["JAZAKALLAH"],
    "here i am": ["LABBAIK"], "repeating divine names": ["DHIKR", "ZIKR"], "baitullah": ["KAABA", "KABAH"],
    "tawaaf-e-kaaba": ["TAWAF"], "hypocrite": ["MUNAFIQ"], "mairaj": ["MIRAJ", "ASCENSION"],
    "innovation in religion": ["BIDAH"], "alms gift": ["SADAQAH", "CHARITY"], "iqamat": ["IQAMAH"],
    "madina": ["MEDINA", "MADINAH"], "masjid e nabvi": ["PROPHET'S MOSQUE"], "islamic calendar": ["HIJRI"],
    "attributing partners to allah": ["SHIRK", "POLYTHEISM", "ASSOCIATE"], "unity of allah": ["TAWHEED", "MONOTHEISM"],
    "allah the creator": ["CREATOR", "KHALIQ"], "allah the owner": ["MALIK", "SOVEREIGN"],
    "allah the sustainer": ["SUSTAINER", "PROVIDER", "RAZZAQ"], "divine help": ["TAUFIQ"],
    "forgiveness of sins": ["FORGIVENESS"], "reward of good deeds": ["REWARD"], "deeds": ["DEED"],
    "blessings": ["BLESSING"], "prostrate": ["PROSTRATION", "SAJDAH", "SUJOOD"], "recitation": ["RECITE", "TILAWAT"],
    "companion of prophet muhammed": ["COMPANION", "SAHABI"], "companions of prophet muhammad": ["COMPANIONS", "SAHABA"],
    "ummah": ["NATION", "COMMUNITY"], "migrate": ["EMIGRATE", "HIJRAH"], "envy": ["JEALOUSY"],
    "arrogance": ["PRIDE", "ARROGANT"], "back biting": ["BACKBITING", "GHEEBAT"], "uniqueness": ["UNIQUE"],
    "sincerity": ["SINCERE", "IKHLAS"], "torment": ["PUNISHMENT", "AZAB"], "unlawful": ["HARAM", "FORBIDDEN"],
    "pure": ["PURE", "PAK"], "said": ["SAY"], "virtue": ["RIGHTEOUSNESS"], "devout": ["PIOUS", "RIGHTEOUS"],
    "salvation": ["SUCCESS", "NAJAT"], "fortune": ["DESTINY", "DECREE", "TAQDEER"], "mortal": ["PERISHABLE"],
    "self sufficient": ["FREE OF NEED"], "divine light": ["LIGHT", "NOOR"], "divine power": ["POWER", "QUDRAT"],
    "creatures": ["CREATURE", "CREATION"], "endurance": ["PATIENCE", "SABR"], "persistence": ["STEADFAST"],
    "favor": ["FAVOUR", "BOUNTY"], "embezzlement": ["BETRAYAL", "KHIYANAT"], "dissimulation": ["SHOW OFF", "RIYA"],
    "miserliness": ["STINGY", "MISER"], "humility and fear": ["HUMBLE", "HUMILITY"], "self desire": ["DESIRE"],
    "superior excellence": ["EXCELLENCE", "IHSAN"], "slave": ["SERVANT", "BONDSMAN"], "idol": ["IDOLS"],
    "translation": ["TRANSLATE"], "bani israel": ["CHILDREN OF ISRAEL", "ISRAELITES"], "jew": ["JEWS"],
    "hajr e aswad": ["BLACK STONE"], "holy water of zamzam": ["ZAMZAM"], "shab-e-qadar": ["LAYLAT AL QADR", "DECREE NIGHT"],
    "meditational solitude": ["ITIKAF"], "missed salah": ["QAZA NAMAZ"], "funeral prayer": ["JANAZAH"],
    "fajr prayer": ["FAJR", "DAWN PRAYER"], "zohr prayer": ["ZUHR", "DHUHR", "NOON PRAYER"],
    "asr prayer": ["ASR", "AFTERNOON PRAYER"], "tahajjud": ["NIGHT PRAYER"], "grave": ["QABR"],
    "hoor": ["HOURI"], "namrood": ["NIMROD"], "qaroon": ["KORAH", "QARUN"], "abu jehal": ["ABU JAHL"],
    "bait ul muqaddas": ["JERUSALEM", "AL AQSA"], "khulfa e rashideen": ["RIGHTLY GUIDED CALIPHS"],
}
PROPHETS = {"nooh": "NOAH", "isa": "JESUS", "ismail": "ISHMAEL", "ayub": "JOB", "shoaib": "SHUAYB", "yunus": "JONAH",
            "dawood": "DAVID", "adam": "ADAM", "ibrahim": "ABRAHAM", "moosa": "MOSES", "yousuf": "JOSEPH", "loot": "LOT",
            "suleman": "SOLOMON", "zakariya": "ZECHARIAH", "jibrael": "GABRIEL", "mikael": "MICHAEL",
            "izrael": "AZRAEL", "israfil": "ISRAFIL", "khizar": "KHIDR", "hawa": "EVE"}
HONORIFIC = re.compile(r"\((a\.?s|r\.?a|pbuh|sallallahu[^)]*|razi[^)]*|alaih[^)]*)\.?\)|\bra\b$", re.I)


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- the site

def get(url, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def payload(html):
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, flags=re.S)
    return "".join(json.loads('"' + c + '"') for c in chunks)


def array(s, key):
    i = s.find(f'"{key}":')
    return None if i < 0 else json.JSONDecoder().raw_decode(s[i + len(key) + 3:])[0]


def page_videos(c):
    """The concept's videos (cached in pages/<id>.json): [{id, title, title_secondary, video_url, ...}]."""
    p = PAGES / f"{c['concept']}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    html = get(f"{SITE}/{c['category']}/{c['concept']}-{c['slug']}").decode("utf-8")
    keep = ("id", "title", "title_secondary", "title_roman", "video_url", "concept_id", "sign_variant", "created_at")
    vids = [{k: v.get(k) for k in keep} for v in (array(payload(html), "videos") or [])]
    PAGES.mkdir(parents=True, exist_ok=True)
    write_atomic(p, json.dumps(vids, ensure_ascii=False))
    time.sleep(1.0)
    return vids


# ---------------------------------------------------------------- glosses and the plan

def names_of(c):
    """English glosses of a concept: the title without honorifics, alternatives split on / and , and a parenthesis
    that is not an honorific as an alias; Islamic aliases (Qur'an-translation words, roman Urdu)."""
    from isharati.text.english import normalize_en
    t = HONORIFIC.sub("", c["title"]).strip()
    paren = [p for p in re.findall(r"\(([^)]*)\)", t)]
    base = re.sub(r"\([^)]*\)", " ", t)
    alts = [a for a in re.split(r"[/,;]", base)] + paren
    names = []
    for a in alts:
        n = normalize_en(a.replace("-", " ")).strip() if a.strip() else ""
        if n and n not in names:
            names.append(n)
    key = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", c["title"])).strip().lower()
    extra = list(ISLAMIC_ALIASES.get(key, [])) if c.get("category_id") == 83 else []
    words = key.replace("prophet ", "").replace("hazrat ", "").replace("amma ", "").split()
    if c.get("category_id") == 83 and words and words[0] in PROPHETS and len(words) <= 2:
        extra.insert(0, PROPHETS[words[0]])
    out = [g.upper() for g in names]
    if extra and out and out[0].startswith(("PROPHET ", "HAZRAT ", "AMMA ")) and extra[0] in PROPHETS.values():
        out.insert(0, extra.pop(0))  # «Prophet Nooh» -> NOAH first
    for g in extra:
        g = normalize_en(g).upper() or g
        if g not in out:
            out.append(g)
    return out


def urdu_glosses():
    """Glosses of the Urdu lexicon's word sources (CISLR, WSLP, ISLRTC): single words (corpus_eda's measure) and all."""
    from isharati.text.english import normalize_en
    single, every = set(), set()
    for f in ("cislr", "wslp", "islrtc"):
        p = DATA / "lexicon_isl" / f / "entries.jsonl"
        for e in (load(p) if p.exists() else []):
            if e.get("is_letter"):
                continue
            g = e["gloss"].lower().strip()
            every.add(normalize_en(g))
            if " " not in g:
                single.add(g)
    return single, every


def corpus_counts():
    from islrtc import corpus_counts as cc
    return cc()


def catalog():
    from islrtc import RELIGIOUS
    from isharati.text.english import STOPWORDS, candidates, normalize_en
    concepts = load(CONCEPTS)
    single, every = urdu_glosses()
    c_all, _, cap = corpus_counts()
    content = {w: n for w, n in c_all.items() if w not in STOPWORDS}
    missing = {w: n for w, n in content.items() if not any(x in single for x in candidates(w))}
    by_form = {}
    for w in missing:
        for f in candidates(w) + [w]:
            by_form.setdefault(f, set()).add(w)
    rows = []
    for c in concepts:
        if c["category_id"] in SKIP_CATS or not c.get("title"):
            continue
        names = names_of(c)
        if not names or len(names[0].split()) > 4 or "?" in c["title"]:
            continue
        toks = {x for n in names for w in n.lower().split() for x in candidates(w)}
        islamic = c["category_id"] == 83 or bool((toks & RELIGIOUS) - {"fast", "grave"} and c["category_id"] in (43, 30))
        gap = 0
        gap_word = ""
        for n in names:
            n = n.lower()
            if " " in n:
                continue
            ws = set(by_form.get(n, ())) | {w for x in candidates(n) for w in by_form.get(x, ())}
            ws = {w for w in ws if n in candidates(w) or n in (w, w + "s", w + "es") or w in candidates(n)}
            for w in ws:
                if missing[w] > gap:
                    gap, gap_word = missing[w], w
        new = not any(normalize_en(n) in every for n in names)
        tier = 0 if islamic and (new or gap) else 1 if gap else 2 if new else 3 if islamic else 4
        rows.append({"concept": c["id"], "category": c["category"], "slug": c["slug"], "title": c["title"],
                     "urdu": c.get("title_secondary") or "", "urdu_roman": c.get("title_roman") or "",
                     "glosses": "|".join(names), "islamic": int(islamic), "tier": tier, "gap_word": gap_word,
                     "gap_count": gap})
    rows.sort(key=lambda r: (r["tier"], -r["gap_count"], r["concept"]))
    for k, r in enumerate(rows):
        r["priority"] = k
    with open(PLAN.with_suffix(".tmp"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(PLAN.with_suffix(".tmp"), PLAN)
    t = Counter(r["tier"] for r in rows)
    names = {0: "Islamic, new to the Urdu lexicon", 1: "corpus gap word", 2: "word new to the lexicon",
             3: "Islamic, already in the lexicon", 4: "word the lexicon has"}
    print(f"{len(rows)} concepts planned ({len(concepts)} in the dictionary) -> {PLAN}")
    for k in sorted(t):
        print(f"  tier {k} ({names[k]}): {t[k]}")
    print("gap occurrences the plan can fill:", sum(missing[w] for w in {r['gap_word'] for r in rows if r['gap_word']}),
          "of", sum(missing.values()), "missing,", sum(content.values()), "content")


def plan():
    return list(csv.DictReader(open(PLAN, encoding="utf-8")))


# ---------------------------------------------------------------- per video

def disk_used():
    total = 0
    for d in (VIDEOS, TRACKS, PAGES, TMP):
        for p in (d.iterdir() if d.exists() else ()):
            try:
                total += p.stat().st_size
            except OSError:  # a temp file renamed meanwhile
                pass
    return total


def fetch_video(v):
    """Download (tmp), re-encode to 960 px / 25 fps without audio into videos/<id>.mp4, drop the original."""
    out = VIDEOS / f"{v['id']}.mp4"
    if out.exists():
        return out
    VIDEOS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    import threading
    tag = f"{os.getpid()}_{threading.get_ident()}"  # the prefetcher and a worker may fetch one video at once
    raw = TMP / f"{v['id']}.{tag}.src.mp4"
    raw.write_bytes(get(v["video_url"]))
    tmp = TMP / f"{v['id']}.{tag}.mp4"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vf", "scale=960:-2,fps=25", "-c:v", "libx264",
                        "-crf", "26", "-preset", "veryfast", "-an", str(tmp)], capture_output=True, text=True)
    raw.unlink(missing_ok=True)
    if r.returncode or not tmp.exists():
        raise RuntimeError("ffmpeg: " + r.stderr[-200:])
    os.replace(tmp, out)
    return out


def load_track(vid):
    z = np.load(TRACKS / f"{vid}.npz")
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def clip_span(raw, fps=25):
    """The first performance: the first signing run of >= 10 frames (the video signs the word twice, under the English
    and then the Urdu caption, with the hands lowered between), cut as islrtc.clip_span cuts a run (fingerspelling
    tail, a repetition without lowering the hands, a final hold), padded; then the checks. Returns (a, b, reason,
    second run or None)."""
    from islrtc import fingerspelling_start, first_repetition, signing_runs, trim_hold
    from isharati.pose.keypoints import LH, R_SH, RH
    if np.isnan(raw[:, R_SH, 0]).all():
        return None, None, "no shoulders", None
    runs = [(s, e) for s, e in signing_runs(raw) if e - s >= 10]
    if not runs:
        return None, None, "no signing run", None
    a, b = runs[0]
    second = runs[1] if len(runs) > 1 else None
    if second and second[0] - b < int(0.3 * fps):
        return a, b, f"first run not separated from the next ({(second[0] - b) / fps:.2f} s rest)", second
    fs = fingerspelling_start(raw, a, b)
    if fs is not None:
        if fs - a < int(MIN_S * fps):
            fs = None  # a fingerspelled word (letters are its sign): kept whole
        else:
            b = fs
    b = first_repetition(raw, a, b)
    a, b = trim_hold(raw, a, b)
    a, b = max(0, a - PAD), min(len(raw), b + PAD)
    seg = raw[a:b]
    dom = max((~np.isnan(seg[:, LH, 0]).all(axis=1)).mean(), (~np.isnan(seg[:, RH, 0]).all(axis=1)).mean())
    n_runs = len([r for r in signing_runs(seg) if r[1] - r[0] >= 6])
    dur = (b - a) / fps
    if dur < MIN_S:
        return a, b, f"too short ({dur:.2f} s)", second
    if dur > MAX_S:
        return a, b, f"too long ({dur:.2f} s)", second
    if dom < MIN_HAND:
        return a, b, f"dominant hand seen in {dom:.0%} of frames", second
    if n_runs != 1:
        return a, b, f"{n_runs} signing runs in the span", second
    return a, b, None, second


def run_dist(raw, r1, r2):
    """DTW distance (shoulder widths, wrists + hand centroids) between two runs: how alike the two performances are."""
    from islrtc import hand_features
    f = hand_features(raw)
    x, y = f[r1[0]:r1[1]], f[r2[0]:r2[1]]
    x, y = np.nan_to_num(x - np.nanmean(x, 0)), np.nan_to_num(y - np.nanmean(y, 0))
    n, m = len(x), len(y)
    if not n or not m:
        return None
    d = np.abs(x[:, None, :] - y[None, :, :]).mean(-1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = d[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def process(r, v):
    """One video of a concept: fetch, track (once), cut, save clip + face. Returns "ok" | "rejected"."""
    sys.path.insert(0, str(ROOT / "scripts" / "face"))
    from attach import fifty, save_face
    from isharati.pose.keypoints import interpolate_missing, normalize
    vid = str(v["id"])
    npy, bad = OUT / "signs" / f"{vid}.npy", OUT / "signs" / f"{vid}.failed"
    if npy.exists():
        return "ok"
    if bad.exists():
        return "rejected"
    video = fetch_video(v)
    tr_path = TRACKS / f"{vid}.npz"
    if not tr_path.exists():
        import cv2
        from track import track
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        # the first performance ends within the first seconds: track UNTIL s first, the whole video only when the
        # first run is not over (followed by rest) inside that stretch
        tr, meta = track(video, crop=(0, 0, int(w * CROP), h), until=UNTIL if dur > UNTIL + 2 else None)
        if dur > UNTIL + 2:
            a, b, reason, second = clip_span(fifty(tr))
            if a is None or b >= len(tr["time"]) - int(0.3 * 25) - PAD:
                tr, meta = track(video, crop=(0, 0, int(w * CROP), h))
        small = {k: tr[k] for k in ("pose", "left_hand", "right_hand", "blend", "time")}  # no face mesh on F:
        TRACKS.mkdir(parents=True, exist_ok=True)
        tmp = tr_path.with_name(tr_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **small)
        os.replace(tmp, tr_path)
    else:
        tr, meta = load_track(vid)
    raw = fifty(tr)
    a, b, reason, second = clip_span(raw)
    filled = missing = None
    if reason is None:
        filled, missing = interpolate_missing(raw[a:b])
        if missing > MAX_MISSING:
            reason = f"missing hand {missing:.2f}"
    if reason is None and "face" not in tr:  # re-cut from a reduced track: the face needs the full one
        from track import track
        import cv2
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        tr, meta = track(video, crop=(0, 0, int(w * CROP), h))
    t = tr["time"]
    if reason is not None:
        bad.write_text(json.dumps({"concept": r["concept"], "gloss": r["glosses"].split("|")[0], "reason": reason,
                                   "start": None if a is None else round(float(t[a]), 3),
                                   "end": None if b is None else round(float(t[b - 1]), 3)}), encoding="utf-8")
        return "rejected"
    clip = normalize(filled).astype(np.float32)
    hands = float(np.mean(~(np.isnan(raw[a:b, 8:29, 0]).all(1) & np.isnan(raw[a:b, 29:50, 0]).all(1))))
    rep = None
    if second is not None:
        try:
            rep = run_dist(raw, (a, b), second)
        except Exception:
            rep = None
    row = {"sign_id": f"psl_{vid}", "video": video.name, "start": round(float(t[a]), 3),
           "end": round(float(t[b - 1]), 3)}
    save_face(FACE, row["sign_id"], tr, raw, meta, a, b - a, 0.0, row, clip)
    np.save(npy.with_name(npy.stem + ".tmp.npy"), clip)
    os.replace(npy.with_name(npy.stem + ".tmp.npy"), npy)
    write_atomic(npy.with_suffix(".json"), json.dumps({
        "concept": int(r["concept"]), "video_id": int(vid), "missing": round(float(missing), 3), "hands": round(hands, 3),
        "start": row["start"], "end": row["end"], "frames": b - a, "face_detected": row["face_detected"],
        "repeat_dist": None if rep is None else round(rep, 3), "url": v["video_url"]}))
    return "ok"


def choose(r, vids):
    """Which of a concept's videos to process, in order: its own recording (id == concept id) first."""
    vids = sorted(vids, key=lambda v: (v["id"] != int(r["concept"]), v["id"]))
    return vids[:3] if r["islamic"] == "1" else vids[:2]


def work(k, n):
    """Worker k of n over the plan (rows i % n == k), until the disk budget is used."""
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    FACE.mkdir(parents=True, exist_ok=True)
    flag = WORK / f"WORKING_{k}"
    flag.write_text("running", encoding="utf-8")
    stats = Counter()
    try:
        for i, r in enumerate(plan()):
            if i % n != k:
                continue
            done = WORK / "done" / f"{r['concept']}"
            if done.exists():
                continue
            if disk_used() > BUDGET:
                print("disk budget reached", flush=True)
                break
            try:
                vids = page_videos(r)
            except Exception as e:
                print(r["concept"], "page error:", repr(e)[:200], flush=True)
                stats["page error"] += 1
                continue
            got, errored = 0, False
            for v in choose(r, vids):
                if not v.get("video_url"):
                    continue
                if got and r["islamic"] != "1":
                    break  # non-Islamic concepts: one accepted clip
                try:
                    s = process(r, v)
                except Exception as e:
                    s = "error"
                    print(r["concept"], v["id"], "error:", repr(e)[:200], flush=True)
                    # an error (out of memory, a broken download) is not a reject: the concept is retried next run
                    (WORK / "errors").mkdir(exist_ok=True)
                    (WORK / "errors" / f"{v['id']}.txt").write_text(f"{r['concept']} {e!r}", encoding="utf-8")
                stats[s] += 1
                got += s == "ok"
                errored |= s == "error"
            if got or not errored:
                done.parent.mkdir(exist_ok=True)
                done.write_text(str(got), encoding="utf-8")
            if sum(stats.values()) % 20 == 0:
                print(f"{time.strftime('%H:%M')} worker {k}/{n} row {i}: {dict(stats)}; F: {disk_used() / 1e9:.2f} GB",
                      flush=True)
    finally:
        flag.unlink(missing_ok=True)
    print(f"worker {k}/{n} finished: {dict(stats)}", flush=True)


def prefetch(threads=3, ahead=150):
    """No MediaPipe: fetch pages and re-encoded videos ahead of the workers (plan order, at most `ahead` concepts past
    the last finished one), so tracking never waits on the network."""
    from concurrent.futures import ThreadPoolExecutor
    rows = plan()

    def one(r):
        try:
            for v in choose(r, page_videos(r))[:3 if r["islamic"] == "1" else 1]:
                if v.get("video_url") and not (OUT / "signs" / f"{v['id']}.npy").exists():
                    fetch_video(v)
        except Exception as e:
            print(r["concept"], "prefetch error:", repr(e)[:150], flush=True)

    with ThreadPoolExecutor(threads) as ex:
        i = 0
        while i < len(rows) and disk_used() < BUDGET:
            done = {p.name for p in (WORK / "done").iterdir()} if (WORK / "done").exists() else set()
            last = max((k for k, r in enumerate(rows) if r["concept"] in done), default=0)
            if i > last + ahead:
                time.sleep(20)
                continue
            batch = [r for r in rows[i:i + 30] if r["concept"] not in done]
            list(ex.map(one, batch))
            i += 30
            if i % 300 == 0:
                print(f"{time.strftime('%H:%M')} prefetched to row {i}; F: {disk_used() / 1e9:.2f} GB", flush=True)
    print("prefetch finished", flush=True)


# ---------------------------------------------------------------- entries

def clip_info():
    out = {}
    for p in (OUT / "signs").glob("*.json"):
        if (OUT / "signs" / f"{p.stem}.npy").exists():
            out.setdefault(json.loads(p.read_text(encoding="utf-8"))["concept"], []).append(
                (p.stem, json.loads(p.read_text(encoding="utf-8"))))
    return out


def medoid(stems):
    """The clip closest on average to the concept's other clips (DTW of normalised hand features)."""
    from islrtc import hand_features
    feats = [np.nan_to_num(hand_features(np.load(OUT / "signs" / f"{s}.npy"))) for s in stems]

    def dtw(x, y):
        d = np.abs(x[:, None] - y[None]).mean(-1)
        D = np.full((len(x) + 1, len(y) + 1), np.inf)
        D[0, 0] = 0
        for i in range(1, len(x) + 1):
            for j in range(1, len(y) + 1):
                D[i, j] = d[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D[-1, -1] / (len(x) + len(y))
    tot = [sum(dtw(feats[i], feats[j]) for j in range(len(stems)) if j != i) for i in range(len(stems))]
    return stems[int(np.argmin(tot))]


def entries():
    """entries.jsonl from every accepted clip in plan order, written atomically. Per concept one main sign (the
    medoid of 3+ clips, else its own recording) with a row per gloss (a gloss taken by an earlier concept is left
    out), the other clips as "variant_of" rows. Returns (signs, variants, rows)."""
    info = clip_info()
    out, taken = [], set()
    for r in plan():
        clips = info.get(int(r["concept"]))
        if not clips:
            continue
        stems = [s for s, _ in sorted(clips, key=lambda c: (c[1]["video_id"] != int(r["concept"]), c[1]["video_id"]))]
        main = medoid(stems) if len(stems) >= 3 else stems[0]
        base = lambda s: {"sign_id": f"psl_{s}", "dataset": DATASET, "keypoints_path": f"psl/signs/{s}.npy",
                          "review_status": "pending", "is_religious": r["islamic"] == "1", "is_letter": False,
                          "sign_language": "PSL", "urdu": r["urdu"], "urdu_roman": r["urdu_roman"],
                          "english": r["title"], "category": r["category"], "concept_id": int(r["concept"]),
                          "source_url": f"{SITE}/{r['category']}/{r['concept']}-{r['slug']}"}
        names = [g for g in r["glosses"].split("|") if g and g not in taken]
        for g in names:
            taken.add(g)
            out.append({"gloss": g, **base(main)})
        first = (names or [r["glosses"].split("|")[0]])[0]
        for s in stems:
            if s != main:
                out.append({"gloss": first, **base(s), "variant_of": f"psl_{main}"})
    OUT.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT / "entries.jsonl", "\n".join(json.dumps(e, ensure_ascii=False) for e in out) + "\n")
    signs = len({e["sign_id"] for e in out if "variant_of" not in e})
    return signs, sum("variant_of" in e for e in out), len(out)


def faces():
    """The PSL signs' rows in data/face/ur/manifest.jsonl (attach.py's format), other datasets' rows kept: read and
    rewritten at once (run when no other builder is rewriting the manifest)."""
    path = FACE / "manifest.jsonl"
    keep = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path.exists() else []
    before = len(keep)
    keep = [m for m in keep if m.get("dataset") != DATASET]
    gl = {e["sign_id"]: e["gloss"] for e in load(OUT / "entries.jsonl")}
    for cid, clips in clip_info().items():
        for s, i in clips:
            if (FACE / f"psl_{s}.npz").exists() and f"psl_{s}" in gl:
                keep.append({"sign_id": f"psl_{s}", "gloss": gl[f"psl_{s}"], "dataset": DATASET, "video": f"{s}.mp4",
                             "start": i["start"], "end": i["end"], "frames": i["frames"], "match": 0.0, "found": True,
                             "face_detected": i["face_detected"]})
    write_atomic(path, "\n".join(json.dumps(m, ensure_ascii=False) for m in keep) + "\n")
    print(f"face manifest: {before} rows before, {len(keep)} after")


# ---------------------------------------------------------------- QA and coverage

def contact_sheet(video, times, labels, out, title="", width=200):
    import cv2
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for t, lab in zip(times, labels):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, img = cap.read()
        if not ok:
            img = np.zeros((540, 960, 3), np.uint8)
        img = img[:, :int(img.shape[1] * CROP)]
        h, w = img.shape[:2]
        img = cv2.resize(img, (width, int(h * width / w)))
        cv2.rectangle(img, (0, 0), (width, 20), (0, 0, 0), -1)
        cv2.putText(img, f"{lab} {t:.2f}s", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cap.release()
    sheet = np.concatenate(tiles, axis=1)
    bar = np.zeros((24, sheet.shape[1], 3), np.uint8)
    cv2.putText(bar, title[:110], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.concatenate([bar, sheet], axis=0))


def qa(n=20, seed=0):
    """Contact sheets of n random accepted clips (before, start, 3 inside, end, after) and of the rejects' spans, in
    F:/psl_new/pslorg/qa/; rejects.csv lists every reject with its reason."""
    out = WORK / "qa"
    out.mkdir(parents=True, exist_ok=True)
    gl = {e["sign_id"]: e for e in load(OUT / "entries.jsonl")}
    ok = sorted(p.stem for p in (OUT / "signs").glob("*.npy") if f"psl_{p.stem}" in gl)
    pick = random.Random(seed).sample(ok, min(n, len(ok)))
    for k, s in enumerate(pick):
        i = json.loads((OUT / "signs" / f"{s}.json").read_text(encoding="utf-8"))
        a, b = i["start"], i["end"]
        e = gl[f"psl_{s}"]
        contact_sheet(VIDEOS / f"{s}.mp4", [a - 0.3, a, a + (b - a) / 4, (a + b) / 2, a + 3 * (b - a) / 4, b, b + 0.3],
                      ["before", "start", "", "mid", "", "end", "after"], out / f"accepted_{k:02d}_{s}.jpg",
                      f"{e['gloss']} ({e['urdu']}) {'variant' if 'variant_of' in e else ''} kept {a:.2f}-{b:.2f}s "
                      f"hands {i['hands']:.0%} rep {i['repeat_dist']}")
    rej = []
    for p in (OUT / "signs").glob("*.failed"):
        d = json.loads(p.read_text(encoding="utf-8"))
        rej.append({"video_id": p.stem, **d})
    with open(out / "rejects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "concept", "gloss", "reason", "start", "end"])
        w.writeheader()
        w.writerows(rej)
    print(f"{len(pick)} contact sheets, {len(rej)} rejects -> {out}")
    print(Counter(re.sub(r"[\d.]+", "#", r["reason"]) for r in rej).most_common(12))


def coverage_report(out_path=None):
    """corpus_eda's isl_on_en measure (English corpus, single-word glosses, candidate forms) without and with PSL."""
    from islrtc import coverage
    c_all, c_ur, _ = corpus_counts()
    single, _ = urdu_glosses()
    psl = {e["gloss"].lower().strip() for e in load(OUT / "entries.jsonl") if not e.get("variant_of")} \
        if (OUT / "entries.jsonl").exists() else set()
    after = single | {g for g in psl if " " not in g}
    res = {"glosses_before": len(single), "glosses_after": len(after), "psl_glosses": len(psl),
           "psl_new_single_glosses": len(after - single)}
    for name, c in (("english_corpus_all", c_all), ("urdu_mode_passages", c_ur)):
        res[name] = {"before": coverage(c, single), "after": coverage(c, after)}
    out_path = Path(out_path or WORK / "coverage.json")
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "catalog":
        catalog()
    elif mode == "work":
        work(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "prefetch":
        prefetch()
    elif mode == "collect":  # while the workers run: entries.jsonl every 5 min; at the end, the face manifest too
        time.sleep(120)
        while any(WORK.glob("WORKING_*")):
            print(time.strftime("%H:%M"), "%d signs, %d variants, %d rows" % entries(), flush=True)
            time.sleep(300)
        print(time.strftime("%H:%M"), "final: %d signs, %d variants, %d rows" % entries(), flush=True)
        faces()
        coverage_report()
    elif mode == "entries":
        print("%d signs, %d variants, %d rows" % entries())
    elif mode == "faces":
        faces()
    elif mode == "qa":
        qa()
    elif mode == "coverage":
        coverage_report(sys.argv[2] if len(sys.argv) > 2 else None)
