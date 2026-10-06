"""Indian Sign Language word signs from the ISLRTC dictionary on YouTube -> ISL lexicon (data/lexicon_isl/islrtc/).

The Indian Sign Language Research and Training Centre (Govt of India, DEPwD) publishes its ISL dictionary (~10,000
terms, signed by Deaf signers) on its YouTube channel, one word per video, titled with the English word («Ability»,
«marry, wed», «bad (sign 2)»). CISLR was cut from an earlier part of the same dictionary. Every dictionary video is
taken, in priority order (catalog): the gap list first (content words of the Qur'an + hadith corpus that no ISL gloss
covers, measured as scripts/eval/corpus_eda.py measures Urdu mode: English corpus of the same passages, single-word
glosses, the English normaliser's candidate forms; ranked by frequency), then Islamic words, then words the lexicon
lacks, then the rest. The ISL lexicon loads CISLR and WSLP first, so a word they have keeps its sign; a second clip of
a gloss becomes a variant row ("variant_of"), which the lexicon skips.

Licence: the ISLRTC channel gives no open licence (standard YouTube terms; the centre makes the dictionary freely
available for learning ISL). Treat these signs like WSLP: research and the demo only, ask ISLRTC before any public or
commercial redistribution (ISHARATI_URDU_ISLRTC=0 turns them off).

Video structure (10 videos inspected, F:/isl_new/islrtc/qa/overview_*.jpg): the everyday/academic/legal dictionary
entries are 3-8 s: rest, ONE performance of the sign (sometimes held), rest; the English word is captioned top left.
Longer entries (15-60 s; financial, mental-health, academic terms) add a title card («Explanation of ...»), a definition
or an example sentence signed after the sign, or several synonyms signed in a row. Segmentation (clip_span): the first
signing run (hand above waist level); rejected when further signing follows within 1.5 s (which run is the sign is
ambiguous); fingerspelling inside the run is cut off; a repetition is cut to the first performance; a final hold is cut
to 8 frames; 2 frames of padding. Checks: 0.4-4 s, dominant hand in 70% of frames, exactly one signing run; failures
are written as signs/<id>.failed with the reason. Each video goes once through the MediaPipe Holistic task
(scripts/face/track.py, tracks kept on F:); the clip is normalize(interpolate_missing()) of the kept frames and the
face of exactly those frames is saved as data/face/ur/islrtc_<id>.npz (scripts/face/attach.py's format).

  gaps          missing content words -> F:/isl_new/missing_isl.csv
  list          channel listing x gap list -> F:/isl_new/islrtc/manifest.csv
  catalog       every dictionary video, prioritised -> F:/isl_new/islrtc/manifest_all.csv
  download      the gap-list manifest (first pass)
  download_all  worker k of n over manifest_all.csv, until 7 GB of videos (ISLRTC_NO_COOKIES=1: anonymous)
  extract       worker k of n: track + clip + face per downloaded video, loops while downloads run (MediaPipe venv)
  collect       rewrite entries.jsonl (atomically) every minute while the workers run; coverage every 200 signs
  entries       entries.jsonl now;  coverage  before/after;  qa  contact sheets + skeleton videos;  overview <ids>

  PYTHONPATH=src py scripts/lexicon/isl/islrtc.py gaps|list|catalog|download|entries|collect|coverage|qa
  PYTHONPATH=src py scripts/lexicon/isl/islrtc.py download_all <k> <n>
  "D:/islam/gathring data/.venv/Scripts/python" scripts/lexicon/isl/islrtc.py extract <k> <n>
Output: data/lexicon_isl/islrtc/{signs/<id>.npy, <id>.json, <id>.failed, entries.jsonl}, data/face/ur/islrtc_<id>.npz
and manifest.jsonl rows; videos, tracks, manifests and QA on F:/isl_new/islrtc/.
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402

WORK = Path(os.environ.get("ISLRTC_WORK", r"F:\isl_new"))
SRC = WORK / "islrtc"
VIDEOS, TRACKS = SRC / "videos", SRC / "tracks"
GAPS = WORK / "missing_isl.csv"
MANIFEST = SRC / "manifest.csv"          # the gap-list words (step 1)
MANIFEST_ALL = SRC / "manifest_all.csv"  # every dictionary video, gap and Islamic words first (catalog)
OUT = DATA / "lexicon_isl" / "islrtc"
FACE = DATA / "face" / "ur"
CHANNEL = "UC3AcGIlqVI4nJWCwHgHFXtg"  # Indian Sign Language Research and Training Centre
YTDLP = Path(r"D:\islam\yt-dlp")
COOKIES = Path(r"D:\islam\cookies\cookies_2.txt")
LICENCE = "ISLRTC (Govt of India) YouTube channel; no open licence stated, standard YouTube terms; research use"
MAX_VIDEOS, MAX_BYTES = 1500, 6 * 1024 ** 3   # the gap-list pass (download)
BUDGET = 7 * 1024 ** 3                         # all videos on F: (download_all stops there)
MAX_SECONDS = 60      # longer videos are lessons, not dictionary entries
MAX_MISSING = 0.45    # as for the other YouTube dictionaries: blurred fast frames lose the hand, they are filled in
PAD = 2
UNTIL = 12            # long videos (explanations after the sign) are tracked for their first 12 s only
RELIGIOUS = set("""allah god prophet messenger quran islam muslim pray prayer mosque angel paradise heaven hell fast
fasting charity hajj pilgrimage worship faith believe believer belief sin forgive forgiveness mercy holy soul spirit
devil satan jinn resurrection judgement judgment hereafter revelation scripture verse religion religious blessing
bless sacred sacrifice mosque namaz eid ramadan zakat imam mecca kaaba halal haram dua sunnah hadith hijab burqa
covenant oath grave tomb righteous pious obey obedience reward punishment repent repentance merciful creator creation
miracle""".split())


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def isl_glosses(with_islrtc=False):
    """Single-word ISL glosses, lower case, as corpus_eda.py collects them."""
    out = set()
    for f in ("wslp", "cislr") + (("islrtc",) if with_islrtc else ()):
        p = DATA / "lexicon_isl" / f / "entries.jsonl"
        if p.exists():
            for e in load(p):
                g = e["gloss"].lower().strip()
                if " " not in g:
                    out.add(g)
    return out


def corpus_counts():
    """Word counts of the English corpus (all passages, as corpus_eda.py) and of the passages Urdu mode answers from,
    and how often each word is capitalised (names)."""
    from isharati.text.english import normalize_en
    ur = {r["pid"] for r in load(DATA / "ur" / "corpus.jsonl")}
    words = lambda t: [w for w in normalize_en(t).split() if re.fullmatch(r"[a-z][a-z']*", w)]
    c_all, c_ur, cap = Counter(), Counter(), Counter()
    for r in load(DATA / "asl" / "corpus.jsonl"):
        ws = words(r["text"])
        c_all.update(ws)
        if r["pid"] in ur:
            c_ur.update(ws)
        for m in re.finditer(r"\b([A-Z][\w’‘ʿʾ'-]*)", r["text"]):
            cap.update(words(m.group(1)))
    return c_all, c_ur, cap


def coverage(counts, glosses):
    from isharati.text.english import STOPWORDS, candidates
    content = {w: n for w, n in counts.items() if w not in STOPWORDS}
    cov = lambda w: any(x in glosses for x in candidates(w))
    tok = sum(content.values())
    hit = sum(n for w, n in content.items() if cov(w))
    types = sum(1 for w in content if cov(w))
    return {"content_tokens": tok, "covered_tokens": hit, "token_coverage": round(hit / max(1, tok), 4),
            "content_types": len(content), "covered_types": types, "type_coverage": round(types / max(1, len(content)), 4)}


def gaps():
    from isharati.text.english import STOPWORDS, candidates
    c_all, c_ur, cap = corpus_counts()
    isl = isl_glosses()
    content = {w: n for w, n in c_all.items() if w not in STOPWORDS}
    total = sum(content.values())
    missing = [(w, n) for w, n in Counter(content).most_common() if not any(x in isl for x in candidates(w))]
    WORK.mkdir(parents=True, exist_ok=True)
    cum = 0
    with open(GAPS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "count", "count_ur_passages", "share", "cum_share_of_missing", "cum_share_of_content",
                    "likely_name", "forms"])
        miss_tok = sum(n for _, n in missing)
        for k, (word, n) in enumerate(missing, 1):
            cum += n
            name = cap[word] >= 0.9 * n and n >= 2
            w.writerow([k, word, n, c_ur[word], round(n / total, 5), round(cum / miss_tok, 4), round(cum / total, 4),
                        int(name), "|".join(candidates(word))])
    top = missing[:50]
    print(f"{len(missing)} missing content words, {sum(n for _, n in missing)} of {total} content occurrences "
          f"({sum(n for _, n in missing) / total:.1%}); top 50 = {sum(n for _, n in top)} "
          f"({sum(n for _, n in top) / total:.1%} of content, {sum(n for _, n in top) / miss_tok:.1%} of missing)")
    print(", ".join(f"{w} ({n})" for w, n in top))
    print("->", GAPS)


def ytdlp(*args, cookies=COOKIES, **kw):
    """yt-dlp writes its cookie jar back on exit: parallel workers each pass their own copy (cookies=...);
    cookies=None (or ISLRTC_NO_COOKIES=1) goes without: when the cookie account is throttled ("Video unavailable" for
    every video) anonymous requests still work."""
    jar = [] if cookies is None or os.environ.get("ISLRTC_NO_COOKIES") == "1" else ["--cookies", str(cookies)]
    return subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", *jar,
                           "--js-runtimes", "node", "--remote-components", "ejs:github", *args],
                          cwd=YTDLP, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def title_gloss(title):
    """The English word of a dictionary title («Ability», «Angel (Farishta)», «Eat - Food»): the first alternative,
    lower case; None for module / lesson / announcement titles."""
    t = re.sub(r"\([^)]*\)|\[[^]]*\]|#\S+|\bISL\b|Indian Sign Language|ISLRTC", " ", title, flags=re.I)
    parts = [re.sub(r"\s+", " ", p).strip(" -–:.,/|'\"") for p in re.split(r"[/,;|]| - | – ", t)]
    parts = [p.lower() for p in parts if re.fullmatch(r"[A-Za-z][A-Za-z' -]*", p or "") and len(p) <= 30]
    if not parts or re.search(r"\b(module|lesson|chapter|episode|part|unit|webinar|news|day|week)\b \d", title, re.I):
        return None
    return parts


def matches(alt, word):
    """A title word names a gap word: the same word, one of the gap word's lemma guesses («blessings» -> blessing,
    «said» -> say), or its plain plural («angels» for angel). Not the other way round in general: «evening» is not
    «even»."""
    from isharati.text.english import candidates
    return alt in candidates(word) or alt in (word + "s", word + "es", word[:-1] + "ies")


def list_videos():
    """The channel's videos (the videos tab and the dictionary playlists, which hold unlisted entries) matched against
    the gap list."""
    from isharati.text.english import candidates, normalize_en
    listing = SRC / "channel_videos.tsv"
    if not listing.exists():
        r = ytdlp("--flat-playlist", "--print", "%(id)s	%(duration)s	%(title)s",
                  f"https://www.youtube.com/channel/{CHANNEL}/videos")
        listing.write_text(r.stdout, encoding="utf-8")
    gap, by_form = {}, {}
    for row in csv.DictReader(open(GAPS, encoding="utf-8")):
        if row["likely_name"] == "0":
            gap[row["word"]] = (int(row["rank"]), row["word"], int(row["count"]))
            for form in row["forms"].split("|"):
                by_form.setdefault(form, set()).add(row["word"])
    lines = {}
    for f in [listing] + sorted(SRC.glob("pl_*.tsv")):
        for line in f.read_text(encoding="utf-8").splitlines():
            lines.setdefault(line.split("	")[0], line)
    rows = []
    for line in lines.values():
        vid, dur, title = (line.split("	") + ["", ""])[:3]
        try:
            dur = float(dur)
        except ValueError:
            continue
        names = title_gloss(title)
        if not names or dur > MAX_SECONDS:
            continue
        hits = []  # every single-word alternative of the title («marry, wed») is tried; the first one ranks first
        for k, alt in enumerate(names):
            alt = normalize_en(alt)
            if not alt or " " in alt:
                continue
            words = set(by_form.get(alt, ())) | {w for c in candidates(alt) for w in by_form.get(c, ())}
            hits += [(gap[w], k, alt) for w in words if matches(alt, w)]
        if not hits:
            continue
        (rank, word, count), k, name = min(hits)
        variant = re.search(r"\(sign (\d)\)", title, re.I)
        qualified = bool(re.search(r"\((?!sign \d)[^)]*\)", title, re.I))  # «pass (ticket)»: a narrower sense
        aliases = [normalize_en(a) for a in names]
        aliases = [a for a in aliases if a and " " not in a and a != name]
        rows.append({"id": vid, "url": f"https://youtu.be/{vid}", "title": title, "gloss": name.upper(),
                     "aliases": "|".join(a.upper() for a in aliases),
                     "gap_word": word, "gap_rank": rank, "gap_count": count, "seconds": dur,
                     "file": "", "licence": LICENCE,
                     "_key": (int(variant.group(1)) > 1 if variant else False, qualified, k > 0, dur)})
    # per gap word: the plain title or «(sign 1)», unqualified, the word first in the title, then the shortest video;
    # at most 2 videos per word (the second is the fallback when the first is rejected)
    rows.sort(key=lambda r: (r["gap_rank"], r["_key"]))
    per = Counter()
    keep = []
    for r in rows:
        per[r["gap_word"]] += 1
        if per[r["gap_word"]] <= 2:
            r.pop("_key")
            r["choice"] = per[r["gap_word"]]
            keep.append(r)
    rows = keep
    write_manifest(rows)
    print(f"{len(rows)} videos match {len({r['gap_word'] for r in rows})} gap words "
          f"({sum(gap[w][2] for w in {r['gap_word'] for r in rows})} occurrences) -> {MANIFEST}")


FIELDS = ["id", "url", "title", "gloss", "aliases", "gap_word", "gap_rank", "gap_count", "choice", "seconds", "file", "licence"]


def write_manifest(rows):
    SRC.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, MANIFEST)


def manifest(path=None):
    """The full catalogue (manifest_all.csv, priority order) when it exists, else the gap-list manifest."""
    path = path or (MANIFEST_ALL if MANIFEST_ALL.exists() else MANIFEST)
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def video_of(vid):
    return next((p for p in VIDEOS.glob(f"{vid}.*") if p.suffix in (".mp4", ".webm", ".mkv")), None)


def download(batch=20, pause=(1, 3), backoff=1800):
    """Most frequent gap words first, in small batches with pauses, until MAX_VIDEOS or MAX_BYTES; resumable."""
    VIDEOS.mkdir(parents=True, exist_ok=True)
    rows = manifest(MANIFEST)
    gone_file = VIDEOS / "unavailable.txt"
    gone = set(gone_file.read_text(encoding="utf-8").split()) if gone_file.exists() else set()
    size = lambda: sum(p.stat().st_size for p in VIDEOS.glob("*") if p.suffix in (".mp4", ".webm", ".mkv"))
    have = lambda: sum(1 for r in rows if video_of(r["id"]))
    order = sorted(rows, key=lambda r: (int(r["choice"]), int(r["gap_rank"])))  # every word's first video first
    todo = [r["id"] for r in order if r["id"] not in gone and not video_of(r["id"])]
    refusals, k = 0, 0
    (SRC / "DOWNLOADING").write_text("running", encoding="utf-8")
    while k < len(todo) and have() < MAX_VIDEOS and size() < MAX_BYTES:
        chunk = todo[k:k + min(batch, MAX_VIDEOS - have())]
        urls = SRC / "urls.txt"
        urls.write_text("\n".join(f"https://youtu.be/{i}" for i in chunk), encoding="utf-8")
        r = ytdlp("-a", str(urls), "--ignore-errors", "-f", "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/b",
                  "--sleep-requests", "0.5", "--sleep-interval", str(pause[0]), "--max-sleep-interval", str(pause[1]),
                  "-o", str(VIDEOS / "%(id)s.%(ext)s"))
        got = {i for i in chunk if video_of(i)}
        if not got:
            refusals += 1
            if refusals >= 3:
                print("YouTube keeps refusing: stopped; run download again later", flush=True)
                print(r.stderr[-800:])
                break
            print(f"batch refused ({refusals}/3): backing off {backoff // 60} min", flush=True)
            time.sleep(backoff)
            continue
        refusals = 0
        missed = [i for i in chunk if i not in got]  # failed while its neighbours downloaded: really gone
        if missed:
            with open(gone_file, "a", encoding="utf-8") as f:
                f.write("\n".join(missed) + "\n")
        k += len(chunk)
        print(f"{have()} videos, {size() / 1e9:.2f} GB", flush=True)
    for r in rows:
        v = video_of(r["id"])
        r["file"] = str(v) if v else ""
    write_manifest(rows)
    (SRC / "DOWNLOADING").unlink(missing_ok=True)
    print(f"download finished: {have()} videos, {size() / 1e9:.2f} GB in {VIDEOS}", flush=True)



FIELDS_ALL = FIELDS + ["tier", "priority"]
TIERS = {0: "gap-list word", 1: "Islamic word", 2: "word new to the ISL lexicon", 3: "word the lexicon already has"}


def all_glosses(sources=("cislr", "wslp")):
    """Every gloss (single words and phrases) of the given ISL sources, normalised."""
    from isharati.text.english import normalize_en
    out = set()
    for f in sources:
        p = DATA / "lexicon_isl" / f / "entries.jsonl"
        if p.exists():
            out |= {normalize_en(e["gloss"]) for e in load(p) if not e.get("is_letter")}
    return out


def catalog():
    """Every dictionary video of the channel listing and the dictionary playlists (a title of 1-3 English words,
    alternatives after commas, at most MAX_SECONDS long) -> manifest_all.csv in download / extraction priority:
    tier 0 the gap-list videos (manifest.csv), 1 Islamic words, 2 words the ISL lexicon lacks, 3 the rest (loaded
    after CISLR and WSLP, so they never displace a sign); within a tier, videos up to 15 s first (dictionary entries;
    longer ones are explanations, mostly rejected), then each word's first-choice video, then the shortest."""
    from isharati.text.english import candidates, normalize_en
    have = all_glosses()
    gap_rows = {r["id"]: r for r in manifest(MANIFEST)}
    lines = {}
    for f in [SRC / "channel_videos.tsv"] + sorted(SRC.glob("pl_*.tsv")):
        for line in f.read_text(encoding="utf-8").splitlines():
            lines.setdefault(line.split("\t")[0], line)
    rows = []
    for line in lines.values():
        vid, dur, title = (line.split("\t") + ["", ""])[:3]
        try:
            dur = float(dur)
        except ValueError:
            continue
        names = title_gloss(title)
        if not names or dur > MAX_SECONDS:
            continue
        names = [n for n in (normalize_en(x) for x in names) if n and len(n.split()) <= 3]
        if not names:
            continue
        if vid in gap_rows:
            r = dict(gap_rows[vid], tier=0)
        else:
            words = {c for n in names for w in n.split() for c in candidates(w)}
            tier = 1 if words & RELIGIOUS else 2 if not any(n in have for n in names) else 3
            r = {"id": vid, "url": f"https://youtu.be/{vid}", "title": title, "gloss": names[0].upper(),
                 "aliases": "|".join(n.upper() for n in names[1:]), "gap_word": "", "gap_rank": "", "gap_count": "",
                 "seconds": dur, "file": "", "licence": LICENCE, "tier": tier}
        variant = re.search(r"\(sign (\d)\)", title, re.I)
        r["_key"] = (int(variant.group(1)) > 1 if variant else False,
                     bool(re.search(r"\((?!sign \d)[^)]*\)", title, re.I)), float(r["seconds"]))
        rows.append(r)
    rows.sort(key=lambda r: (r["gloss"], r["_key"]))
    per = Counter()
    for r in rows:
        per[r["gloss"]] += 1
        r["choice"] = per[r["gloss"]]
    rows.sort(key=lambda r: (int(r["tier"]), float(r["seconds"]) > 15, int(r["choice"]),
                             int(r["gap_rank"] or 0), float(r["seconds"])))
    for k, r in enumerate(rows):
        r.pop("_key")
        r["priority"] = k
    tmp = MANIFEST_ALL.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_ALL)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, MANIFEST_ALL)
    c = Counter(int(r["tier"]) for r in rows)
    print(f"{len(rows)} dictionary videos, {len(per)} glosses -> {MANIFEST_ALL}")
    for t, n in sorted(c.items()):
        print(f"  tier {t} ({TIERS[t]}): {n} videos")


def videos_bytes():
    return sum(p.stat().st_size for p in VIDEOS.glob("*") if p.suffix in (".mp4", ".webm", ".mkv"))


def download_all(k, n, batch=10, pause=(1, 3), backoff=1800):
    """Download worker k of n over manifest_all.csv in priority order (rows i % n == k), until BUDGET is used;
    DOWNLOADING_<k> exists while it runs, so the extraction workers keep waiting for more."""
    VIDEOS.mkdir(parents=True, exist_ok=True)
    flag = SRC / f"DOWNLOADING_{k}"
    flag.write_text("running", encoding="utf-8")
    gone = set()
    for g in VIDEOS.glob("unavailable*.txt"):
        if g.name != "unavailable_first_pass.txt":
            gone |= set(g.read_text(encoding="utf-8").split())
    gone_file = VIDEOS / f"unavailable_{k}.txt"
    jar = SRC / f"cookies_{k}.txt"
    if not jar.exists():
        jar.write_bytes(COOKIES.read_bytes())
    todo = [r["id"] for i, r in enumerate(manifest(MANIFEST_ALL)) if i % n == k]
    todo = [i for i in todo if i not in gone and not video_of(i)]
    refusals, j = 0, 0
    try:
        while j < len(todo) and videos_bytes() < BUDGET:
            chunk = todo[j:j + batch]
            urls = SRC / f"urls_{k}.txt"
            urls.write_text("\n".join(f"https://youtu.be/{i}" for i in chunk), encoding="utf-8")
            r = ytdlp("-a", str(urls), "--ignore-errors", "-f", "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/b",
                      "--sleep-requests", "0.5", "--sleep-interval", str(pause[0]), "--max-sleep-interval",
                      str(pause[1]), "-o", str(VIDEOS / "%(id)s.%(ext)s"), cookies=jar)
            got = {i for i in chunk if video_of(i)}
            if not got:
                print(r.stderr[-400:], flush=True)
                refusals += 1
                if refusals >= 3:
                    print("YouTube keeps refusing: stopped; run again later", r.stderr[-500:], flush=True)
                    break
                print(f"batch refused ({refusals}/3): backing off {backoff // 60} min", flush=True)
                time.sleep(backoff)
                continue
            refusals = 0
            missed = [i for i in chunk if i not in got]
            if missed:
                with open(gone_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(missed) + "\n")
            j += len(chunk)
            print(f"worker {k}: {j}/{len(todo)}; F: videos {videos_bytes() / 1e9:.2f} GB", flush=True)
    finally:
        flag.unlink(missing_ok=True)
    print(f"download worker {k} finished: {videos_bytes() / 1e9:.2f} GB", flush=True)


MIN_S, MAX_S, MIN_HAND = 0.4, 4.0, 0.7   # accepted clip length (s) and dominant-hand detection share
HOLD_KEEP = 8                             # frames of a final hold kept (the sign's end position, ~0.3 s)


def signing_runs(raw):
    sys.path.insert(0, str(ROOT / "scripts" / "lexicon" / "asl"))
    from islamic_terms import signing_runs as runs
    return runs(raw)


def hand_features(raw):
    """Per frame, shoulder-width units relative to the neck: both wrists and both hand centroids (8 numbers);
    NaN where unseen."""
    from isharati.pose.keypoints import L_SH, L_WR, LH, NECK, R_SH, R_WR, RH
    sw = np.nanmedian(np.abs(raw[:, R_SH, 0] - raw[:, L_SH, 0]))
    neck = (raw[:, R_SH, :2] + raw[:, L_SH, :2]) / 2
    with np.errstate(all="ignore"):
        f = [raw[:, R_WR, :2], raw[:, L_WR, :2], np.nanmean(raw[:, RH, :2], 1), np.nanmean(raw[:, LH, :2], 1)]
    f = [np.where(np.isnan(x), np.nan, x - neck) / max(sw, 1e-6) for x in f]
    return np.concatenate(f, axis=1)


def wrist_speed(raw):
    """Per frame, the faster wrist's displacement (shoulder widths per frame), gaps filled, 3-frame median."""
    f = hand_features(raw)[:, :4]
    for j in range(f.shape[1]):
        v, ok = f[:, j], ~np.isnan(f[:, j])
        f[:, j] = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok]) if ok.any() else 0.0
    d = np.r_[0, np.abs(np.diff(f, axis=0)).max(axis=1)]
    return np.array([np.median(d[max(0, i - 1):i + 2]) for i in range(len(d))])


def trim_hold(raw, a, b, fps=25):
    """The sign's final position held (hands still for more than HOLD_KEEP frames) and followed by nothing but the
    lowering of the hands (at most 0.8 s): cut to HOLD_KEEP frames of the hold."""
    still = wrist_speed(raw[a:b]) < 0.012
    L, holds, i = len(still), [], 0
    while i < L:
        if still[i]:
            j = i
            while j < L and still[j]:
                j += 1
            if j - i > HOLD_KEEP:
                holds.append((i, j))
            i = j
        else:
            i += 1
    if holds:
        i, j = holds[-1]
        if L - j <= 0.8 * fps and i >= 0.3 * fps:
            b = a + i + HOLD_KEEP
    return a, b


def first_repetition(raw, a, b, fps=25):
    """When a run performs the sign twice without lowering the hands, the end of the first performance: the smallest
    lag p (>= 0.4 s) at which the opening of the run (its first p frames, while they move) comes back closely."""
    f = hand_features(raw[a:b])
    L = len(f)
    if L < 2 * int(0.4 * fps):
        return b
    for p in range(int(0.4 * fps), L // 2 + 1):
        w = min(p, L - p)
        x, y = f[:w], f[p:p + w]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 0.6 * x.size:
            continue
        travel = np.nansum(np.abs(np.diff(f[:p], axis=0)))  # the opening must move (not a hold that matches itself)
        if travel < 1.0:
            continue
        if np.mean(np.abs(x[ok] - y[ok])) < 0.08:
            return a + p
    return b


def fingerspelling_start(raw, a, b, fps=25):
    """Start (absolute frame) of a fingerspelled stretch inside [a, b), or None: at least 1 s in which the wrists stay
    near one place (median speed under 0.02 shoulder widths per frame) while the handshape keeps changing (the mean
    change of the fingertips relative to the wrist over 0.04 per frame), the signature of spelling a word letter by
    letter after (or before) its sign."""
    from isharati.pose.keypoints import L_WR, LH, R_WR, RH
    seg = raw[a:b]
    speed = wrist_speed(seg)
    shape = np.zeros(len(seg))
    for hand, wr in ((RH, R_WR), (LH, L_WR)):
        rel = seg[:, hand, :2] - seg[:, wr:wr + 1, :2]
        sw = np.nanmedian(np.abs(raw[:, 2, 0] - raw[:, 5, 0])) or 1.0
        d = np.nanmean(np.linalg.norm(np.diff(rel, axis=0), axis=-1), axis=1) / sw
        shape = np.maximum(shape, np.r_[0, np.nan_to_num(d)])
    spell = (speed < 0.02) & (shape > 0.04)
    win = int(fps)
    for i in range(0, max(0, len(seg) - win)):
        if spell[i:i + win].mean() > 0.6:
            return a + i
    return None


def clip_span(raw, fps=25):
    """The kept span of a dictionary video, (a, b, reason): ONE performance of the sign.
    1. the first signing run (islamic_terms.signing_runs: a hand above waist level, gaps of up to 6 frames bridged)
       of at least 10 frames. When more signing follows, the run must stand apart (1.5 s of rest or a title card
       before the next run: the financial terms' «Explanation of ...» card); otherwise which run is the sign is
       ambiguous (several synonyms signed in a row, a sign followed by «this sign means ...») and the video is
       rejected;
    2. a fingerspelled stretch inside the run is cut off (rejected when the run starts with it);
    3. a repetition inside the run (hands not lowered between the two) is cut to the first performance;
    4. a final hold followed only by lowering the hands is cut to HOLD_KEEP frames; then PAD frames either side.
    reason is None when the span passes the checks (MIN_S..MAX_S, dominant hand seen in MIN_HAND of the frames,
    exactly one signing run inside)."""
    from isharati.pose.keypoints import LH, R_SH, RH
    if np.isnan(raw[:, R_SH, 0]).all():
        return None, None, "no shoulders"
    runs = [(s, e) for s, e in signing_runs(raw) if e - s >= 10]
    if not runs:
        return None, None, "no signing run"
    a, b = runs[0]
    if len(runs) > 1 and runs[1][0] - b < 1.5 * fps:
        return a, b, f"first run not separated from further signing ({(runs[1][0] - b) / fps:.2f} s rest)"
    fs = fingerspelling_start(raw, a, b)
    if fs is not None:
        if fs - a < int(MIN_S * fps):
            return a, b, "starts with fingerspelling"
        b = fs
    b = first_repetition(raw, a, b)
    a, b = trim_hold(raw, a, b)
    a, b = max(0, a - PAD), min(len(raw), b + PAD)
    seg = raw[a:b]
    lh = ~np.isnan(seg[:, LH, 0]).all(axis=1)
    rh = ~np.isnan(seg[:, RH, 0]).all(axis=1)
    dom = max(lh.mean(), rh.mean())
    n_runs = len([r for r in signing_runs(seg) if r[1] - r[0] >= 6])
    dur = (b - a) / fps
    if dur < MIN_S:
        return a, b, f"too short ({dur:.2f} s)"
    if dur > MAX_S:
        return a, b, f"too long ({dur:.2f} s)"
    if dom < MIN_HAND:
        return a, b, f"dominant hand seen in {dom:.0%} of frames"
    if n_runs != 1:
        return a, b, f"{n_runs} signing runs in the span"
    return a, b, None


def extract_one(r):
    """Track (once, kept on F:), clip, face. Returns a status string."""
    sys.path.insert(0, str(ROOT / "scripts" / "face"))
    from attach import fifty, save_face
    from isharati.pose.keypoints import interpolate_missing, normalize
    vid = r["id"]
    npy, bad = OUT / "signs" / f"{vid}.npy", OUT / "signs" / f"{vid}.failed"
    if npy.exists() or bad.exists():
        return "skip"
    video = video_of(vid)
    if video is None:
        return "no video"
    tr_path = TRACKS / f"{vid}.npz"
    until = UNTIL if float(r["seconds"] or 0) > UNTIL + 3 else None
    if not tr_path.exists():
        from track import track
        data, meta = track(video, until=until)
        tmp = tr_path.with_name(tr_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **data)
        os.replace(tmp, tr_path)
    z = np.load(tr_path)
    meta = json.loads(str(z["meta"]))
    tr = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    raw = fifty(tr)
    a, b, reason = clip_span(raw)
    if reason is None and until is not None and b >= len(raw) - PAD:
        reason = "signing runs past the tracked first seconds"
    if reason is None:
        filled, missing = interpolate_missing(raw[a:b])
        if missing > MAX_MISSING:
            reason = f"missing hand {missing:.2f}"
    if reason is not None:
        bad.write_text(json.dumps({"reason": reason, "start": None if a is None else round(float(tr["time"][a]), 3),
                                   "end": None if b is None else round(float(tr["time"][b - 1]), 3)}),
                       encoding="utf-8")
        return "rejected"
    clip = normalize(filled).astype(np.float32)
    np.save(npy, clip)
    hands = float(np.mean(~(np.isnan(raw[a:b, 8:29, 0]).all(1) & np.isnan(raw[a:b, 29:50, 0]).all(1))))
    row = {"sign_id": f"islrtc_{vid}", "gloss": r["gloss"], "dataset": "islrtc_isl", "video": video.name,
           "start": round(float(tr["time"][a]), 3), "end": round(float(tr["time"][b - 1]), 3), "frames": b - a,
           "match": 0.0, "found": True}
    save_face(FACE, row["sign_id"], tr, raw, meta, a, b - a, 0.0, row, clip)
    npy.with_suffix(".json").write_text(json.dumps({"missing": round(missing, 3), "hands": round(hands, 3),
                                                    "start": row["start"], "end": row["end"], "frames": b - a,
                                                    "face_detected": row["face_detected"]}), encoding="utf-8")
    return "ok"


def extract(k, n):
    """Worker k of n over the downloaded videos; keeps looping while the download runs (F:/isl_new/islrtc/DOWNLOADING)."""
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    TRACKS.mkdir(parents=True, exist_ok=True)
    FACE.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    flag = SRC / f"EXTRACTING_{k}"
    flag.write_text("running", encoding="utf-8")
    while True:
        downloading = (SRC / "DOWNLOADING").exists() or any(SRC.glob("DOWNLOADING_*"))
        rows = [r for i, r in enumerate(manifest()) if i % n == k]
        todo = [r for r in rows if video_of(r["id"]) and not (OUT / "signs" / f"{r['id']}.npy").exists()
                and not (OUT / "signs" / f"{r['id']}.failed").exists()]
        for j, r in enumerate(todo, 1):
            try:
                s = extract_one(r)
            except Exception as e:  # a broken file: note it, go on
                s = "error"
                print(r["id"], "error:", e, flush=True)
                (OUT / "signs" / f"{r['id']}.failed").write_text(json.dumps({"reason": f"error: {e}"}), encoding="utf-8")
            stats[s] += 1
            if sum(stats.values()) % 25 == 0:
                print(f"worker {k}/{n}: {dict(stats)}", flush=True)
        if not downloading:
            break
        time.sleep(60)
    flag.unlink(missing_ok=True)
    print(f"worker {k}/{n} finished: {dict(stats)}", flush=True)


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def entries():
    """entries.jsonl from every accepted clip, in manifest priority order, written atomically (temp file + replace),
    so a loader never sees half a file. Per gloss the first clip is the sign; a later clip of the same gloss is a
    variant row ("variant_of": the sign's id), which the ISL lexicon skips. Returns (signs, variants, rows)."""
    out, main = [], {}
    rows = manifest()
    for r in rows:
        if not (OUT / "signs" / f"{r['id']}.npy").exists():
            continue
        sid = f"islrtc_{r['id']}"
        base = {"sign_id": sid, "dataset": "islrtc_isl", "keypoints_path": f"islrtc/signs/{r['id']}.npy",
                "review_status": "pending", "is_letter": False,
                "is_religious": bool({r["gap_word"], *r["gloss"].lower().split()} & RELIGIOUS)}
        # the title word, the gap word it was picked for, then the title's other alternatives
        names = list(dict.fromkeys([r["gloss"]] + ([r["gap_word"].upper()] if r.get("gap_word") else [])
                                   + [a for a in r.get("aliases", "").split("|") if a]))
        if r["gloss"] in main:
            out.append({"gloss": r["gloss"], **base, "variant_of": main[r["gloss"]]})
        for g in names:
            if g not in main:
                main[g] = sid
                out.append({"gloss": g, **base})
    write_atomic(OUT / "entries.jsonl", "\n".join(json.dumps(e, ensure_ascii=False) for e in out) + "\n")
    signs = len({e["sign_id"] for e in out if "variant_of" not in e})
    variants = sum("variant_of" in e for e in out)
    face_manifest(rows)
    return signs, variants, len(out)


def face_manifest(rows):
    """The new signs' rows in data/face/ur/manifest.jsonl (attach.py's format; other datasets' rows are kept)."""
    path = FACE / "manifest.jsonl"
    keep = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path.exists() else []
    keep = [m for m in keep if m.get("dataset") != "islrtc_isl"]
    for r in rows:
        info = OUT / "signs" / f"{r['id']}.json"
        if info.exists() and (FACE / f"islrtc_{r['id']}.npz").exists():
            i = json.loads(info.read_text(encoding="utf-8"))
            keep.append({"sign_id": f"islrtc_{r['id']}", "gloss": r["gloss"], "dataset": "islrtc_isl",
                         "video": f"{r['id']}.mp4", "start": i["start"], "end": i["end"], "frames": i["frames"],
                         "match": 0.0, "found": True, "face_detected": i["face_detected"]})
    write_atomic(path, "\n".join(json.dumps(m, ensure_ascii=False) for m in keep) + "\n")


def isl_on_en():
    """ISL content-word token coverage of the English corpus, corpus_eda.py's isl_on_en method (all word sources)."""
    c_all, _, _ = corpus_counts()
    return coverage(c_all, isl_glosses(True))


def collect(every=60, report=200):
    """While downloads or extraction run: rewrite entries.jsonl every `every` s; print the sign count and the ISL
    coverage every `report` new signs."""
    last = -report
    while True:
        running = any(SRC.glob("DOWNLOADING*")) or any(SRC.glob("EXTRACTING_*"))
        signs, variants, rows = entries()
        if signs - last >= report or not running:
            cov = isl_on_en()
            print(f"{time.strftime('%H:%M')} {signs} signs, {variants} variants, {rows} rows; ISL coverage "
                  f"{cov['token_coverage']:.2%} of content tokens", flush=True)
            last = signs
        if not running:
            break
        time.sleep(every)


def contact_sheet(video, times, labels, out, title="", width=240):
    """One row of frames of a video at the given times (seconds), labelled."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for t, lab in zip(times, labels):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, img = cap.read()
        if not ok:
            img = np.zeros((360, 640, 3), np.uint8)
        h, w = img.shape[:2]
        img = cv2.resize(img, (width, int(h * width / w)))
        cv2.rectangle(img, (0, 0), (width, 22), (0, 0, 0), -1)
        cv2.putText(img, f"{lab} {t:.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cap.release()
    sheet = np.concatenate(tiles, axis=1)
    bar = np.zeros((26, sheet.shape[1], 3), np.uint8)
    cv2.putText(bar, title[:120], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.concatenate([bar, sheet], axis=0))


def overview(vid, out, step=0.4):
    """Inspection sheet of a whole video: a frame every `step` s with the hand timeline (B/L/R/.) and the signing
    runs from its track, to work out the structure of a source's videos."""
    import cv2
    z = np.load(TRACKS / f"{vid}.npz")
    sys.path.insert(0, str(ROOT / "scripts" / "face"))
    from attach import fifty
    tr = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    raw = fifty(tr)
    runs = signing_runs(raw)
    a, b, reason = clip_span(raw)
    video = video_of(vid)
    cap = cv2.VideoCapture(str(video))
    T, tiles = len(raw), []
    for i in range(0, T, int(step * 25)):
        t = float(tr["time"][i])
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (200, int(h * 200 / w)))
        lh, rh = not np.isnan(raw[i, 8, 0]), not np.isnan(raw[i, 29, 0])
        inrun = any(s <= i < e for s, e in runs)
        kept = a is not None and a <= i < b
        col = (0, 255, 0) if kept else (0, 200, 255) if inrun else (128, 128, 128)
        cv2.rectangle(img, (0, 0), (199, img.shape[0] - 1), col, 4 if kept else 2)
        cv2.putText(img, f"{t:.1f} {'B' if lh and rh else 'L' if lh else 'R' if rh else '.'}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cap.release()
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[k:k + cols], axis=1) for k in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.concatenate(rows, axis=0))
    return {"runs_s": [(round(s / 25, 2), round(e / 25, 2)) for s, e in runs],
            "kept_s": None if a is None else (round(a / 25, 2), round(b / 25, 2)), "reason": reason}


def qa(n=20, render=5, seed=0):
    """Contact sheets (1 frame before, start, middle, end, 1 frame after the kept span) of n random accepted clips
    and skeleton videos of `render` of them, in F:/isl_new/islrtc/qa/."""
    import random
    from isharati.pose.render import SkeletonRenderer
    out = SRC / "qa"
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in manifest()}
    ok = sorted(p.stem for p in (OUT / "signs").glob("*.npy"))
    pick = random.Random(seed).sample(ok, min(n, len(ok)))
    for k, vid in enumerate(pick):
        info = json.loads((OUT / "signs" / f"{vid}.json").read_text(encoding="utf-8"))
        s, e = info["start"], info["end"]
        contact_sheet(video_of(vid), [s - 0.24, s, (s + e) / 2, e, e + 0.24],
                      ["before", "start", "middle", "end", "after"], out / f"accepted_{k:02d}_{vid}.jpg",
                      f"{rows[vid]['gloss']} | title: {rows[vid]['title']} | kept {s:.2f}-{e:.2f}s "
                      f"hands {info['hands']:.0%}")
        if k < render:
            SkeletonRenderer().render(np.load(OUT / "signs" / f"{vid}.npy"), out / f"skeleton_{k:02d}_{vid}.mp4")
    print(f"{len(pick)} contact sheets, {min(render, len(pick))} skeleton videos -> {out}")
    return pick


def report_coverage():
    c_all, c_ur, _ = corpus_counts()
    before, after = isl_glosses(False), isl_glosses(True)
    res = {"glosses_before": len(before), "glosses_after": len(after), "new_glosses": len(after - before)}
    for name, c in (("english_corpus_all", c_all), ("urdu_mode_passages", c_ur)):
        res[name] = {"before": coverage(c, before), "after": coverage(c, after)}
    (SRC / "coverage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gaps":
        gaps()
    elif mode == "list":
        list_videos()
    elif mode == "download":
        download()
    elif mode == "extract":
        extract(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "entries":
        print("%d signs, %d variants, %d rows" % entries())
    elif mode == "catalog":
        catalog()
    elif mode == "download_all":
        download_all(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "collect":
        collect()
    elif mode == "coverage":
        report_coverage()
    elif mode == "overview":
        for v in sys.argv[2:]:
            print(v, overview(v, SRC / "qa" / f"overview_{v}.jpg"))
    elif mode == "qa":
        qa()
