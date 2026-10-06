"""Turkish Sign Language (TİD) word signs from two YouTube dictionaries -> a second TİD lexicon source: first the
Turkish words of the Qur'an/hadith corpus that the TİD dictionary cannot match, then every other word the two channels
sign (the largest lexicon possible, so the system generalises beyond the Qur'an/hadith core).

  tid_sozluk  «Türk İşaret Dili Sözlüğü» (youtube.com/channel/UCq2tlBndwWjJb2Zlf9QoOzQ): ~5,000 clips of 3-15 s, one
              word each, titled with the word in capitals («KILIÇ BALIĞI»). The existing TİD dictionary
              (data/lexicon_tr/dictionaries, scripts/lexicon/youtube_dictionaries.py) downloaded only ~1,380 of them;
              the rest are taken here.
  tidislam    «TiDiSLaM» (youtube.com/channel/UC7zHDrJRMxSd65cE3vxfksg): ~600 Islamic terms in TİD («MÜZDELİFE»,
              «ABDEST»), 10-40 s each: the sign, then its meaning explained in TİD. Never downloaded before.

Licence: neither channel states an open licence (standard YouTube terms; both publish the signs freely for learning
TİD). Like the existing TİD dictionary: research and the demo, ask the channel owners before any redistribution.

The gap list is the content words of the Turkish corpus (data/tr/corpus.jsonl) that no TİD gloss covers, measured as
scripts/eval/corpus_eda.py measures Turkish (tr_norm words, TR_STOP removed, a gloss covers a word when they share a
tr_forms form: the word or its first five letters), ranked by frequency. A video is taken for a gap word when its title
word shares a form with it AND the word is a plausible inflection of the title word (the title word, its verb stem
before -mak/-mek, or its stem with a final p/ç/t/k softened, begins the word): «namaz» for namazı, «göndermek» for
gönderdi, «kitap» for kitabı; not «kimsesiz» (orphan) for kimse, nor «kaldırım» (pavement) for kaldı. The manifests
order the videos in tiers (TIERS): the gap words' first choices, their fallbacks, TiDiSLaM's other terms, the
dictionary's other words. A second recording of a word already signed becomes a variant ("variant_of" in its entry).

Streaming: 2 download workers and 3 extract workers run side by side; each extracted video rewrites entries.jsonl at
once (atomically), so tid_lexicon() sees the lexicon grow during the run.

Each video goes once through the MediaPipe Holistic task (scripts/face/track.py: body, hands, 478 face points, 52
blendshapes on the 25 fps grid); the track becomes the lexicon's 50-point clip (normalize(interpolate_missing(raw)) of
ONE performance of the sign, see clip_span) and the face of exactly those frames is saved as
data/face/tr/tidyt_<id>.npz (scripts/face/attach.py's save_face format).

  gaps      missing content words -> F:/tid_new/missing_tid.csv                        (plain python)
  list      channel listings x gap list -> F:/tid_new/<source>/manifest.csv             (plain python, yt-dlp)
  download  worker k of n: the manifests' videos in order, politely                    (plain python, yt-dlp)
  extract   worker k of n: track + clip + face per downloaded video                    (the MediaPipe venv)
  overview  <source> <id>...: whole-video inspection sheets and hand timelines          (the MediaPipe venv)
  qa        contact sheets of 20 random accepted clips, 5 skeleton videos, reject list (plain python + cv2)
  entries   collect data/lexicon_tr/tid_youtube/entries.jsonl                          (plain python)
  coverage  TİD content-word coverage before / after -> F:/tid_new/coverage.json       (plain python)

  PYTHONPATH=src py scripts/lexicon/tid/tid_youtube.py gaps|list|qa|entries|coverage
  PYTHONPATH=src py scripts/lexicon/tid/tid_youtube.py download <k> <n>                  # k = 0, 1 of n = 2
  "D:/islam/gathring data/.venv/Scripts/python" scripts/lexicon/tid/tid_youtube.py extract <k> <n>   # n = 3
Output: data/lexicon_tr/tid_youtube/{signs/<id>.npy, entries.jsonl}, data/face/tr/tidyt_<id>.npz; videos, tracks,
manifests, QA and the reject list on F:/tid_new/. tid_lexicon() loads the entries with add_words (a gloss the lexicon
already signs keeps its sign); scripts/hub/publish.py packs them as tr/lexicon_tid_youtube.jsonl and
tr/tid_youtube_poses.npz.
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

WORK = Path(os.environ.get("TID_WORK", r"F:\tid_new"))
GAPS = WORK / "missing_tid.csv"
LEX = DATA / "lexicon_tr"
OUT = LEX / "tid_youtube"
FACE = DATA / "face" / "tr"
PREFIX = "tidyt"
OLD_ENTRIES = [LEX / "dictionaries" / "entries_tid_dictionary.jsonl", LEX / "names" / "entries.jsonl"]
OLD_SIGNS = LEX / "dictionaries" / "signs"
OLD_VIDEOS = Path(r"D:\islam\gathring data\turkish_dictionaries")  # the first TİD download (read only)
SOURCES = {
    "tid_sozluk": {"channel": "UCq2tlBndwWjJb2Zlf9QoOzQ", "dataset": "tid_sozluk_yt", "max_seconds": 15,
                   "licence": "Türk İşaret Dili Sözlüğü YouTube channel (UCq2tlBndwWjJb2Zlf9QoOzQ); no open licence "
                              "stated, standard YouTube terms; research use"},
    "tidislam": {"channel": "UC7zHDrJRMxSd65cE3vxfksg", "dataset": "tidislam", "max_seconds": 60,
                 "licence": "TiDiSLaM YouTube channel (UC7zHDrJRMxSd65cE3vxfksg); no open licence stated, standard "
                            "YouTube terms; research use"},
}
YTDLP = Path(r"D:\islam\yt-dlp")
# cookies: TID_COOKIES=<file>; "none" (the default) downloads anonymously: the shared cookie sessions get throttled
# («Video unavailable» for every video) when several agents download with them at once
COOKIES = os.environ.get("TID_COOKIES", "none")
MAX_VIDEOS, MAX_BYTES = 6000, int(4 * 1024 ** 3)  # videos on F: (tracks add ~1.5 GB)
TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
SOFT = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}  # final consonant softening before a vowel: kitap -> kitabı
# corpus words signed by a video titled otherwise (lower-case titles): names that tr_norm splits at the apostrophe
# («Kur'an» -> kur an), and the hadith formulas, each by the sign of what it means: «aleyhi (ve sellem)» by
# ALEYHİSSELAM (peace be upon him), «radıyallahu (anh)» by ALLAH RAZI OLSUN (may Allah be pleased with him),
# «Rasûlullah» by ALLAH'IN ELÇİSİ (the Messenger of Allah), «nebi» (prophet) by PEYGAMBER
TITLE_ALIASES = {"kur": ["kur'an", "kuran", "kur'an-ı kerim", "kuran-ı kerim"],
                 "aleyhi": ["aleyhisselam"], "radıyallahu": ["allah razı olsun"],
                 "rasûlullah": ["allah'ın elçisi"], "nebi": ["peygamber"], "nebî": ["peygamber"]}


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def src_dir(source):
    return WORK / source


def corpus_counts():
    """Counts of the corpus words (all / Qur'an / hadith), and per word how often it directly follows an apostrophe
    (a case ending split from a name: Allah'ın -> allah ın) and how often it is written with a capital."""
    from isharati.retrieval.turkish_urdu import tr_norm
    c = {"all": Counter(), "quran": Counter(), "hadith": Counter()}
    after_apos, cap = Counter(), Counter()
    for r in load(DATA / "tr" / "corpus.jsonl"):
        ws = [w for w in tr_norm(r["text"]).split() if w.isalpha()]
        c["all"].update(ws)
        c[r["kind"]].update(ws)
        for m in re.finditer(r"['’‘]([^\W\d_]+)", r["text"]):
            after_apos.update(w for w in tr_norm(m.group(1)).split()[:1])
        for m in re.finditer(r"(?<![\w'’])([A-ZÇĞİÖŞÜÂÎÛ][^\W\d_]*)", r["text"]):
            cap.update(tr_norm(m.group(1)).split()[:1])
    return c, after_apos, cap


def gloss_forms(files):
    """form -> gloss of every single-word gloss in the entries files, as corpus_eda.py builds it."""
    from isharati.retrieval.turkish_urdu import tr_forms, tr_norm
    forms = {}
    for p in files:
        for e in (load(p) if Path(p).exists() else []):
            if e.get("is_letter"):
                continue
            g = tr_norm(e["gloss"])
            if g and " " not in g:
                for f in tr_forms(g):
                    forms.setdefault(f, g)
    return forms


def coverage(files):
    """corpus_eda.py's Turkish measure: share of content-word occurrences (and types) some gloss covers."""
    from isharati.retrieval.turkish_urdu import TR_STOP, tr_forms
    c, _, _ = corpus_counts()
    forms = gloss_forms(files)
    content = {w: n for w, n in c["all"].items() if w not in TR_STOP}
    cov = [w for w in content if any(f in forms for f in tr_forms(w))]
    tok = sum(content.values())
    return {"glosses": len(set(forms.values())), "content_tokens": tok,
            "token_coverage": round(sum(content[w] for w in cov) / tok, 4),
            "type_coverage": round(len(cov) / len(content), 4)}


def gaps():
    from isharati.retrieval.turkish_urdu import TR_STOP, tr_forms
    c, after_apos, cap = corpus_counts()
    forms = gloss_forms(OLD_ENTRIES)
    content = Counter({w: n for w, n in c["all"].items() if w not in TR_STOP})
    total = sum(content.values())
    missing = [(w, n) for w, n in content.most_common() if not any(f in forms for f in tr_forms(w))]
    miss_tok = sum(n for _, n in missing)
    WORK.mkdir(parents=True, exist_ok=True)
    cum = 0
    with open(GAPS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "count", "quran", "hadith", "share", "cum_share_of_missing", "cum_share_of_content",
                    "fragment", "likely_name", "forms"])
        for k, (word, n) in enumerate(missing, 1):
            cum += n
            # a case ending cut from a name (ın, a, dan) or a single letter: no sign of its own
            fragment = int(len(word) < 2 or after_apos[word] >= 0.5 * n)
            name = int(cap[word] >= 0.9 * n and n >= 2)
            w.writerow([k, word, n, c["quran"][word], c["hadith"][word], round(n / total, 5), round(cum / miss_tok, 4),
                        round(cum / total, 4), fragment, name, "|".join(sorted(tr_forms(word)))])
    top500 = sum(n for _, n in missing[:500])
    print(f"content occurrences {total}, covered {1 - miss_tok / total:.2%}; {len(missing)} missing words "
          f"({miss_tok} occurrences); the top 500 = {top500 / total:.1%} of all content occurrences, "
          f"{top500 / miss_tok:.1%} of the missing ones")
    print(", ".join(f"{w} ({n})" for w, n in missing[:50]))
    print("->", GAPS)


def ytdlp(*args, **kw):
    cookies = [] if COOKIES == "none" else ["--cookies", COOKIES]
    return subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", *cookies,
                           "--js-runtimes", "node", "--remote-components", "ejs:github", *args],
                          cwd=YTDLP, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def title_names(title):
    """Turkish gloss and aliases of a title («KILIÇ BALIĞI», «MÜ'MİN», «ABDEST / ABDEST ALMAK»), Turkish casing,
    as youtube_dictionaries.glosses_tr reads them."""
    t = re.sub(r"\([^)]*\)|\[[^]]*\]|#\S+|Türk İşaret Dili Sözlüğü|işaret dili|İşaret Dili", " ", title, flags=re.I)
    names = [re.sub(r"\s+", " ", p).strip(" -–:.,/\"").translate(TR_LOWER).lower() for p in re.split(r"[/,;]| - ", t)]
    return list(dict.fromkeys(n for n in names if re.search(r"[a-zçğıöşüâîû]", n) and len(n) <= 40))


def stems(g):
    """Beginnings that an inflected form of the title word g starts with: g itself, the verb stem of an infinitive
    (göndermek -> gönder), g with its final p/ç/t/k softened (kitap -> kitab, ihtiyaç -> ihtiyac)."""
    out = {g}
    if re.search(r"m[ae]k$", g) and len(g) > 5:
        out.add(g[:-3])
    if g and g[-1] in SOFT:
        out.add(g[:-1] + SOFT[g[-1]])
    return out


# inflectional (and a few close derivational) suffix morphs, every vowel-harmony variant: a word is an inflection of
# a title word when what follows the title word's stem splits wholly into these («kıyam» + «et» does not: kıyamet is
# not a form of kıyam)
_V4, _V2 = "ıiuü", "ae"
_MORPHS = set()
for _m in ("lAr", "I", "yI", "nI", "A", "yA", "nA", "DA", "nDA", "DAn", "nDAn", "In", "nIn", "Im", "m", "Iz", "ImIz",
           "mIz", "InIz", "nIz", "sI", "lArI", "ylA", "lA", "DIr", "ki", "ken", "yken", "DI", "mIş", "yDI", "ymIş",
           "Ir", "Ar", "r", "AcAk", "yAcAk", "AcAğ", "yAcAğ", "Iyor", "yor", "mAlI", "sA", "ysA", "mA", "mAk", "mAsI",
           "DIk", "DIğ", "An", "yAn", "Ip", "yIp", "IncA", "yIncA", "ArAk", "yArAk", "sIn", "sInlAr", "lIm", "yIm",
           "yIz", "k", "Il", "n", "Iş", "DIr", "Abil", "yAbil", "mAz", "AmAz", "yAmAz", "lIk", "lI", "sIz", "cI",
           "CA", "dAş", "sAl", "y", "mIş", "Iniz", "nIzI", "mA", "mAyA", "mAdAn", "mAk", "mAklA", "AlIm", "AsI"):
    _alts = [""]
    for ch in _m:
        opts = _V4 if ch == "I" else _V2 if ch == "A" else "dt" if ch == "D" else "cç" if ch == "C" else ch
        _alts = [a + o for a in _alts for o in opts]
    _MORPHS.update(_alts)
_MAXM = max(map(len, _MORPHS))


def inflection(rest):
    """rest splits wholly into suffix morphs."""
    ok = [True] + [False] * len(rest)
    for i in range(len(rest)):
        if ok[i]:
            for j in range(i + 1, min(len(rest), i + _MAXM) + 1):
                if rest[i:j] in _MORPHS:
                    ok[j] = True
    return ok[-1]


def compatible(g, w):
    """The matcher would use title word g for corpus word w, and w is an inflection of g (one of g's stems followed
    by suffixes only)."""
    from isharati.retrieval.turkish_urdu import tr_forms
    if not tr_forms(g) & tr_forms(w):
        return False
    return any(w.startswith(s) and inflection(w[len(s):]) for s in stems(g))


class OldMatcher:
    """What the current TİD lexicon matches (TIDLexicon.match_token on the dictionary and names), to see which corpus
    words a new gloss would take: words that had no sign, and words whose sign it would replace."""

    def __init__(self, files=OLD_ENTRIES):
        from isharati.retrieval.turkish_urdu import tr_norm
        from isharati.lexicon.turkish import WRONG_SIGN  # signs tid_lexicon() withdraws: not signed any more
        self.keys = {tr_norm(e["gloss"]) for p in files if Path(p).exists() for e in load(p) if not e.get("is_letter")}
        self.keys -= {tr_norm(g) for g in WRONG_SIGN}
        self.prefix = {k[:5] for k in self.keys if len(k) > 5}

    def takes(self, g, w):
        """Would adding gloss g (TIDLexicon.add_words: never over an existing key; its 5-letter prefix only where none
        exists) make the matcher return g's sign for word w?"""
        if g in self.keys or w in self.keys:
            return False
        if w == g:
            return True
        if len(w) <= 5:
            return False
        if w[:5] == g:  # an exact key is looked up before any prefix
            return True
        return len(g) > 5 and w[:5] == g[:5] and w[:5] not in self.keys and w[:5] not in self.prefix


def effect(g, content, old, by5):
    """(occurrences g signs correctly, occurrences it would sign wrongly) over the corpus content words."""
    good = bad = 0
    for w in by5.get(g[:5], ()):
        if old.takes(g, w):
            if w == g or compatible(g, w):
                good += content[w]
            else:
                bad += content[w]
    return good, bad


def channel_listing(source):
    """The channel's videos and shorts (id, seconds, title), cached in F:/tid_new/<source>/channel_videos.tsv."""
    d = src_dir(source)
    d.mkdir(parents=True, exist_ok=True)
    listing = d / "channel_videos.tsv"
    if not listing.exists():
        out = []
        for tab in ("videos", "shorts"):
            r = ytdlp("--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s",
                      f"https://www.youtube.com/channel/{SOURCES[source]['channel']}/{tab}")
            out += r.stdout.splitlines()
            time.sleep(3)
        listing.write_text("\n".join(dict.fromkeys(out)), encoding="utf-8")
    rows = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        vid, dur, title = (line.split("\t") + ["", ""])[:3]
        try:
            rows.append((vid, float(dur), title))
        except ValueError:
            continue
    return rows


def old_ids():
    """Videos the first TİD build made a clip of (the ones it could not cut, .failed, are tried again here from its
    local copy of the video)."""
    return {p.stem for p in OLD_SIGNS.glob("*.npy")}


FIELDS = ["order", "tier", "id", "url", "title", "gloss", "aliases", "gap_word", "gap_rank", "gap_count", "covers",
          "covers_count", "choice", "seconds", "file", "licence"]
TIERS = {0: "gap word, first choice", 1: "gap word, fallback", 2: "TiDiSLaM, every other term",
         3: "dictionary, every other word"}


def list_videos():
    """Both channels against the gap list. Gap words are taken most frequent first; a word that an already chosen
    video covers is skipped. Per word up to two videos: TiDiSLaM first (the Islamic terms matter most here), then
    the dictionary, an exact title word before an inflection, a shorter video first; the second is the fallback
    when the first is rejected."""
    from isharati.retrieval.turkish_urdu import TR_STOP, tr_forms, tr_norm
    gap = []
    for row in csv.DictReader(open(GAPS, encoding="utf-8")):
        if row["fragment"] == "0":
            gap.append((int(row["rank"]), row["word"], int(row["count"])))
    c, _, _ = corpus_counts()
    content = Counter({w: n for w, n in c["all"].items() if w not in TR_STOP})
    by5 = {}
    for w in content:
        by5.setdefault(w[:5], []).append(w)
    old, memo = OldMatcher(), {}
    gap_words = {w for _, w, _ in gap}

    def safe(n):
        """g signs at least twice as many occurrences correctly as wrongly (kıyam would take kıyamet: no)."""
        if n not in memo:
            good, bad = effect(n, content, old, by5)
            memo[n] = good >= 2 * bad
        return memo[n]
    used = old_ids()
    cands = []  # (source, id, seconds, title, gloss names)
    by_form = {}
    for source in SOURCES:
        for vid, dur, title in channel_listing(source):
            if dur > SOURCES[source]["max_seconds"] or vid in used:
                continue
            names = [tr_norm(n) for n in title_names(title)]
            # an alias word is signed only by its alias titles («KUR» is to set up, not the Qur'an)
            names = [n for n in names if n and " " not in n and n not in TITLE_ALIASES]
            t = re.sub(r"\s+", " ", title.translate(TR_LOWER).lower().replace("’", "'")).strip()
            names += [w for w, titles in TITLE_ALIASES.items() if t in titles]
            if not names:
                continue
            k = len(cands)
            cands.append((source, vid, dur, title, names))
            for n in names:
                for f in tr_forms(n):
                    by_form.setdefault(f, set()).add(k)
    chosen, covered_by, rows = {}, {}, []
    for rank, word, count in gap:
        if word in covered_by:
            continue
        ks = {k for f in tr_forms(word) for k in by_form.get(f, ())}
        hits = []
        for k in ks:
            source, vid, dur, title, names = cands[k]
            for j, n in enumerate(names):
                if (n == word or compatible(n, word)) and safe(n):
                    hits.append(((source != "tidislam", n != word, j > 0, dur), k, n))
                    break
        if not hits:
            continue
        hits.sort()
        for choice, (_, k, n) in enumerate(hits[:2], 1):
            source, vid, dur, title, names = cands[k]
            if vid in chosen:
                continue
            chosen[vid] = True
            rows.append({"source": source, "id": vid, "url": f"https://youtu.be/{vid}", "title": title, "gloss": n,
                         "aliases": "|".join(x for x in names if x != n), "gap_word": word, "gap_rank": rank,
                         "gap_count": count, "choice": choice, "seconds": dur, "file": "",
                         "licence": SOURCES[source]["licence"]})
        # every gap word the first choice also covers needs no video of its own
        _, k0, n0 = hits[0]
        for x in cands[k0][4]:
            for w2 in by5.get(x[:5], ()):
                if w2 in gap_words and w2 not in covered_by and (x == w2 or compatible(x, w2)) and safe(x):
                    covered_by[w2] = cands[k0][1]
    cover_words = {}
    for w2, vid in covered_by.items():
        cover_words.setdefault(vid, []).append(w2)
    counts = {w: n for _, w, n in gap}
    for r in rows:
        ws = cover_words.get(r["id"], [])
        r["covers"] = "|".join(ws[:12])
        r["covers_count"] = sum(counts[w] for w in ws)
        r["tier"] = 0 if r["choice"] == 1 else 1
    # then everything else the channels hold (the largest lexicon possible): TiDiSLaM's other terms, then the
    # dictionary's other words, multi-word titles included («ALLAH RAZI OLSUN», «KILIÇ BALIĞI»: the glosser matches
    # phrases); in channel order
    for tier, source in ((2, "tidislam"), (3, "tid_sozluk")):
        for vid, dur, title in channel_listing(source):
            if dur > SOURCES[source]["max_seconds"] or vid in used or vid in chosen:
                continue
            names = title_names(title)
            if not names:
                continue
            chosen[vid] = True
            rows.append({"source": source, "id": vid, "url": f"https://youtu.be/{vid}", "title": title,
                         "gloss": names[0], "aliases": "|".join(names[1:]), "gap_word": "", "gap_rank": "",
                         "gap_count": 0, "choice": 0, "seconds": dur, "file": "", "covers": "", "covers_count": 0,
                         "licence": SOURCES[source]["licence"], "tier": tier})
    rows.sort(key=lambda r: (r["tier"], r["gap_rank"] if r["tier"] < 2 else 0))  # stable: channel order in a tier
    for i, r in enumerate(rows):
        r["order"] = i
    for source in SOURCES:
        write_manifest(source, [r for r in rows if r["source"] == source])
    print(f"{len(rows)} videos; " + ", ".join(f"tier {t} ({TIERS[t]}): {sum(r['tier'] == t for r in rows)}"
                                              for t in TIERS)
          + f"; the gap videos cover {len(covered_by)} gap words ({sum(counts[w] for w in covered_by)} occurrences); "
          + ", ".join(f"{s}: {sum(r['source'] == s for r in rows)}" for s in SOURCES))


def write_manifest(source, rows):
    path = src_dir(source) / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def manifest(source=None):
    out = []
    for s in ([source] if source else SOURCES):
        p = src_dir(s) / "manifest.csv"
        if p.exists():
            out += [dict(r, source=s) for r in csv.DictReader(open(p, encoding="utf-8"))]
    return out


def video_of(source, vid):
    for d in (src_dir(source) / "videos", OLD_VIDEOS / source):
        if d.exists():
            v = next((p for p in d.glob(f"{vid}.*") if p.suffix in (".mp4", ".webm", ".mkv")), None)
            if v is not None:
                return v
    return None


def downloading():
    return any(WORK.glob("DOWNLOADING_*"))


def download(k=0, n=1, batch=20, pause=(2, 5), backoff=1800):
    """Download worker k of n: every n-th video of the manifests in their order (gap words first, then TiDiSLaM's
    other terms, then the dictionary's other words), in batches of one source with pauses between videos, until
    MAX_VIDEOS or MAX_BYTES; resumable. Backs off 30 min when YouTube refuses most of a batch, and stops after six
    refusals. F:/tid_new/DOWNLOADING_<k> marks it running (the extract workers wait for more videos meanwhile)."""
    rows = sorted(manifest(), key=lambda r: int(r["order"]))
    vids = lambda: [p for s in SOURCES for p in (src_dir(s) / "videos").glob("*") if p.suffix in (".mp4", ".webm", ".mkv")]
    size = lambda: sum(p.stat().st_size for p in vids())
    gone = set()
    for s in SOURCES:
        (src_dir(s) / "videos").mkdir(parents=True, exist_ok=True)
        for g in (src_dir(s) / "videos").glob("unavailable*.txt"):
            gone |= set(g.read_text(encoding="utf-8").split())
    todo = [r for i, r in enumerate(rows) if i % n == k and r["id"] not in gone and not video_of(r["source"], r["id"])]
    refusals, got_n = 0, 0
    marker = WORK / f"DOWNLOADING_{k}"
    marker.write_text("running", encoding="utf-8")
    try:
        while todo and len(vids()) < MAX_VIDEOS and size() < MAX_BYTES:
            source = todo[0]["source"]  # the next videos of the first one's source (one output folder per call)
            chunk = [r for r in todo if r["source"] == source][:batch]
            folder = src_dir(source) / "videos"
            urls = WORK / f"urls_{k}.txt"
            urls.write_text("\n".join(r["url"] for r in chunk), encoding="utf-8")
            r = ytdlp("-a", str(urls), "--ignore-errors", "-f", "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/b",
                      "--sleep-requests", "0.5", "--sleep-interval", str(pause[0]), "--max-sleep-interval", str(pause[1]),
                      "-o", str(folder / "%(id)s.%(ext)s"))
            got = {x["id"] for x in chunk if video_of(source, x["id"])}
            if len(got) < len(chunk) / 2:  # YouTube throttles with «Video unavailable» for the rest of a batch
                if got:
                    todo = [x for x in todo if x["id"] not in got]
                    got_n += len(got)
                refusals += 1
                if refusals >= 6:
                    print(f"worker {k}: YouTube keeps refusing: stopped; run download again later", flush=True)
                    print(r.stderr[-800:])
                    break
                print(f"worker {k}: batch refused ({refusals}/6): backing off {backoff // 60} min\n{r.stderr[-400:]}",
                      flush=True)
                time.sleep(backoff)
                continue
            refusals = 0
            missed = [x["id"] for x in chunk if x["id"] not in got]  # failed while its neighbours downloaded: gone
            if missed:
                with open(folder / f"unavailable_{k}.txt", "a", encoding="utf-8") as f:
                    f.write("\n".join(missed) + "\n")
            done = {x["id"] for x in chunk}
            todo = [x for x in todo if x["id"] not in done]
            got_n += len(got)
            print(f"worker {k}: +{got_n} videos; all sources {len(vids())} videos, {size() / 1e9:.2f} GB", flush=True)
    finally:
        marker.unlink(missing_ok=True)
    if not downloading():
        finish_manifests()
    print(f"download worker {k} finished: {got_n} videos", flush=True)


def finish_manifests():
    for s in SOURCES:
        rows = manifest(s)
        for r in rows:
            v = video_of(s, r["id"])
            r["file"] = str(v) if v else ""
        write_manifest(s, rows)


# ---- clips -------------------------------------------------------------------------------------------------------
FPS = 25
MIN_S, MAX_S, MIN_HAND = 0.4, 4.0, 0.7   # accepted clip length (s) and dominant-hand detection share
MAX_MISSING = 0.45                        # as for the other YouTube dictionaries (interpolate_missing's ratio)
PAD = 2
HOLD_KEEP = 8                             # frames of a final hold kept (the sign's end position)


CHEST = 0.6     # shoulder widths below the shoulder line (square units): a wrist above it is signing
MOVING = 0.025  # shoulder widths per frame: a wrist moving faster than this is signing


def on_shot(raw, settle=10):
    """Frames showing the main shot's signer: shoulders seen, the neck within half a shoulder width of its median place
    (TiDiSLaM's intro cards show the signer small at the side, then dissolve into the main shot), and not in the
    first `settle` frames after the signer comes into place (the dissolve itself, two images overlapping)."""
    from isharati.pose.keypoints import L_SH, R_SH
    neck = (raw[:, R_SH, :2] + raw[:, L_SH, :2]) / 2
    sw = np.abs(raw[:, R_SH, 0] - raw[:, L_SH, 0])
    ok = ~np.isnan(neck[:, 0])
    if not ok.any():
        return ok
    med, msw = np.nanmedian(neck[ok], axis=0), np.nanmedian(sw[ok])
    with np.errstate(invalid="ignore"):
        place = ok & (np.linalg.norm(neck - med, axis=1) < 0.5 * msw) & (np.abs(sw - msw) < 0.25 * msw)
    out = place.copy()
    for i in range(1, len(place)):
        if place[i] and not place[i - 1]:
            out[i:i + settle] = False
    return out


def active(raw):
    """Per frame, is the signer signing? islamic_terms.signing_runs counts a seen hand above waist level (1.3 shoulder
    widths below the shoulders); these signers rest with their hands clasped in front of the belly, above that line,
    so here a wrist (the body model's, which keeps a blurred hand the hand model loses) is signing when it is above
    chest level (CHEST), or when it moves (MOVING, displacement over 4 frames) while the hand model sees its hand (a
    hand out of the frame at rest jitters); on the main shot only (on_shot). raw in square units (square())."""
    from isharati.pose.keypoints import L_SH, L_WR, LH, R_SH, R_WR, RH
    sh_y = (raw[:, R_SH, 1] + raw[:, L_SH, 1]) / 2
    sw = np.abs(raw[:, R_SH, 0] - raw[:, L_SH, 0])
    sw = np.where(np.isfinite(sw) & (sw > 1e-6), sw, np.nanmedian(sw))
    shot = on_shot(raw)
    out = np.zeros(len(raw), bool)
    for wrist, hand in ((R_WR, RH), (L_WR, LH)):
        seen = ~np.isnan(raw[:, hand, 0]).all(axis=1)
        xy = raw[:, wrist, :2].copy()
        for c in range(2):
            v, ok = xy[:, c], ~np.isnan(xy[:, c])
            xy[:, c] = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok]) if ok.any() else 0.0
        # displacement over 4 frames: a clasped hand's landmark jitter cancels out, a movement does not
        sp = np.zeros(len(xy))
        sp[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=1) / 4 / sw[2:-2]
        with np.errstate(invalid="ignore"):
            high = raw[:, wrist, 1] < sh_y + CHEST * sw
            out |= shot & ((seen & (high | (sp > MOVING))) | high)
    return out


def square(raw, meta):
    """The track in square units (x times the frame's aspect), for measuring motion; the clip keeps the builders'
    crop-normalised coordinates."""
    out = raw.copy()
    out[..., 0] *= float(meta.get("aspect", 1.0))
    return out


def signing_runs(raw, min_len=8, max_gap=6):
    """Signing frames (active) as [start, end) runs, gaps of up to max_gap frames bridged, as
    islamic_terms.signing_runs builds them."""
    runs, start, gap = [], None, 0
    for i, a in enumerate(active(raw)):
        if a:
            start = i if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(raw)))
    return [(a, b) for a, b in runs if b - a >= min_len]


def hand_features(raw):
    """Per frame, in shoulder widths relative to the neck: both wrists and both hand centroids (8 numbers); NaN where
    unseen."""
    from isharati.pose.keypoints import L_SH, L_WR, LH, R_SH, R_WR, RH
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


def trim_hold(raw, a, b):
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
        if L - j <= 0.8 * FPS and i >= 0.3 * FPS:
            b = a + i + HOLD_KEEP
    return a, b


def first_repetition(raw, a, b):
    """When a run performs the sign twice without lowering the hands, the end of the first performance: the smallest
    lag p (>= 0.4 s) at which the opening of the run (its first p frames, while they move) comes back closely."""
    f = hand_features(raw[a:b])
    L = len(f)
    if L < 2 * int(0.4 * FPS):
        return b
    for p in range(int(0.4 * FPS), L // 2 + 1):
        w = min(p, L - p)
        x, y = f[:w], f[p:p + w]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 0.6 * x.size:
            continue
        if np.nansum(np.abs(np.diff(f[:p], axis=0))) < 1.0:  # the opening must move (not a hold matching itself)
            continue
        if np.mean(np.abs(x[ok] - y[ok])) < 0.08:
            return a + p
    return b


def fingerspelling_start(raw, a, b):
    """Start (absolute frame) of a fingerspelled stretch inside [a, b), or None: at least 1 s in which the wrists stay
    near one place while the handshape keeps changing (fingertips relative to the wrist), the signature of spelling
    the word letter by letter."""
    from isharati.pose.keypoints import L_SH, L_WR, LH, R_SH, R_WR, RH
    seg = raw[a:b]
    speed = wrist_speed(seg)
    sw = np.nanmedian(np.abs(raw[:, R_SH, 0] - raw[:, L_SH, 0])) or 1.0
    shape = np.zeros(len(seg))
    for hand, wr in ((RH, R_WR), (LH, L_WR)):
        rel = seg[:, hand, :2] - seg[:, wr:wr + 1, :2]
        with np.errstate(all="ignore"):
            d = np.nanmean(np.linalg.norm(np.diff(rel, axis=0), axis=-1), axis=1) / sw
        shape = np.maximum(shape, np.r_[0, np.nan_to_num(d)])
    spell = (speed < 0.02) & (shape > 0.04)
    for i in range(0, max(0, len(seg) - FPS)):
        if spell[i:i + FPS].mean() > 0.6:
            return a + i
    return None


def clip_span(raw, source):
    """The kept span of a video, (a, b, reason): ONE performance of the sign.
    1. the first signing run (signing_runs: a seen hand above chest level or moving, gaps of up to 6 frames bridged)
       of at least 8 frames.
       tid_sozluk shows one sign per clip (rest, sign, rest): when more signing follows, it must stand apart (0.8 s
       of rest), else which part is the sign is ambiguous (a compound, «YAKIN / UZAK», a detection gap) -> rejected.
       TiDiSLaM shows the sign under its title card, a rest, then its meaning explained in TİD and often the sign
       again at the end: the first run is the sign (a run that flows on into the explanation fails the length check);
    2. a fingerspelled stretch inside the run is cut off (rejected when the run starts with it);
    3. a repetition inside the run (hands not lowered between the two) is cut to the first performance;
    4. a final hold followed only by lowering the hands is cut to HOLD_KEEP frames; then PAD frames either side.
    reason is None when the span passes the checks (MIN_S..MAX_S, dominant hand seen in MIN_HAND of the frames,
    exactly one signing run inside)."""
    from isharati.pose.keypoints import LH, R_SH, RH
    if np.isnan(raw[:, R_SH, 0]).all():
        return None, None, "no shoulders"
    runs = signing_runs(raw)
    if not runs:
        return None, None, "no signing run"
    a, b = runs[0]
    if source != "tidislam" and len(runs) > 1 and runs[1][0] - b < 0.8 * FPS:
        return a, b, f"first run not separated from further signing ({(runs[1][0] - b) / FPS:.2f} s rest)"
    fs = fingerspelling_start(raw, a, b)
    if fs is not None:
        if fs - a < int(MIN_S * FPS):
            return a, b, "starts with fingerspelling"
        b = fs
    b = first_repetition(raw, a, b)
    a, b = trim_hold(raw, a, b)
    a, b = max(0, a - PAD), min(len(raw), b + PAD)
    seg = raw[a:b]
    lh = ~np.isnan(seg[:, LH, 0]).all(axis=1)
    rh = ~np.isnan(seg[:, RH, 0]).all(axis=1)
    dom = max(lh.mean(), rh.mean())
    n_runs = len([r for r in signing_runs(seg, min_len=6) if r[1] - r[0] >= 6])
    dur = (b - a) / FPS
    if dur < MIN_S:
        return a, b, f"too short ({dur:.2f} s)"
    if dur > MAX_S:
        return a, b, f"too long ({dur:.2f} s)"
    if dom < MIN_HAND:
        return a, b, f"dominant hand seen in {dom:.0%} of frames"
    if n_runs != 1:
        return a, b, f"{n_runs} signing runs in the span"
    return a, b, None


def load_track(source, vid):
    sys.path.insert(0, str(ROOT / "scripts" / "face"))
    from attach import fifty
    z = np.load(src_dir(source) / "tracks" / f"{vid}.npz")
    tr = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    return tr, fifty(tr), json.loads(str(z["meta"]))


def track_video(source, vid):
    """Holistic task track (scripts/face/track.py), once per video, kept on F:."""
    tr_path = src_dir(source) / "tracks" / f"{vid}.npz"
    if tr_path.exists():
        return tr_path
    sys.path.insert(0, str(ROOT / "scripts" / "face"))
    from track import track
    data, meta = track(video_of(source, vid))
    tr_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tr_path.with_name(tr_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **data)
    os.replace(tmp, tr_path)
    return tr_path


def extract_one(r):
    """Track, clip, face of one manifest row. Returns a status string."""
    sys.path.insert(0, str(ROOT / "scripts" / "face"))
    from attach import save_face
    from isharati.pose.keypoints import interpolate_missing, normalize
    source, vid = r["source"], r["id"]
    npy, bad = OUT / "signs" / f"{vid}.npy", OUT / "signs" / f"{vid}.failed"
    if npy.exists() or bad.exists():
        return "skip"
    video = video_of(source, vid)
    if video is None:
        return "no video"
    track_video(source, vid)
    tr, raw, meta = load_track(source, vid)
    a, b, reason = clip_span(square(raw, meta), source)
    if reason is None:
        filled, missing = interpolate_missing(raw[a:b])
        if missing > MAX_MISSING:
            reason = f"missing hand {missing:.2f}"
    t = tr["time"]
    if reason is not None:
        bad.write_text(json.dumps({"source": source, "gloss": r["gloss"], "reason": reason,
                                   "start": None if a is None else round(float(t[a]), 3),
                                   "end": None if b is None else round(float(t[b - 1]), 3)}, ensure_ascii=False),
                       encoding="utf-8")
        return "rejected"
    clip = normalize(filled).astype(np.float32)
    hands = float(np.mean(~(np.isnan(raw[a:b, 8:29, 0]).all(1) & np.isnan(raw[a:b, 29:50, 0]).all(1))))
    row = {"sign_id": f"{PREFIX}_{vid}", "gloss": r["gloss"], "dataset": SOURCES[source]["dataset"],
           "video": video.name, "start": round(float(t[a]), 3), "end": round(float(t[b - 1]), 3), "frames": b - a,
           "match": 0.0, "found": True}
    save_face(FACE, row["sign_id"], tr, raw, meta, a, b - a, 0.0, row, clip)
    face_manifest_append(row)
    np.save(npy, clip)
    npy.with_suffix(".json").write_text(json.dumps({"source": source, "gloss": r["gloss"], "a": a, "b": b,
                                                    "missing": round(missing, 3), "hands": round(hands, 3),
                                                    "start": row["start"], "end": row["end"], "frames": b - a,
                                                    "face_detected": row["face_detected"]}, ensure_ascii=False),
                                        encoding="utf-8")
    return "ok"


def face_manifest_append(row):
    """One row per sign in data/face/tr/manifest.jsonl (scripts/face/attach.py's), appended under a lock file."""
    lock = FACE / "manifest.jsonl.lock"
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        fd = None  # a stale lock: write anyway
    try:
        with open(FACE / "manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if fd is not None:
            os.close(fd)
            lock.unlink(missing_ok=True)


def extract(k, n, only=None):
    """Worker k of n over the downloaded videos, in manifest order; after each video the entries file is rewritten
    (atomically), so the lexicon grows while the run goes on. Keeps looping while a download worker runs
    (F:/tid_new/DOWNLOADING_*). Worker 0 prints the count and the coverage every 200 new signs."""
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    FACE.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    state = EntryState()
    shown = len(list((OUT / "signs").glob("*.npy"))) // 200
    while True:
        running = downloading()
        rows = sorted(manifest(), key=lambda r: int(r["order"]))
        rows = [r for i, r in enumerate(rows) if i % n == k and (only is None or r["id"] in only)]
        todo = [r for r in rows if video_of(r["source"], r["id"]) and not (OUT / "signs" / f"{r['id']}.npy").exists()
                and not (OUT / "signs" / f"{r['id']}.failed").exists()]
        for r in todo:
            try:
                s = extract_one(r)
            except Exception as e:  # a broken file: note it, go on
                s = "error"
                print(r["id"], "error:", repr(e), flush=True)
                (OUT / "signs" / f"{r['id']}.failed").write_text(
                    json.dumps({"source": r["source"], "gloss": r["gloss"], "reason": f"error: {e!r}"}), encoding="utf-8")
            stats[s] += 1
            if s == "ok":
                write_entries(state)
                done = len(list((OUT / "signs").glob("*.npy")))
                if k == 0 and done // 200 > shown:
                    shown = done // 200
                    cov = coverage(OLD_ENTRIES + [OUT / "entries.jsonl"])
                    print(f"{done} signs; TİD content-word coverage {cov['token_coverage']:.2%} "
                          f"({cov['glosses']} glosses)", flush=True)
            if sum(stats.values()) % 50 == 0:
                print(f"worker {k}/{n}: {dict(stats)}", flush=True)
        if not running or only is not None:
            break
        time.sleep(30)
    write_entries(state)
    print(f"worker {k}/{n} finished: {dict(stats)}", flush=True)


# ---- inspection and QA -------------------------------------------------------------------------------------------
def overview(source, vid, step=0.4):
    """Inspection sheet of a whole video: a frame every `step` s, framed green inside the kept span, orange inside a
    signing run, grey elsewhere, with the hands seen (B/L/R/.); and a timeline plot (wrist speed, hands seen, the
    signing frames, runs, kept span). Both in F:/tid_new/<source>/qa/."""
    import cv2
    tr, raw, meta = load_track(source, vid)
    runs = signing_runs(square(raw, meta))
    a, b, reason = clip_span(square(raw, meta), source)
    video = video_of(source, vid)
    cap = cv2.VideoCapture(str(video))
    T, tiles = len(raw), []
    for i in range(0, T, max(1, int(step * FPS))):
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
        col = (0, 255, 0) if kept else (0, 165, 255) if inrun else (128, 128, 128)
        cv2.rectangle(img, (0, 0), (199, img.shape[0] - 1), col, 5 if kept else 2)
        cv2.putText(img, f"{t:.1f} {'B' if lh and rh else 'L' if lh else 'R' if rh else '.'}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cap.release()
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[k:k + cols], axis=1) for k in range(0, len(tiles), cols)]
    qa_dir = src_dir(source) / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qa_dir / f"overview_{vid}.jpg"), np.concatenate(rows, axis=0))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = tr["time"]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, wrist_speed(square(raw, meta)), lw=1, label="wrist speed (sw/frame)")
    ax.plot(t, ~np.isnan(raw[:, 29, 0]) * 0.11, lw=1, label="right hand seen")
    ax.plot(t, ~np.isnan(raw[:, 8, 0]) * 0.10, lw=1, label="left hand seen")
    ax.plot(t, active(square(raw, meta)) * 0.12, lw=1, label="signing (active)")
    for s, e in runs:
        ax.axvspan(t[s], t[e - 1], color="orange", alpha=0.15)
    if a is not None:
        ax.axvspan(t[a], t[b - 1], color="green", alpha=0.3)
    ax.set_ylim(0, 0.16)
    ax.set_xlabel("s")
    ax.set_title(f"{vid}  kept {None if a is None else (round(float(t[a]), 2), round(float(t[b - 1]), 2))}  "
                 f"{reason or 'accepted'}", fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(qa_dir / f"timeline_{vid}.png", dpi=90)
    plt.close(fig)
    return {"runs_s": [(round(s / FPS, 2), round(e / FPS, 2)) for s, e in runs],
            "kept_s": None if a is None else (round(a / FPS, 2), round(b / FPS, 2)), "reason": reason}


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
    cv2.putText(bar, title[:110], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.concatenate([bar, sheet], axis=0))


def qa(n=20, render=5, seed=0):
    """Contact sheets (a frame just before, start, middle, end, a frame just after the kept span) of n random accepted
    clips and skeleton videos of `render` of them in F:/tid_new/qa/; the reject list F:/tid_new/rejects.csv; counts."""
    import random
    from isharati.pose.render import SkeletonRenderer
    out = WORK / "qa"
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in manifest()}
    ok = sorted(p.stem for p in (OUT / "signs").glob("*.npy") if p.stem in rows)
    pick = random.Random(seed).sample(ok, min(n, len(ok)))
    for k, vid in enumerate(pick):
        info = json.loads((OUT / "signs" / f"{vid}.json").read_text(encoding="utf-8"))
        s, e = info["start"], info["end"]
        r = rows[vid]
        contact_sheet(video_of(r["source"], vid), [s - 0.12, s, (s + e) / 2, e, e + 0.12],
                      ["before", "start", "middle", "end", "after"], out / f"accepted_{k:02d}_{vid}.jpg",
                      f"{r['source']} | {r['gloss']} | title: {r['title']} | kept {s:.2f}-{e:.2f}s "
                      f"hands {info['hands']:.0%}")
        if k < render:
            SkeletonRenderer().render(np.load(OUT / "signs" / f"{vid}.npy"), out / f"skeleton_{k:02d}_{vid}.mp4")
    rej, reasons = [], Counter()
    for p in sorted((OUT / "signs").glob("*.failed")):
        if p.stem not in rows:
            continue
        info = json.loads(p.read_text(encoding="utf-8"))
        r = rows[p.stem]
        rej.append({"source": r["source"], "id": p.stem, "gloss": r["gloss"], "title": r["title"],
                    "reason": info["reason"], "start": info.get("start"), "end": info.get("end")})
        reasons[(r["source"], re.sub(r"\s*\(.*|\d+%.*|[\d.]+ s.*|\d+ signing.*", "", info["reason"]).strip())] += 1
    with open(WORK / "rejects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "id", "gloss", "title", "reason", "start", "end"])
        w.writeheader()
        w.writerows(rej)
    acc = Counter(rows[v]["source"] for v in ok)
    print("accepted:", dict(acc), "rejected:", dict(Counter(r["source"] for r in rej)))
    for (s, why), c in reasons.most_common():
        print(f"  {s:10s} {c:4d}  {why}")
    print(f"{len(pick)} contact sheets, {min(render, len(pick))} skeleton videos -> {out}; rejects -> {WORK / 'rejects.csv'}")


ISLAMIC = set("""allah peygamber nebi nebî rasûlullah aleyhi radıyallahu kur namaz abdest oruç zekat zekât hac hacı
kuran dua cami mescit mescid melek şeytan cennet cehennem iman günah sevap helal haram bayram inşallah maşallah şükür
ramazan kurban ahiret secde rab rabb tövbe sadaka kıble ezan imam hadis sünnet farz mübarek cuma ibadet din kıyamet
müslüman mümin kâfir kafir münafık müşrik şirk amel cihad cihat ümmet sahabe tevhid takva sure ayet vahiy mucize
kabe umre ihram tavaf hutbe vaaz fıkıh fetva helâl tesbih zikir kader kaza rızık hidayet rahmet rahman rahim melekler
hicret hicri peygamberlik islam islâm tekbir selam""".split())


class EntryState:
    """What writing the entries needs from the corpus and the old lexicon, computed once per process."""

    def __init__(self):
        from isharati.retrieval.turkish_urdu import TR_STOP, tr_norm
        c, _, _ = corpus_counts()
        self.content = Counter({w: n for w, n in c["all"].items() if w not in TR_STOP})
        self.by5 = {}
        for w in self.content:
            self.by5.setdefault(w[:5], []).append(w)
        self.old = OldMatcher()
        self.old_ids = {}
        for p in OLD_ENTRIES:
            for e in load(p):
                if not e.get("is_letter") and tr_norm(e["gloss"]) in self.old.keys:
                    self.old_ids.setdefault(tr_norm(e["gloss"]), e["sign_id"])
        self.memo = {}

    def safe(self, key):
        """A gloss (single word) that signs at least twice as many corpus occurrences rightly as wrongly."""
        if " " in key:
            return True  # a phrase is matched whole
        if key not in self.memo:
            good, bad = effect(key, self.content, self.old, self.by5)
            self.memo[key] = good >= 2 * bad
        return self.memo[key]


def entry_rows(state):
    """The entries, mains first, then variants. Clips are taken in manifest order (gap words first); per clip its
    glosses: the title word, the gap word it was picked for, the title's other alternatives. A gloss is the clip's
    main sign when nothing signs it yet (and it would not sign other corpus words wrongly, OldMatcher); a clip whose
    title word is already signed (by the TİD dictionary or an earlier clip) becomes a variant of that sign
    ("variant_of"), which the lexicon keeps but does not use (add_words keeps the first sign of a gloss)."""
    from isharati.retrieval.turkish_urdu import tr_norm
    mains, variants, taken = [], [], {}
    for r in sorted(manifest(), key=lambda r: int(r["order"])):
        npy = OUT / "signs" / f"{r['id']}.npy"
        if not npy.exists():
            continue
        source = r["source"]
        base = {"sign_id": f"{PREFIX}_{r['id']}", "dataset": SOURCES[source]["dataset"],
                "keypoints_path": f"tid_youtube/signs/{r['id']}.npy", "review_status": "pending", "is_letter": False}
        names = list(dict.fromkeys([r["gloss"]] + ([r["gap_word"]] if r["gap_word"] else [])
                                   + [a for a in r["aliases"].split("|") if a]))
        for j, g in enumerate(names):
            key = tr_norm(g)
            if not key:
                continue
            religious = source == "tidislam" or key in ISLAMIC or r["gap_word"] in ISLAMIC
            row = {"gloss": g, **base, "is_religious": religious}
            owner = state.old_ids.get(key) or taken.get(key)
            if owner:
                if j == 0 and owner != base["sign_id"]:  # another recording of the title word
                    variants.append({**row, "variant_of": owner})
                continue
            if not state.safe(key):
                continue
            taken[key] = base["sign_id"]
            mains.append(row)
    return mains, variants


def write_entries(state=None):
    """data/lexicon_tr/tid_youtube/entries.jsonl, rewritten through a temporary file (os.replace), so a reader never
    sees half a file."""
    state = state or EntryState()
    mains, variants = entry_rows(state)
    path = OUT / "entries.jsonl"
    tmp = path.with_name(f"entries.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in mains + variants), encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:  # Windows: a reader has it open for a moment
            time.sleep(0.2)
    return mains, variants


def entries():
    mains, variants = write_entries()
    print(f"{len(mains)} glosses ({len({e['sign_id'] for e in mains})} signs, "
          f"{sum(e['is_religious'] for e in mains)} religious glosses) + {len(variants)} variants "
          f"-> {OUT / 'entries.jsonl'}")


def report_coverage():
    before = coverage(OLD_ENTRIES)
    after = coverage(OLD_ENTRIES + [OUT / "entries.jsonl"])
    res = {"before": before, "after": after, "new_glosses": after["glosses"] - before["glosses"]}
    (WORK / "coverage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gaps":
        gaps()
    elif mode == "list":
        list_videos()
    elif mode == "download":
        download(*(int(x) for x in sys.argv[2:4]))
    elif mode == "extract":
        extract(int(sys.argv[2]), int(sys.argv[3]), set(sys.argv[4:]) or None)
    elif mode == "track":
        for v in sys.argv[3:]:
            print(track_video(sys.argv[2], v), flush=True)
    elif mode == "overview":
        for v in sys.argv[3:]:
            print(v, overview(sys.argv[2], v), flush=True)
    elif mode == "qa":
        qa()
    elif mode == "entries":
        entries()
    elif mode == "coverage":
        report_coverage()
