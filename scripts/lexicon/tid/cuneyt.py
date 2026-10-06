"""TİD word signs from Cüneyt Küçükoğlu's «Hareketli Sözlük» YouTube channel (~5,700 one-word videos titled
«<Word> - işaret dili», some with a variant number «Şart 2.» or a sense «Dolu ( Yağış )») -> a gap-filling TİD source.

Each video: a red title card (~4 s), then one signer performs the sign several times (normal speed, slow motion,
normal again; some videos show two or three variants in turn), then the card again. Per video:
  1. track with the Holistic task (scripts/face/track.py's track(), in memory: no track file is kept);
  2. hand-raised runs: a detected hand above the hips on frames where the body is seen (the cards have none), gaps of
     up to GAP frames bridged, runs shorter than MIN_RUN dropped;
  3. DTW (isharati.pose.dtw) between the runs on the hand points (window_frame of each run), complete-linkage groups:
     two groups join when their link is <= HARD and <= RATIO x the larger internal spread (at least BASE), so the
     three passes of one sign group together and a different sign stays apart (zekat: 0.16-0.22 within, 0.33-0.43
     across);
  4. the shortest member of the largest group of >= 2 runs (ties: the earliest group) is the sign; no agreeing pair
     -> rejected. QA as tid_youtube.py: 0.4-4 s, dominant hand in >= 70% of frames, one signing run, missing hand;
  5. clip = window_frame(rest_undetected_hands(fifty(track))[span]) and its face (attach.save_face) as
     reviewed_recordings.py; the face manifest row appended under tid_youtube's lock. Videos are kept.

Queue order: (0) titles whose word is a corpus gap (the current lexicon matches no form of it, or only by the prefix
fallback; F:/tid_new/missing_tid.csv words and results/tr_uncovered_lemmas.tsv, matched by zeyrek lemma), most frequent
first; (1) their numbered/sense variants; (2) other corpus words by frequency; (3) the rest of the channel.

entries.jsonl (rewritten atomically by `entries`, which the extract workers start every few accepted videos): a gloss
the lexicon without this source signs (any route but the prefix fallback), or an earlier clip of this source, gets a
"variant_of" row; sense titles are glossed «word (sense)». tid_lexicon() loads it with add_words(stems=True).

  PYTHONPATH=src py scripts/lexicon/tid/cuneyt.py queue|download|entries|qa|supervise
  "D:/islam/gathring data/.venv/Scripts/python" scripts/lexicon/tid/cuneyt.py extract <k> <n>
Output: data/lexicon_tr/cuneyt/{signs/<id>.npy|.json|.failed, entries.jsonl}, data/face/tr/cuneyt_<id>.npz;
videos, queue, logs on F:/cuneyt (status: bash F:/cuneyt/status.sh).
"""
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "face"), str(ROOT / "scripts" / "lexicon" / "arsl"),
                str(ROOT / "scripts" / "lexicon" / "tid")]
from isharati.config import DATA  # noqa: E402

WORK = Path(os.environ.get("CUNEYT_WORK", r"F:\cuneyt"))
LISTING = WORK / "channel_videos.txt"   # id | seconds | title
QUEUE = WORK / "queue.csv"
VIDEOS = WORK / "videos"
OUT = DATA / "lexicon_tr" / "cuneyt"
FACE = DATA / "face" / "tr"
PREFIX, DATASET = "cuneyt", "tid_cuneyt"
CHANNEL = "Cüneyt Küçükoğlu (Türk İşaret Dili), «Hareketli Sözlük»"
MP_PYTHON = Path(r"D:\islam\gathring data\.venv\Scripts\python.exe")
PY = Path(sys.executable) if "gathring" not in sys.executable else Path("py")
YTDLP = Path(r"D:\islam\yt-dlp")
CAP_BYTES, MIN_FREE, LOW_FREE = 7 * 1024 ** 3, 2 * 1024 ** 3, 4 * 1024 ** 3
FPS, GAP, MIN_RUN, PAD = 25, 6, 10, 3
HARD, RATIO, BASE = 1.0, 1.5, 0.25
TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_title(title):
    """(gloss, variant number, sense) of «Şart 2. - işaret dili», «Dolu ( Yağış ) işaret dili»; None for other videos."""
    t = title.replace("’", "'")
    m = re.match(r"(.*?)[\s\-–]*[İiIı]şaret\s+[Dd]il[iı]\s*$", t)
    if not m:
        return None
    name, variant, senses = m.group(1), 1, []
    for p in re.findall(r"\(([^)]*)\)", name):
        d = re.search(r"\b(\d+)\s*\.?\s*$", p.strip())
        if d:
            variant = int(d.group(1))
            p = p[:d.start()]
        p = p.strip(" -.")
        if p:
            senses.append(p)
    name = re.sub(r"\([^)]*\)", " ", name)
    ilce = re.search(r"\s-\s*(ilçe|il)\s*$", name)
    if ilce:
        senses.append(ilce.group(1))
        name = name[:ilce.start()]
    d = re.search(r"\s(\d+)\s*\.?\s*$", " " + name.strip(" -–"))
    if d:
        variant = int(d.group(1))
        name = (" " + name.strip(" -–"))[:d.start()]
    name = re.sub(r"\s+", " ", name).strip(" -–.,").translate(TR_LOWER).lower()
    if not re.search(r"[a-zçğıöşüâîû]", name) or len(name) > 40 or re.search(r"\d", name):
        return None
    sense = re.sub(r"\s+", " ", ", ".join(senses)).translate(TR_LOWER).lower() or ""
    return name, variant, sense


def display_gloss(gloss, sense):
    return f"{gloss} ({sense})" if sense else gloss


# ---- queue -------------------------------------------------------------------------------------------------------
def base_lexicon():
    from isharati.lexicon.turkish import tid_lexicon
    return tid_lexicon(cuneyt=False)


def signed(lex, gloss):
    """The sign the lexicon without this source gives the gloss (any route but the prefix fallback), or None."""
    e, route = lex.match(gloss)
    return None if route in (None, "prefix") else e


def queue():
    import tid_youtube as ty
    from isharati.retrieval.turkish_urdu import TR_STOP
    from isharati.text import turkish_morph as morph
    WORK.mkdir(parents=True, exist_ok=True)
    lex = base_lexicon()
    c, _, _ = ty.corpus_counts()
    gap, cov = Counter(), Counter()
    missing = {r["word"] for r in csv.DictReader(open(ty.GAPS, encoding="utf-8")) if r["fragment"] == "0"}
    for w, n in c["all"].items():
        if w in TR_STOP or not w.isalpha():
            continue
        lems = list(dict.fromkeys([morph.fold(w)] + morph.lemmas(w)))
        if w in missing and signed(lex, w) is None:
            for lem in lems:
                gap[lem] += n
        else:
            for lem in lems:
                cov[lem] += n
    unc = ROOT / "results" / "tr_uncovered_lemmas.tsv"
    if unc.exists():
        for row in csv.DictReader(open(unc, encoding="utf-8"), delimiter="\t"):
            k = morph.fold(row["lemma"])
            gap[k] = max(gap[k], int(row["tokens"]))
    words = {w for line in LISTING.read_text(encoding="utf-8").splitlines() for w in
             (parse_title(line.split("|", 2)[-1]) or ("",))[0].split()}
    for _ in range(10):  # the lemma cache, saved once (another process may hold the file a moment)
        try:
            morph.analyze(sorted(words))
            break
        except PermissionError:
            time.sleep(2)
    rows = []
    for k, line in enumerate(LISTING.read_text(encoding="utf-8").splitlines()):
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) < 3:
            continue
        vid, dur, title = parts
        p = parse_title(title)
        if p is None or float(dur or 0) > 120:
            continue
        gloss, variant, sense = p
        key = morph.fold(gloss)
        has = signed(lex, gloss) is not None
        g, n = gap.get(key, 0), cov.get(key, 0)
        plain = variant == 1 and not sense
        if g and not has:
            tier, score = (0 if plain else 1), g
        elif g + n:
            tier, score = 2, g + n
        else:
            tier, score = 3, 0
        rows.append({"id": vid, "seconds": dur, "title": title, "gloss": gloss, "variant": variant, "sense": sense,
                     "tier": tier, "score": score, "signed": int(has), "channel_order": k})
    rows.sort(key=lambda r: (r["tier"], -r["score"], r["channel_order"]))
    for i, r in enumerate(rows):
        r["order"] = i
    tmp = QUEUE.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, QUEUE)
    print(f"{len(rows)} videos -> {QUEUE}; per tier {dict(Counter(r['tier'] for r in rows))}")


def read_queue():
    return sorted(csv.DictReader(open(QUEUE, encoding="utf-8")), key=lambda r: int(r["order"]))


# ---- download ----------------------------------------------------------------------------------------------------
def video_path(vid):
    p = VIDEOS / f"{vid}.mp4"
    return p if p.exists() else None


def videos_bytes():
    return sum(p.stat().st_size for p in VIDEOS.glob("*.mp4"))


def download(batch=10, backoff=900):
    """The queue's videos in order, 360p, anonymous, ~1 s between requests; stops at CAP_BYTES or below MIN_FREE free
    on F:, skips duplicate-only words (tier >= 2 already signed, numbered variants) below LOW_FREE. Resumable."""
    VIDEOS.mkdir(parents=True, exist_ok=True)
    gone_f = VIDEOS / "unavailable.txt"
    gone = set(gone_f.read_text(encoding="utf-8").split()) if gone_f.exists() else set()
    todo = [r for r in read_queue() if r["id"] not in gone and not video_path(r["id"])]
    refusals = 0
    while todo:
        free = shutil.disk_usage(VIDEOS).free
        if free < MIN_FREE or videos_bytes() > CAP_BYTES:
            print(f"disk: {free / 1e9:.1f} GB free, videos {videos_bytes() / 1e9:.2f} GB: stopped", flush=True)
            (WORK / "DOWNLOAD_STOPPED").write_text("disk", encoding="utf-8")
            return
        if free < LOW_FREE:
            todo = [r for r in todo if not (int(r["tier"]) >= 2 and r["signed"] == "1") and r["variant"] == "1"]
        chunk = todo[:batch]
        urls = WORK / "urls.txt"
        urls.write_text("\n".join(f"https://www.youtube.com/watch?v={r['id']}" for r in chunk), encoding="utf-8")
        res = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--ignore-errors", "-a", str(urls),
                              "-f", "bv*[height<=360][ext=mp4]+ba/b[height<=360]", "--merge-output-format", "mp4",
                              "--sleep-requests", "1", "-o", str(VIDEOS / "%(id)s.%(ext)s")],
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONPATH": str(YTDLP)})
        got = [r for r in chunk if video_path(r["id"])]
        if len(got) < len(chunk) / 2:
            refusals += 1
            print(f"batch refused ({refusals}): {res.stderr[-300:]}", flush=True)
            if refusals >= 6:
                return
            todo = [r for r in todo if r not in got]
            time.sleep(backoff)
            continue
        refusals = 0
        missed = [r["id"] for r in chunk if r not in got]
        if missed:
            with open(gone_f, "a", encoding="utf-8") as f:
                f.write("\n".join(missed) + "\n")
        todo = todo[len(chunk):]
        print(f"{len(list(VIDEOS.glob('*.mp4')))} videos, {videos_bytes() / 1e9:.2f} GB", flush=True)
    (WORK / "DOWNLOAD_DONE").write_text("done", encoding="utf-8")


# ---- selection and clips -----------------------------------------------------------------------------------------
def hand_runs(track):
    """[start, end) runs of a detected hand above the hips, on frames with the body seen."""
    p = track["pose"]
    sw = np.abs(p[:, 11, 0] - p[:, 12, 0])
    hip = (p[:, 23, 1] + p[:, 24, 1]) / 2
    body = ~np.isnan(p[:, 11, 0]) & ~np.isnan(p[:, 12, 0])
    act = np.zeros(len(p), bool)
    with np.errstate(invalid="ignore"):
        for h in (track["right_hand"], track["left_hand"]):
            act |= ~np.isnan(h[:, 0, 0]) & (np.nan_to_num(h[:, 0, 1], nan=9) < hip - 0.15 * sw)
    act &= body
    runs, s, gap = [], None, 0
    for t, a in enumerate(act):
        if a:
            s, gap = (t if s is None else s), 0
        elif s is not None:
            gap += 1
            if gap > GAP:
                runs.append((s, t - gap + 1))
                s, gap = None, 0
    if s is not None:
        runs.append((s, len(act)))
    return [(a, b) for a, b in runs if b - a >= MIN_RUN]


def group_runs(D):
    """Complete-linkage groups of runs (see the module docstring): [(members, internal spread)]."""
    cl = [([i], 0.0) for i in range(len(D))]
    while len(cl) > 1:
        best = None
        for x, y in itertools.combinations(range(len(cl)), 2):
            h = max(D[i, j] for i in cl[x][0] for j in cl[y][0])
            if best is None or h < best[0]:
                best = (h, x, y)
        h, x, y = best
        if h > HARD or h > RATIO * max(cl[x][1], cl[y][1], BASE):
            break
        merged = (sorted(cl[x][0] + cl[y][0]), h)
        cl = [c for k, c in enumerate(cl) if k not in (x, y)] + [merged]
    return cl


def select(track, raw50):
    """(a, b, info, reason): the shortest run of the largest agreeing group."""
    from attach import window_frame
    from isharati.pose.dtw import dtw_distance
    runs = hand_runs(track)
    info = {"runs_s": [[round(a / FPS, 2), round(b / FPS, 2)] for a, b in runs]}
    if len(runs) < 2:
        return None, None, info, f"{len(runs)} hand-raised run(s)"
    feats = [window_frame(raw50[a:b])[0][:, 8:50, :2] for a, b in runs]
    D = np.zeros((len(runs), len(runs)))
    for i, j in itertools.combinations(range(len(runs)), 2):
        D[i, j] = D[j, i] = dtw_distance(feats[i], feats[j])
    groups = [g for g in group_runs(D) if len(g[0]) >= 2]
    info["groups"] = [[m for m in g[0]] for g in group_runs(D)]
    if not groups:
        return None, None, info, "no two runs agree"
    members, spread = max(groups, key=lambda g: (len(g[0]), -min(g[0])))
    pick = min(members, key=lambda i: runs[i][1] - runs[i][0])
    others = [i for i in range(len(runs)) if i not in members]
    info.update(group=members, chosen=pick, agreement=round(float(spread), 3),
                separation=None if not others else round(float(min(D[i, j] for i in members for j in others)), 3))
    a, b = runs[pick]
    return max(0, a - PAD), min(len(raw50), b + PAD), info, None


def qa_span(raw50, track, meta, a, b):
    import tid_youtube as ty
    from isharati.pose.keypoints import interpolate_missing
    dur = (b - a) / FPS
    if dur < ty.MIN_S or dur > ty.MAX_S:
        return f"{'too short' if dur < ty.MIN_S else 'too long'} ({dur:.2f} s)"
    raw = ty_fifty(track)  # before rest_undetected_hands: a resting hand counts as unseen
    seg = raw[a:b]
    dom = max((~np.isnan(seg[:, 8:29, 0]).all(1)).mean(), (~np.isnan(seg[:, 29:50, 0]).all(1)).mean())
    if dom < ty.MIN_HAND:
        return f"dominant hand seen in {dom:.0%} of frames"
    # one run: the hand-raised runs of the span itself (tid_youtube's chest-level runs split these signs, which start
    # and end at the hips: they rejected the team's zekat cut)
    n = len(hand_runs({k: track[k][a:b] for k in ("pose", "left_hand", "right_hand")}))
    if n != 1:
        return f"{n} signing runs in the span"
    _, missing = interpolate_missing(seg)
    if missing > ty.MAX_MISSING:
        return f"missing hand {missing:.2f}"
    return None


def ty_fifty(track):
    from attach import fifty
    return fifty(track)


def extract_one(r):
    from attach import fifty, save_face, window_frame
    from reviewed_recordings import rest_undetected_hands
    from track import track as run_track
    import tid_youtube as ty
    vid = r["id"]
    data, meta = run_track(video_path(vid))   # in memory: the full track is never kept
    t = data["time"]
    raw50 = rest_undetected_hands(fifty(data))
    a, b, info, reason = select(data, raw50)
    if reason is None:
        reason = qa_span(raw50, data, meta, a, b)
    span = None if a is None else [round(float(t[a]), 3), round(float(t[b - 1]), 3)]
    base = {"id": vid, "gloss": r["gloss"], "variant": int(r["variant"]), "sense": r["sense"], "title": r["title"],
            "span_s": span, **info}
    if reason is not None:
        (OUT / "signs" / f"{vid}.failed").write_text(json.dumps({**base, "reason": reason}, ensure_ascii=False),
                                                       encoding="utf-8")
        return "rejected"
    clip, _, _ = window_frame(raw50[a:b])
    clip = clip.astype(np.float32)
    sid = f"{PREFIX}_{vid}"
    row = {"sign_id": sid, "gloss": display_gloss(r["gloss"], r["sense"]), "dataset": DATASET, "video": video_path(vid).name,
           "start": span[0], "end": span[1], "frames": b - a, "match": info["agreement"], "found": True}
    save_face(FACE, sid, data, raw50, meta, a, b - a, info["agreement"], row, clip)
    ty.face_manifest_append(row)
    np.save(OUT / "signs" / f"{vid}.npy", clip)
    (OUT / "signs" / f"{vid}.json").write_text(json.dumps({**base, "frames": b - a, "face_detected": row["face_detected"]},
                                                          ensure_ascii=False), encoding="utf-8")
    return "ok"


def spawn_entries():
    """`entries` in the plain interpreter (zeyrek), in the background; a lock keeps one at a time."""
    lock = WORK / "entries.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < 600:
        return
    subprocess.Popen(["py", str(Path(__file__)), "entries"], cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "src"},
                     stdout=open(WORK / "logs" / "entries.log", "a"), stderr=subprocess.STDOUT)


def extract(k, n, every=5):
    """Worker k of n over the downloaded videos in queue order; loops while downloads continue."""
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    stats, since = Counter(), 0
    while True:
        dl_over = (WORK / "DOWNLOAD_DONE").exists() or (WORK / "DOWNLOAD_STOPPED").exists()
        rows = [r for i, r in enumerate(read_queue()) if i % n == k]
        todo = [r for r in rows if video_path(r["id"]) and not (OUT / "signs" / f"{r['id']}.npy").exists()
                and not (OUT / "signs" / f"{r['id']}.failed").exists()]
        for r in todo:
            try:
                s = extract_one(r)
            except Exception as e:  # a broken file: note it, go on
                s = "error"
                (OUT / "signs" / f"{r['id']}.failed").write_text(json.dumps(
                    {"id": r["id"], "gloss": r["gloss"], "reason": f"error: {e!r}"}, ensure_ascii=False), encoding="utf-8")
            stats[s] += 1
            print(r["id"], r["gloss"], s, dict(stats), flush=True)
            if s == "ok":
                since += 1
                if since >= every:
                    spawn_entries()
                    since = 0
        if not todo and dl_over:
            break
        time.sleep(60 if not todo else 1)
    spawn_entries()
    (WORK / f"EXTRACT_DONE_{k}").write_text(json.dumps(stats), encoding="utf-8")


# ---- entries -----------------------------------------------------------------------------------------------------
def write_entries():
    """data/lexicon_tr/cuneyt/entries.jsonl, written through a temporary file (os.replace)."""
    import tid_youtube as ty
    from isharati.text.turkish import tr_norm
    lock = WORK / "entries.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        lex = base_lexicon()
        mains, variants, taken = [], [], {}
        for r in read_queue():
            js = OUT / "signs" / f"{r['id']}.json"
            if not js.exists():
                continue
            info = json.loads(js.read_text(encoding="utf-8"))
            gloss = display_gloss(r["gloss"], r["sense"])
            key = tr_norm(gloss)
            row = {"gloss": gloss, "sign_id": f"{PREFIX}_{r['id']}", "dataset": DATASET,
                   "keypoints_path": f"cuneyt/signs/{r['id']}.npy", "review_status": "pending",
                   "is_religious": tr_norm(r["gloss"]) in ty.ISLAMIC, "is_letter": False,
                   "source": f"https://youtu.be/{r['id']}", "title": r["title"], "channel": CHANNEL,
                   "span_s": info["span_s"], "run_agreement": info.get("agreement"),
                   "run_separation": info.get("separation"), "runs_in_group": len(info.get("group", [])),
                   "variant": int(r["variant"])}
            owner = signed(lex, gloss)
            if owner is not None:
                variants.append({**row, "variant_of": owner.sign_id})
            elif key in taken:
                variants.append({**row, "variant_of": taken[key]})
            else:
                taken[key] = row["sign_id"]
                mains.append(row)
        path = OUT / "entries.jsonl"
        tmp = path.with_name(f"entries.{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in mains + variants), encoding="utf-8")
        for _ in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                time.sleep(0.2)
        print(f"{time.strftime('%H:%M')} {len(mains)} glosses + {len(variants)} variants -> {path}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


# ---- QA ----------------------------------------------------------------------------------------------------------
def qa(n_ok=25, n_bad=10, seed=0):
    import random
    import tid_youtube as ty
    out = WORK / "qa"
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    ok = sorted((OUT / "signs").glob("*.json"))
    bad = sorted((OUT / "signs").glob("*.failed"))
    for k, p in enumerate(rnd.sample(ok, min(n_ok, len(ok)))):
        i = json.loads(p.read_text(encoding="utf-8"))
        s, e = i["span_s"]
        ty.contact_sheet(video_path(p.stem), [s - 0.12, s, (s + e) / 2, e, e + 0.12],
                         ["before", "start", "middle", "end", "after"], out / f"accepted_{k:02d}_{p.stem}.jpg",
                         f"{i['gloss']} | {i['title']} | runs {i['runs_s']} group {i['group']} agr {i['agreement']}")
    reasons = Counter()
    for p in bad:
        reasons[re.sub(r"\s*\(.*|\d+%.*|[\d.]+ s.*|\d+ .*run.*|: .*", "", json.loads(p.read_text(encoding="utf-8"))["reason"])] += 1
    for k, p in enumerate(rnd.sample(bad, min(n_bad, len(bad)))):
        i = json.loads(p.read_text(encoding="utf-8"))
        ts = [a for a, _ in i.get("runs_s", [])][:5] or [5.0]
        if video_path(p.stem):
            ty.contact_sheet(video_path(p.stem), ts, [f"run{j}" for j in range(len(ts))],
                             out / f"rejected_{k:02d}_{p.stem}.jpg", f"{i['gloss']} | {i['reason']}")
    print(f"accepted {len(ok)}, rejected {len(bad)}: {dict(reasons)}")


# ---- supervisor --------------------------------------------------------------------------------------------------
def supervise(workers=2):
    """Keeps the download and `workers` extract processes running (restarts any that exit early); writes their PIDs to
    F:/cuneyt/pids.json. Stops when everything is done; F:/cuneyt/STOP stops it."""
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "src", "OPENBLAS_NUM_THREADS": "1"}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    me = str(Path(__file__))
    specs = {"download": (["py", me, "download"], lambda: (WORK / "DOWNLOAD_DONE").exists()
                          or (WORK / "DOWNLOAD_STOPPED").exists())}
    for k in range(workers):
        specs[f"extract{k}"] = ([str(MP_PYTHON), me, "extract", str(k), str(workers)],
                                lambda k=k: (WORK / f"EXTRACT_DONE_{k}").exists())
    procs, last = {}, {}
    while not (WORK / "STOP").exists():
        for name, (cmd, done) in specs.items():
            p = procs.get(name)
            if (p is None or p.poll() is not None) and not done():
                if p is not None and time.time() - last.get(name, 0) < (1800 if name == "download" else 120):
                    continue  # restart after a pause
                procs[name] = subprocess.Popen(cmd, cwd=str(ROOT), env=env, creationflags=flags,
                                               stdout=open(logs / f"{name}.log", "a"), stderr=subprocess.STDOUT)
                last[name] = time.time()
        (WORK / "pids.json").write_text(json.dumps({"supervisor": os.getpid(), **{n: p.pid for n, p in procs.items()
                                                                                   if p.poll() is None}}))
        if all(done() for _, done in specs.values()):
            break
        time.sleep(30)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "queue":
        queue()
    elif mode == "download":
        download()
    elif mode == "extract":
        extract(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "entries":
        write_entries()
    elif mode == "qa":
        qa()
    elif mode == "supervise":
        supervise()
