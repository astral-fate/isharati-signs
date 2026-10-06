"""Sign dictionaries published one word per video on YouTube -> lexicon entries, per language (SIGN_LANG=ar|tr).

Arabic:

  ArabicSignLanguage (the unified Arabic sign dictionary, ~2,700 clips of ~3 s, titled «وجبة meal»)
  DisabilityApps (the Kuwaiti dictionary and the unified, children's and scouts' dictionaries, ~4,000 clips,
  titled «القاموس الإشاري الكويتي - دكتور - طبيب»)

Each clip shows one sign, so the whole hand-visible span is the sign (isharati.lexicon.asl_citizen.extract_trimmed, as
for ASL Citizen). The gloss is the title's Arabic; alternatives after «-», «ـ», «/» or «،» become aliases. Every entry
is tagged with its dictionary; the demo's merge (merge_qa.py) prefers Saudi sources over these.

Turkish (SIGN_LANG=tr): the Türk İşaret Dili Sözlüğü channel (~5,000 clips of ~5 s, titled with the word, «KILIÇ
BALIĞI») and TiDiSLaM (Islamic terms, 11-40 s: the first signing stretch is kept).

  slow    one download process for all channels of the language, in small batches with pauses; it backs off for 30
          minutes when YouTube refuses a whole batch (it throttles after bursts), and stops after three refusals

  .venv/Scripts/python scripts/lexicon/youtube_dictionaries.py list              # manifests with glosses
  .venv/Scripts/python scripts/lexicon/youtube_dictionaries.py download <source>  # resumable, archive file
  .venv/Scripts/python scripts/lexicon/youtube_dictionaries.py extract <k> <n>    # worker k of n, loops while downloads run
  .venv/Scripts/python scripts/lexicon/youtube_dictionaries.py entries           # collect entries.jsonl
"""
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402

import os  # noqa: E402

YTDLP = Path(r"D:\islam\yt-dlp")
COOKIES = Path(r"D:\islam\cookies\cookies_2.txt")
LANG = os.environ.get("SIGN_LANG", "ar")
CONFIG = {
    "ar": {"videos": r"D:\islam\gathring data\arabic_dictionaries", "out": DATA / "lexicon_v2" / "arabic_dictionaries",
           "channels": {"arabicsignlanguage": "UCg85d6jX2GANVz1msIvFvnA", "disabilityapps": "UCiILcu4Zyk_j4L8tEe2UmVQ"},
           "prefix": "ad", "folder": "arabic_dictionaries"},
    "tr": {"videos": r"D:\islam\gathring data\turkish_dictionaries", "out": DATA / "lexicon_tr" / "dictionaries",
           "channels": {"tid_sozluk": "UCq2tlBndwWjJb2Zlf9QoOzQ", "tidislam": "UC7zHDrJRMxSd65cE3vxfksg"},
           "prefix": "tr", "folder": "dictionaries"},
}[LANG]
VIDEOS = Path(CONFIG["videos"])
OUT = CONFIG["out"]
CHANNELS = CONFIG["channels"]
SERIES = {"القاموس الإشاري الكويتي": "kuwaiti_dictionary", "القاموس الإشاري الموحد": "unified_dictionary",
          "القاموس الإشاري للأطفال": "children_dictionary", "القاموس الإشاري للكشافة": "scouts_dictionary"}
MAX_SECONDS = 15
SHARDS = 3  # download shards per channel
AR = re.compile(r"[\u0600-\u06FF]")


TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def glosses_tr(title, channel):
    """Turkish gloss and aliases from a title («KILIÇ BALIĞI», «MÜ'MİN»), Turkish casing, and its dictionary."""
    t = re.sub(r"\([^)]*\)|\[[^]]*\]|#\S+|Türk İşaret Dili Sözlüğü|işaret dili|İşaret Dili", " ", title, flags=re.I)
    names = [re.sub(r"\s+", " ", p).strip(" -–:.,/").translate(TR_LOWER).lower() for p in re.split(r"[/,;]| - ", t)]
    names = [n for n in names if re.search(r"[a-zçğıöşüâîû]", n) and len(n) <= 40]
    return list(dict.fromkeys(names)), ("tidislam" if channel == "tidislam" else "tid_dictionary")


def glosses(title, channel):
    """Arabic gloss and aliases from a title, and the dictionary it belongs to."""
    if LANG == "tr":
        return glosses_tr(title, channel)
    t = title.replace(".wmv", "").replace(".mp4", "")
    dataset = "unified_dictionary"
    for prefix, name in SERIES.items():
        if t.startswith(prefix):
            t, dataset = t[len(prefix):], name
    t = re.sub(r"لغة الإشارة|Arabic Sign Language|\bASL\b|#\S+", " ", t)
    t = re.sub(r"\([^)]*\)", " ", t)                        # notes such as «(زمن الرسول)»
    t = re.sub(r"^[\s\d\-–:.]+", " ", t)                    # «008- »
    t = re.sub(r"[A-Za-z][A-Za-z ,'’.&]*", " ", t)           # the English translation
    t = re.sub(r"\s+ص\s*$", " ", t)                         # «محمد ص» (ﷺ)
    names = [re.sub(r"\s+", " ", p).strip(" :-–ـ.،,/") for p in re.split(r"\s[-–]+\s|ـ|/|،|--|:", t)]
    names = [n for n in names if AR.search(n) and len(n) <= 40]
    return list(dict.fromkeys(names)), dataset


def ytdlp(*args, **kw):
    return subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--cookies", str(COOKIES),
                           "--js-runtimes", "node", "--remote-components", "ejs:github", *args],
                          cwd=YTDLP, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def list_all():
    OUT.mkdir(parents=True, exist_ok=True)
    for source, cid in CHANNELS.items():
        r = ytdlp("--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s", f"https://www.youtube.com/channel/{cid}/videos")
        rows = []
        for line in r.stdout.splitlines():
            vid, dur, title = (line.split("\t") + ["", ""])[:3]
            try:
                dur = float(dur)
            except ValueError:
                continue
            names, dataset = glosses(title, source)
            if names and dur <= (40 if source == "tidislam" else MAX_SECONDS):
                rows.append({"id": vid, "seconds": dur, "title": title, "gloss": names[0],
                             "aliases": "|".join(names[1:]), "dataset": dataset})
        with open(OUT / f"manifest_{source}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "seconds", "title", "gloss", "aliases", "dataset"])
            w.writeheader()
            w.writerows(rows)
        print(source, len(rows), "one-word clips")


def manifest():
    rows = []
    for source in CHANNELS:
        p = OUT / f"manifest_{source}.csv"
        if p.exists():
            rows += [dict(r, source=source) for r in csv.DictReader(open(p, encoding="utf-8"))]
    return rows


def slow(batch=10, pause=(3, 8), backoff=1800):
    """Every channel of the language, one process, small batches with pauses; backs off when YouTube refuses."""
    refusals = 0
    for source in CHANNELS:
        folder = VIDEOS / source
        folder.mkdir(parents=True, exist_ok=True)
        ids = [r["id"] for r in manifest() if r["source"] == source]
        gone_file = folder / "unavailable.txt"
        failed = set(gone_file.read_text(encoding="utf-8").split()) if gone_file.exists() else set()
        todo = [i for i in ids if i not in failed and not list(folder.glob(f"{i}.*mp4"))]
        print(source, len(todo), "to download", flush=True)
        k = 0
        while k < len(todo):
            chunk = todo[k:k + batch]
            urls = folder / "urls_slow.txt"
            urls.write_text("\n".join(f"https://youtu.be/{i}" for i in chunk), encoding="utf-8")
            r = ytdlp("-a", str(urls), "--ignore-errors", "-f", "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/b",
                      "--sleep-requests", "1", "--sleep-interval", str(pause[0]), "--max-sleep-interval", str(pause[1]),
                      "-o", str(folder / "%(id)s.%(ext)s"))
            got = {i for i in chunk if list(folder.glob(f"{i}.*mp4"))}
            if not got and "unavailable" in (r.stdout + r.stderr):
                refusals += 1
                if refusals >= 3:
                    print("YouTube keeps refusing: stopped; run slow again later", flush=True)
                    return
                print(f"batch refused ({refusals}/3): backing off {backoff // 60} min", flush=True)
                time.sleep(backoff)
                continue                                   # retry the same batch
            refusals = 0
            # a video that fails while its neighbours download is really gone: remember it, do not retry
            gone = [i for i in chunk if i not in got]
            if gone:
                with open(gone_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(gone) + "\n")
            k += batch
            print(f"{source}: {min(k, len(todo))}/{len(todo)}", flush=True)
        (folder / "DONE").write_text("done", encoding="utf-8")
    print("all downloads finished", flush=True)


def download(source, k=0, n=1):
    """Shard k of n (each shard has its own URL list and archive, so shards can run side by side)."""
    folder = VIDEOS / source
    folder.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for i, r in enumerate(x for x in manifest() if x["source"] == source) if i % n == k]
    ids = [i for i in ids if not list(folder.glob(f"{i}.*mp4"))]
    urls = folder / f"urls_{k}.txt"
    urls.write_text("\n".join(f"https://youtu.be/{i}" for i in ids), encoding="utf-8")
    r = ytdlp("-a", str(urls), "--download-archive", str(folder / f"archive_{k}.txt"), "--ignore-errors",
              "-f", "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/b", "--sleep-requests", "0.3",
              "-o", str(folder / "%(id)s.%(ext)s"))
    print(source, "download finished; errors:", r.stdout.count("ERROR") + r.stderr.count("ERROR"))
    (folder / f"DONE_{k}").write_text("done", encoding="utf-8")


MAX_MISSING = 0.45  # these clips are small (392x288): the hand detector drops blurred fast frames, which are filled in


def extract_trimmed(video_path, fps=25, pad=2):
    """Like isharati.lexicon.asl_citizen.extract_trimmed, tuned for small dictionary clips: lower detection and
    tracking confidence, and more missing hand frames allowed inside the signing. Returns (pose, missing share)."""
    import cv2
    import mediapipe as mp
    from isharati.pose.keypoints import LH, R_SH, RH, frame_from_holistic, interpolate_missing, normalize
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None, 1.0
    picks = np.linspace(0, len(frames) - 1, max(1, int(round(len(frames) * fps / src_fps)))).round().astype(int)
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                        min_detection_confidence=0.3, min_tracking_confidence=0.3) as holo:
        raw = np.stack([frame_from_holistic(holo.process(frames[i])) for i in picks])
    hand = ~(np.isnan(raw[:, LH, 0]).all(axis=1) & np.isnan(raw[:, RH, 0]).all(axis=1))
    if hand.sum() < 3 or np.isnan(raw[:, R_SH, 0]).all():
        return None, 1.0
    idx = np.where(hand)[0]
    a, b = max(0, idx[0] - pad), min(len(raw), idx[-1] + 1 + pad)
    if b - a > 4 * fps:  # a long clip (a term with its explanation): keep the first signing stretch only
        sys.path.insert(0, str(ROOT / "scripts" / "lexicon" / "asl"))
        from islamic_terms import signing_runs
        runs = [(s, e) for s, e in signing_runs(raw) if e - s >= 12]
        if runs:
            a, b = max(0, runs[0][0] - pad), min(len(raw), runs[0][1] + pad)
    filled, missing = interpolate_missing(raw[a:b])
    return (normalize(filled) if missing <= MAX_MISSING else None), float(missing)


def extract(k, n):
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    rows = [r for i, r in enumerate(manifest()) if i % n == k]
    while True:
        todo = 0
        for r in rows:
            npy, bad = OUT / "signs" / f"{r['id']}.npy", OUT / "signs" / f"{r['id']}.failed"
            if npy.exists() or bad.exists():
                continue
            video = next((VIDEOS / r["source"]).glob(f"{r['id']}.*mp4"), None) or next((VIDEOS / r["source"]).glob(f"{r['id']}.webm"), None)
            if video is None or ".part" in video.name:
                todo += 1
                continue
            try:
                pose, missing = extract_trimmed(video)
            except Exception as e:  # a broken file: skip it, keep going
                pose, missing = None, 1.0
                print(r["id"], "failed:", e, flush=True)
            if pose is None or len(pose) < 6:
                bad.write_text(f"no usable signing (missing hand {missing:.2f})", encoding="utf-8")
            else:
                np.save(npy, pose.astype(np.float32))
                npy.with_suffix(".json").write_text(json.dumps({"missing": round(missing, 3)}), encoding="utf-8")
        done = all((VIDEOS / s / "DONE").exists() or len(list((VIDEOS / s).glob("DONE_*"))) >= SHARDS for s in CHANNELS)
        if done and todo == 0:
            break
        time.sleep(30)
    print(f"worker {k}/{n} finished", flush=True)


def entries():
    out, n = [], 0
    for r in manifest():
        npy = OUT / "signs" / f"{r['id']}.npy"
        if not npy.exists():
            continue
        n += 1
        for g in [r["gloss"]] + [a for a in r["aliases"].split("|") if a]:
            out.append({"gloss": g, "sign_id": f"{CONFIG['prefix']}_{r['id']}", "dataset": r["dataset"],
                        "keypoints_path": f"{CONFIG['folder']}/signs/{r['id']}.npy", "review_status": "pending",
                        "is_religious": r["dataset"] == "tidislam", "is_letter": False})
    # one file per dictionary, so merge_qa.py can rank them
    by = {}
    for e in out:
        by.setdefault(e["dataset"], []).append(e)
    for dataset, rows in by.items():
        (OUT / f"entries_{dataset}.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in rows), encoding="utf-8")
    print(n, "signs;", {d: len(v) for d, v in by.items()}, "glosses")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "list":
        list_all()
    elif mode == "slow":
        slow()
    elif mode == "download":
        download(sys.argv[2], *(int(x) for x in sys.argv[3:5]))
    elif mode == "extract":
        extract(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "entries":
        entries()
