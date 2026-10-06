"""Face-pass plans (scripts/face/run_plan.py) for the Arabic lexicon's (lexicon_v2/lexicon_qa.jsonl) video sources.

Every builder below ran MediaPipe on the whole, uncropped frame with coordinates relative to it, so every row is
crop null, frame_coords false. Where the builder cached its raw track, the clip's place in it is recovered exactly
(the stored clip is reproduced to float precision) and becomes the row's hint; otherwise the video is searched.

  dictionaries  scripts/lexicon/youtube_dictionaries.py (unified / kuwaiti / children / scouts dictionaries): one
                ~3 s video per sign, trimmed to the hand-visible span (not kept): the whole video is searched
  saudi         scripts/lexicon/arsl/saudi_dictionary.py (saudi_ / arabic_dictionary_explained): sign sd_<vid>_<n> is
                first_sign() after label span n of raw_<vid>.npz; recomputed and checked against the clip -> hint
  jordan        scripts/lexicon/arsl/jordan_shorts.py: the clip is a slice of normalize() of the whole raw_<vid>.npz
                (no times saved); found by exact comparison, raw index -> source frame -> hint
  nas           scripts/lexicon/arsl/ingest_extracted_sources.py over gathring data/process_on_d.py and
                stream_preprocess.py: the Holistic task on every source frame (not the 25 fps grid), first 90 s;
                the clip is frames [int(start_s*25), int(end_s*25)) of that. The copy whose .pose.npz reproduces the
                clip is the video; rows get "fps" = its own rate, exact "frames" and "until" (only the start is tracked). Where no pose file reproduces it
                (missing, corrupt or re-run), a readable copy is searched around start_s. Signs without a local or
                readable video are left out (counted).
  karsl         data/karsl_kp (scripts/lexicon/arsl/karsl_extract.py, JPG frame folders read at 30 fps): every
                lexicon clip equals one candidate .npy exactly, so one row per sign, frames [0, T). The folder is
                written from the zip as an mp4 at 30 fps into data/face/videos/karsl/ (--extract N: first N signs;
                ~85 KB a sign, ~42 MB for all; stream_plan.py deletes each after use). The identity is certain, so
                rows carry a loose max_match (a hand the old model missed inflates the distance).

  PYTHONPATH=src py scripts/face/plans/arabic.py dictionaries|saudi|jordan|nas|nas-extra|karsl [--extract N]
Output: data/face/plans/<dataset>.jsonl (dictionaries: one per dictionary; saudi: both explained datasets)
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.pose.keypoints import interpolate_missing, normalize  # noqa: E402

LEX = DATA / "lexicon_v2"
PLANS = DATA / "face" / "plans"
GATHER = Path(r"D:\islam\gathring data")
DICT_VIDEOS = GATHER / "arabic_dictionaries"
SAUDI_VIDEOS = GATHER / "saudi_dictionary"
JORDAN_VIDEOS = GATHER / "jordan_shorts"
NAS_ROOTS = [(Path(r"D:\processed_signs\raw_mp4"), Path(r"D:\processed_signs\extracted")),  # (videos, pose files)
             (Path(r"E:\new_arabic_sources"), Path(r"E:\new_arabic_sources"))]
KARSL_ZIP = DATA / "kaggle" / "karsl-502" / "karsl-502.zip"
KARSL_VIDEOS = DATA / "face" / "videos" / "karsl"
KARSL_MAX_MATCH = 3.0
NAS_SECONDS = 92  # the builders tracked the first 90 s; a little more so a clip at the end fits
EXACT = 1e-4  # a recomputed clip this close is the stored one


def signs(*datasets):
    """One lexicon row per sign_id (aliases share a clip), in lexicon order."""
    out = {}
    for line in open(LEX / "lexicon_qa.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["dataset"] in datasets:
            out.setdefault(r["sign_id"], r)
    return out


def row(r, video, **extra):
    return {"sign_id": r["sign_id"], "gloss": r["gloss"], "dataset": r["dataset"], "lexicon": "ar",
            "clip": str((LEX / r["keypoints_path"]).resolve()), "video": str(video), "crop": None,
            "frame_coords": False, "hint": None, "frames": None, **extra}


def same(a, b):
    return a.shape == b.shape and float(np.nanmax(np.abs(a - b))) < EXACT


def clip_of(raw):
    """The builders' clip of a raw window (None when no shoulders are seen in it)."""
    try:
        return normalize(interpolate_missing(raw)[0])
    except ValueError:
        return None


def same_clip(raw, clip):
    c = clip_of(raw)
    return c is not None and same(c, clip)


def picks_of(video):
    """The builders' 25 fps grid: source frame of each raw index, and the source rate."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # a set in the builders: a frame picked twice (fps < 25) was run once
    return sorted(set(np.linspace(0, n - 1, int(round(n * 25 / fps))).round().astype(int).tolist())), fps


def dictionaries():
    """{dataset: rows} for the four dictionaries (one video per sign, searched whole)."""
    by_id = {}
    for source in ("arabicsignlanguage", "disabilityapps"):
        for m in csv.DictReader(open(LEX / "arabic_dictionaries" / f"manifest_{source}.csv", encoding="utf-8")):
            by_id.setdefault(m["id"], source)
    out = defaultdict(list)
    for sid, r in signs("unified_dictionary", "kuwaiti_dictionary", "children_dictionary", "scouts_dictionary").items():
        vid = sid.removeprefix("ad_")
        folder = DICT_VIDEOS / by_id.get(vid, "disabilityapps")
        video = next((p for p in folder.glob(f"{vid}.*") if p.suffix in (".mp4", ".webm") and ".part" not in p.name),
                     folder / f"{vid}.mp4")
        out[r["dataset"]].append(row(r, video))
    return out


def saudi():
    sys.path.insert(0, str(ROOT / "scripts" / "lexicon" / "arsl"))
    from saudi_dictionary import LAB, OUT, first_sign
    videos = {p.stem.split(" - ")[-1]: p for p in SAUDI_VIDEOS.glob("*/*.mp4") if ".f" not in p.stem and ".temp" not in p.stem}
    out, raws = [], {}
    for sid, r in signs("saudi_dictionary_explained", "arabic_dictionary_explained").items():
        vid, n = sid[3:].rsplit("_", 1)
        if vid not in raws:
            z = np.load(OUT / f"raw_{vid}.npz")
            raws[vid] = (z["raw"], z["t"], json.loads((LAB / f"{vid}.json").read_text(encoding="utf-8")))
        raw, t, spans = raws[vid]
        clip = np.load(LEX / r["keypoints_path"])
        a, b = spans[int(n)]
        s = None
        found = first_sign(raw, t, a - 0.5, b)  # the builder's own cut, recomputed
        if found and same_clip(raw[found[0]:found[1]], clip):
            s = found[0]
        else:  # labels re-run since the cut: search the span for the clip
            for k in range(max(0, int(np.searchsorted(t, a - 1))), min(len(raw) - len(clip), int(np.searchsorted(t, b)))):
                if same_clip(raw[k:k + len(clip)], clip):
                    s = k
                    break
        if s is None:
            print("not recovered, searched near its label:", sid)
        out.append(row(r, videos[vid], hint=round(float(t[s] if s is not None else a), 3), slack=3 if s is not None else 100))
    return out


def jordan():
    out = []
    videos = {p.stem.split(" - ")[-1]: p for p in JORDAN_VIDEOS.glob("*.mp4")}
    for sid, r in signs("jordan_shorts").items():
        vid = sid.removeprefix("jo_")
        clip = np.load(LEX / r["keypoints_path"])
        p = normalize(interpolate_missing(np.load(LEX / "jordan_shorts" / f"raw_{vid}.npz")["raw"])[0])
        L = len(clip)
        hits = [a for a in range(len(p) - L + 1) if same(p[a:a + L], clip)]
        picks, fps = picks_of(videos[vid])
        if not hits:
            print("not recovered, searched whole:", sid)
            out.append(row(r, videos[vid]))
            continue
        out.append(row(r, videos[vid], hint=round(picks[hits[0]] / fps, 3), slack=3))
    return out


def nas(skip=()):
    import cv2
    entries = {}
    for line in open(LEX / "new_arabic_sources" / "entries.jsonl", encoding="utf-8"):
        e = json.loads(line)
        entries.setdefault(e["sign_id"], e)
    copies = defaultdict(list)  # video file name -> [(video, pose file)]
    for vroot, proot in NAS_ROOTS:
        for v in vroot.rglob("*.mp4") if vroot.exists() else ():
            copies[v.name].append((v, (proot / v.relative_to(vroot)).with_name(f"{v.stem}.pose.npz")))
    poses, rates, out, skipped = {}, {}, [], defaultdict(int)
    for sid, r in signs("new_arabic_sources").items():
        if sid in skip:
            continue
        e = entries[sid]
        if not copies.get(e["source_video"]):
            skipped["no local video"] += 1
            continue
        clip = np.load(LEX / r["keypoints_path"])
        s0 = int(e["start_s"] * 25)  # a source-frame index (the builder sliced its every-frame track at 25/s)
        exact, readable = None, []
        for video, pose in copies[e["source_video"]]:
            if video not in rates:
                cap = cv2.VideoCapture(str(video))
                rates[video] = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            if not 5 < rates[video] < 121:  # a truncated download (no moov atom)
                continue
            readable.append(video)
            if pose not in poses:
                try:
                    poses[pose] = np.load(pose, allow_pickle=True)["pose"].astype(np.float32) if pose.exists() else None
                except Exception:  # a few pose files are corrupt
                    poses[pose] = None
            P = poses[pose]
            if exact is None and P is not None and same_clip(P[s0:s0 + len(clip)], clip):
                exact = video
        if exact is not None:
            out.append(row(r, exact, fps=rates[exact], until=NAS_SECONDS, frames=[s0, s0 + len(clip)]))
        elif readable:  # the pose file is gone, corrupt or re-run: search around the recorded start
            v = readable[0]
            out.append(row(r, v, fps=rates[v], until=NAS_SECONDS, hint=round(s0 / rates[v], 3), slack=50))
            skipped["kept, not verified (searched near start_s)"] += 1
        else:
            skipped["video unreadable"] += 1
    print("not exact / left out:", dict(skipped))
    return out


def karsl(extract=None):
    import zipfile
    import cv2
    cands = defaultdict(list)
    for m in csv.DictReader(open(DATA / "karsl_kp" / "manifest.csv", encoding="utf-8")):
        cands[m["sign_id"]].append(m)
    out, want = [], {}
    for sid, r in signs("karsl").items():
        clip = np.load(LEX / r["keypoints_path"])
        m = next((m for m in cands[sid] if same(np.load(DATA / "karsl_kp" / m["npy_path"]), clip)), None)
        if m is None:
            print("no candidate equals the clip:", sid)
            continue
        video = KARSL_VIDEOS / f"{m['clip_id']}.mp4"
        # the clip is this folder's whole extraction (identical .npy), so the match only checks; a hand the old model
        # never saw is a constant in the clip but tracked now, which inflates the distance (0010: 1.16, body 0.1)
        out.append(row(r, video, frames=[0, len(clip)], max_match=KARSL_MAX_MATCH))
        if not video.exists() and (extract is None or len(want) < extract):
            want[m["clip_id"]] = video
    if want:
        # karsl_extract's clip_id is the folder's relative path with \/. () replaced by _
        with zipfile.ZipFile(KARSL_ZIP) as zf:
            folders = defaultdict(list)
            for name in zf.namelist():
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    d, f = name.rsplit("/", 1)
                    cid = re.sub(r"[\\/. ()]", "_", d)
                    if cid in want:
                        folders[cid].append(name)
            KARSL_VIDEOS.mkdir(parents=True, exist_ok=True)
            num = lambda n: (int(re.findall(r"\d+", Path(n).stem)[-1]) if re.findall(r"\d+", Path(n).stem) else -1, Path(n).name)
            for cid, names in folders.items():
                imgs = [cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR) for n in sorted(names, key=num)]
                imgs = [i for i in imgs if i is not None]
                h, w = imgs[0].shape[:2]
                tmp = want[cid].with_suffix(".part.mp4")
                vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
                for i in imgs:
                    vw.write(i)
                vw.release()
                tmp.replace(want[cid])
        print(f"wrote {len(folders)} KArSL videos to {KARSL_VIDEOS}")
    return out


def write(name, rows):
    PLANS.mkdir(parents=True, exist_ok=True)
    path = PLANS / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    missing = sum(not Path(r["video"]).exists() for r in rows)
    print(f"{name}: {len(rows)} rows, {len({r['sign_id'] for r in rows})} signs, {missing} videos not on disk -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["dictionaries", "saudi", "jordan", "nas", "nas-extra", "karsl"])
    ap.add_argument("--extract", type=int, help="karsl: write only the first N signs' videos (a trial)")
    a = ap.parse_args()
    if a.source == "dictionaries":
        for dataset, rows in dictionaries().items():
            write(dataset, rows)
    elif a.source == "saudi":
        rows = saudi()
        for dataset in ("saudi_dictionary_explained", "arabic_dictionary_explained"):
            write(dataset, [r for r in rows if r["dataset"] == dataset])
    elif a.source == "nas":
        write("new_arabic_sources", nas())
    elif a.source == "nas-extra":  # the signs the first plan left out, now that more copies are on disk
        done = {json.loads(l)["sign_id"] for l in open(PLANS / "new_arabic_sources.jsonl", encoding="utf-8")}
        write("new_arabic_sources_extra", nas(skip=done))
    elif a.source == "jordan":
        write("jordan_shorts", jordan())
    else:
        write("karsl", karsl(a.extract))


if __name__ == "__main__":
    main()
