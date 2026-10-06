"""Face-pass plans (scripts/face/run_plan.py) for the Turkish and Indian one-sign-per-video sources:

  tid    TİD dictionary (scripts/lexicon/youtube_dictionaries.py, SIGN_LANG=tr): YouTube clips in
         gathring data/turkish_dictionaries/tid_sozluk/<id>.mp4                                    -> lexicon "tr"
  cislr  CISLR prototypes (scripts/lexicon/isl/cislr.py): clips inside CISLR_v1.5-a_videos.zip     -> lexicon "ur"
  wslp   WSLP word-recognition train set (scripts/lexicon/isl/wslp.py): clips inside train.zip      -> lexicon "ur"
         (CC BY-NC-ND 4.0: its faces stay private data, like its poses)

All three builders ran the old Holistic solution on the whole frame (no crop, crop-normalised = frame-normalised
coordinates) on the 25 fps grid, and kept the hand-visible span of each clip; the video holds one sign, so the plan
has no hint and run_plan.py searches the whole track. The zipped sources are extracted to data/face/videos/<source>/,
only the members the lexicon uses (WSLP is ~4.8 GB: --extract N limits it to the first N plan rows).

  PYTHONPATH=src py scripts/face/plans/tid_isl.py tid|cislr|wslp [--extract N]
Output: data/face/plans/<source>.jsonl
"""
import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402

TID_VIDEOS = Path(r"D:\islam\gathring data\turkish_dictionaries\tid_sozluk")
CISLR_ZIP = Path(r"D:\islam\CISLR\CISLR_v1.5-a_videos\CISLR_v1.5-a_videos.zip")
WSLP_ZIP = Path(r"D:\islam\WSLP\Shared_task_WR\Train-Dataset\train.zip")
VIDEOS = DATA / "face" / "videos"
PLANS = DATA / "face" / "plans"


def entries(path):
    """Lexicon rows, one per clip (TİD repeats a clip for each alias)."""
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return list({r["sign_id"]: r for r in rows}.values())


def row(e, clip, video, lexicon):
    return {"sign_id": e["sign_id"], "gloss": e["gloss"], "dataset": e["dataset"], "lexicon": lexicon,
            "clip": str(clip), "video": str(video), "crop": None, "frame_coords": False, "hint": None}


def plan_tid(_):
    root = DATA / "lexicon_tr"
    out = []
    for e in entries(root / "dictionaries" / "entries_tid_dictionary.jsonl"):
        vid = Path(e["keypoints_path"]).stem
        # the builder's lookup: <id>.*mp4, else <id>.webm
        video = next(TID_VIDEOS.glob(f"{vid}.*mp4"), None) or next(TID_VIDEOS.glob(f"{vid}.webm"), None)
        if video is None:
            print("no video for", e["sign_id"])
            continue
        out.append(row(e, root / e["keypoints_path"], video, "tr"))
    return out


def plan_zip(zip_path, source, extract):
    root = DATA / "lexicon_isl"
    z = zipfile.ZipFile(zip_path)
    # the builders' member lookup: stem -> name (the last one wins, as in wslp.members())
    names = {Path(n).stem: n for n in z.namelist() if n.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv"))}
    dst = VIDEOS / source
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for e in entries(root / source / "entries.jsonl"):
        uid = Path(e["keypoints_path"]).stem
        if uid not in names:
            print("not in the zip:", e["sign_id"])
            continue
        video = dst / Path(names[uid]).name
        if (extract is None or len(out) < extract) and not video.exists():
            with z.open(names[uid]) as src, open(video, "wb") as f:
                shutil.copyfileobj(src, f)
        # zip + member: stream_plan.py extracts the video when it gets to it and deletes it afterwards
        out.append({**row(e, root / e["keypoints_path"], video, "ur"), "zip": str(zip_path), "member": names[uid]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["tid", "cislr", "wslp"])
    ap.add_argument("--extract", type=int, default=0,
                    help="zipped sources: extract the videos of the first N rows now (default none: stream_plan.py "
                         "extracts each one when it needs it)")
    a = ap.parse_args()
    rows = {"tid": plan_tid, "cislr": lambda n: plan_zip(CISLR_ZIP, "cislr", n),
            "wslp": lambda n: plan_zip(WSLP_ZIP, "wslp", n)}[a.source](a.extract)
    PLANS.mkdir(parents=True, exist_ok=True)
    path = PLANS / f"{a.source}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    have = sum(Path(r["video"]).exists() for r in rows)
    print(f"{len(rows)} signs ({have} videos on disk) -> {path}")


if __name__ == "__main__":
    main()
