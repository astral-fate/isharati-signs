"""Face-pass plans (scripts/face/run_plan.py) for the ASL lexicon's video sources.

All three builders ran isharati.lexicon.asl_citizen.extract_trimmed on the whole, uncropped video: MediaPipe on the
25 fps grid, coordinates relative to the frame, then trimmed to the span where a hand is visible (+2 frames). So every
row is crop null, frame_coords false, and the trim start was not kept: the (short, one-sign) video is searched whole.

  obscure_asl  scripts/lexicon/asl/youtube_signs.py: data/asl/youtube/obscure_asl/<video_id>.mp4, sign obscure_asl_<id>
  wlasl        scripts/lexicon/asl/wlasl.py: D:/islam/datasets_extra/WLASL/videos/<video_id>.mp4, already cut to the
               WLASL frame range (the cut file is what MediaPipe saw), sign wlasl_<video_id>
  asl_citizen  isharati.lexicon.asl_citizen: sign asl_<i> (i = the gloss's index in sorted order) is the DTW medoid of
               up to 3 training clips (one per signer), and which one was not saved. Each candidate is a row with
               "candidate_group": true (the best-matching one per sign is kept) and "zip" / "member": the 46 GB zip
               does not fit on disk, so scripts/face/stream_plan.py pulls each video out just in time to
               data/face/videos/asl_citizen/ and deletes it after. In trials (31 signs spread over the range) the kept
               clip was always the first candidate, so the default is --candidates 1; the signs a run misses can be
               re-planned with --only MISSES.txt --candidates 3.

  PYTHONPATH=src py scripts/face/plans/asl.py obscure_asl|wlasl|asl_citizen [--candidates 1] [--only IDS.txt] [--out P]
  PYTHONPATH=src py scripts/face/stream_plan.py data/face/plans/asl_citizen.jsonl   (run_plan.py for the other two)
Output: data/face/plans/<source>.jsonl
"""
import argparse
import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402

LEX = DATA / "asl" / "lexicon"
PLANS = DATA / "face" / "plans"
WLASL_VIDEOS = Path(r"D:\islam\datasets_extra\WLASL\videos")
CITIZEN_ZIP = Path(r"D:\islam\asl\ASL_Citizen.zip")
CITIZEN_VIDEOS = DATA / "face" / "videos" / "asl_citizen"


def signs(dataset):
    """One lexicon row per sign_id (aliases share a clip), in lexicon order."""
    out = {}
    for line in open(LEX / "lexicon_all.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["dataset"] == dataset:
            out.setdefault(r["sign_id"], r)
    return out


def row(r, video, **extra):
    return {"sign_id": r["sign_id"], "gloss": r["gloss"], "dataset": r["dataset"], "lexicon": "asl",
            "clip": str(LEX / r["keypoints_path"]), "video": str(video), "crop": None, "frame_coords": False,
            "hint": None, "frames": None, **extra}


def obscure_asl():
    vids = DATA / "asl" / "youtube" / "obscure_asl"
    ok = {m["video_id"] for m in csv.DictReader(open(vids / "manifest.csv", encoding="utf-8")) if m["status"] == "ok"}
    out = []
    for sid, r in signs("obscure_asl").items():
        vid = sid.removeprefix("obscure_asl_")
        if vid not in ok:
            print("not in manifest:", sid)
        out.append(row(r, vids / f"{vid}.mp4"))
    return out


def wlasl():
    return [row(r, WLASL_VIDEOS / f"{sid.removeprefix('wlasl_')}.mp4") for sid, r in signs("wlasl").items()]


def asl_citizen(candidates=1, only=None):
    from isharati.lexicon.asl_citizen import member_index, read_splits
    with zipfile.ZipFile(CITIZEN_ZIP) as zf:
        rows, members = read_splits(zf), member_index(zf)
        by_gloss = defaultdict(list)
        for r in rows:
            if Path(r["file"]).name in members:
                by_gloss[r["gloss"]].append(r)
        ids = {f"asl_{i:04d}": g for i, g in enumerate(sorted(by_gloss))}
        out = []
        for sid, r in signs("asl_citizen").items():
            if only and sid not in only:
                continue
            g = ids[sid]
            assert g == r["gloss"], (sid, g, r["gloss"])
            # the builder's candidates: training clips, one per signer, the first 3 signers in split-file order
            train = [x for x in by_gloss[g] if x["split"] == "train"] or by_gloss[g]
            seen, picks = set(), []
            for x in train:
                if x["participant"] not in seen:
                    seen.add(x["participant"])
                    picks.append(members[Path(x["file"]).name])
                if len(picks) == 3:
                    break
            out += [row(r, CITIZEN_VIDEOS / Path(m).name, candidate_group=True, zip=str(CITIZEN_ZIP), member=m)
                    for m in picks[:candidates]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["obscure_asl", "wlasl", "asl_citizen"])
    ap.add_argument("--candidates", type=int, default=1, help="asl_citizen: candidates per sign, in the builder's order")
    ap.add_argument("--only", help="asl_citizen: a file of sign_ids to plan (one per line), e.g. the misses of a run")
    ap.add_argument("--out", type=Path, help="plan path (default data/face/plans/<source>.jsonl)")
    a = ap.parse_args()
    only = set(a.only and Path(a.only).read_text(encoding="utf-8").split() or [])
    out = asl_citizen(a.candidates, only) if a.source == "asl_citizen" else globals()[a.source]()
    PLANS.mkdir(parents=True, exist_ok=True)
    path = a.out or PLANS / f"{a.source}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    missing = sum(not Path(r["video"]).exists() and "zip" not in r for r in out)
    print(f"{a.source}: {len(out)} rows, {len({r['sign_id'] for r in out})} signs, {missing} videos not on disk -> {path}")


if __name__ == "__main__":
    main()
