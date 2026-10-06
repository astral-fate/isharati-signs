"""Face pass for a whole source from a plan file: track every video (scripts/face/track.py, in parallel), then find
each sign's clip in its track and save its face (scripts/face/attach.py's matcher and file format).

A plan is JSON lines, one per sign, written by a source-specific plan builder (scripts/face/plans/*.py):
  {"sign_id", "lexicon": "asl"|"ar"|"tr"|"ur", "clip": <path of the stored [T,50,3] clip>,
   "video": <path>, "crop": [x0,y0,x1,y1] pixels or null, "frame_coords": false,
   "hint": <seconds where the clip starts, or null>, "slack": <frames searched around the hint, default 75>,
   "frames": [i0, i1] or null}   (frames: the clip is exactly these frames of the 25 fps track; checked, not searched)
   optional "candidate_group": true  several rows share the sign_id, one per video the clip may come from (the
   builder kept one and did not record which); only the best-matching row is saved and reported
   optional "fps": <grid rate>  when the clip is not on the 25 fps grid (new_arabic_sources: every source frame,
   "fps" = the video's own rate); "frames" then index that grid
   optional "until": <seconds>  track only the start of the video (the builder never looked further)
crop / frame_coords must reproduce the builder's MediaPipe input, or the 50 points will not line up with the clip.
Tracks are cached by video + crop in data/face/tracks/plan/, so a re-run only does what is missing.

  PYTHONPATH=src python scripts/face/run_plan.py PLAN.jsonl [--workers 4] [--limit N] [--attach-only]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "face"))
import attach as A  # noqa: E402
from isharati.config import DATA  # noqa: E402

MP_PYTHON = r"D:\islam\gathring data\.venv\Scripts\python.exe"  # the venv with the MediaPipe Holistic task
# where tracks are cached: ISHARATI_FACE_TRACKS moves them off a full drive (e.g. F:/isharati_face/tracks), and
# stream_plan.py --keep-tracks keeps them there for later re-extraction instead of deleting each after use
TRACKS = Path(os.environ.get("ISHARATI_FACE_TRACKS") or DATA / "face" / "tracks" / "plan")


def track_path(row):
    key = f"{row['video']}|{row.get('crop')}|{row.get('frame_coords', False)}"
    if row.get("fps"):  # a non-default grid is its own track
        key += f"|fps={row['fps']}"
    if row.get("until"):
        key += f"|until={row['until']}"
    return TRACKS / f"{Path(row['video']).stem[:40]}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.npz"


def run_track(row):
    out = track_path(row)
    if out.exists():
        return None
    cmd = [MP_PYTHON, str(ROOT / "scripts" / "face" / "track.py"), row["video"], str(out)]
    if row.get("crop"):
        cmd += ["--crop", ",".join(str(int(v)) for v in row["crop"])]
    if row.get("frame_coords"):
        cmd.append("--frame-coords")
    if row.get("fps"):
        cmd += ["--fps", str(row["fps"])]
    if row.get("until"):
        cmd += ["--until", str(row["until"])]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return None if out.exists() else f"{row['video']}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}"


def attach_row(row, cache, best):
    """best: sign_id -> lowest match saved so far, so a candidate group's face file ends up from its closest row."""
    clip = np.load(row["clip"])
    tp = track_path(row)
    if tp not in cache:
        cache.clear()  # one video's track in memory at a time (rows are grouped by video)
        z = np.load(tp)
        tr = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
        cache[tp] = (tr, A.fifty(tr), json.loads(str(z["meta"])))
    tr, raw50, meta = cache[tp]
    t, L = tr["time"], len(clip)
    if len(raw50) < L:
        return {"sign_id": row["sign_id"], "found": False, "match": None, "reason": "track shorter than clip"}
    if row.get("frames"):
        s = int(row["frames"][0])
        win, _, _ = A.window_frame(raw50[s:s + L])
        d = A.distance(clip, win, ~np.isnan(raw50[s:s + L, :, 0])) if win is not None else np.inf
    else:
        h = row.get("hint")
        hint = None if h is None else int(np.argmin(np.abs(t - h)))
        s, d = A.locate(clip, raw50, hint, slack=row.get("slack", 75))
    out = {"sign_id": row["sign_id"], "gloss": row.get("gloss"), "dataset": row.get("dataset"),
           "video": Path(meta["video"]).name, "start": round(float(t[s]), 3), "end": round(float(t[s + L - 1]), 3),
           "frames": L, "match": None if not np.isfinite(d) else round(float(d), 3)}
    out["found"] = bool(np.isfinite(d) and d <= row.get("max_match", A.FOUND))
    if out["found"] and not (row.get("candidate_group") and best.get(row["sign_id"], np.inf) <= d):
        best[row["sign_id"]] = d
        out_dir = DATA / "face" / row["lexicon"]
        out_dir.mkdir(parents=True, exist_ok=True)
        A.save_face(out_dir, row["sign_id"], tr, raw50, meta, s, L, float(d), out, clip)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N signs (a trial run)")
    ap.add_argument("--attach-only", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.plan.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if Path(r["video"]).exists() and Path(r["clip"]).exists()]
    if a.limit:  # the first N sign_ids (all of a candidate group's rows)
        keep = set(list(dict.fromkeys(r["sign_id"] for r in rows))[: a.limit])
        rows = [r for r in rows if r["sign_id"] in keep]
    rows.sort(key=lambda r: str(track_path(r)))
    TRACKS.mkdir(parents=True, exist_ok=True)
    if not a.attach_only:
        todo = list({str(track_path(r)): r for r in rows}.values())
        print(f"{len(rows)} signs, {len(todo)} videos to track with {a.workers} workers", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            for k, err in enumerate(ex.map(run_track, todo), 1):
                if err:
                    print("TRACK FAILED", err, flush=True)
                if k % 50 == 0:
                    print(f"  tracked {k}/{len(todo)}", flush=True)
    results, cache, best = [], {}, {}
    for r in rows:
        if not track_path(r).exists():
            results.append({"sign_id": r["sign_id"], "found": False, "match": None, "reason": "no track"})
            continue
        try:
            results.append(attach_row(r, cache, best))
        except Exception as e:  # one bad clip or track must not stop a 4,000-sign run
            results.append({"sign_id": r["sign_id"], "found": False, "match": None, "reason": repr(e)[:200]})
    # a candidate group reports only its closest row (the one whose face was saved)
    pick = {}
    for k, (r, x) in enumerate(zip(rows, results)):
        key = r["sign_id"] if r.get("candidate_group") else ("row", k)
        d = x["match"] if x.get("match") is not None else np.inf
        if key not in pick or d < pick[key][0]:
            pick[key] = (d, k)
    if len(pick) < len(rows):
        keep = sorted(k for _, k in pick.values())
        rows, results = [rows[k] for k in keep], [results[k] for k in keep]
    report = a.plan.with_suffix(".results.jsonl")
    report.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    ok = [x for x in results if x["found"]]
    m = [x["match"] for x in ok]
    print(f"{len(ok)}/{len(results)} signs got a face; median match {np.median(m) if m else float('nan'):.3f}; "
          f"details in {report}")
    # merge into the lexicon's manifest
    by_lex = {}
    for r, x in zip(rows, results):
        by_lex.setdefault(r["lexicon"], []).append({**x, "dataset": r.get("dataset")})
    for lex, xs in by_lex.items():
        path = DATA / "face" / lex / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        old = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
        ids = {x["sign_id"] for x in xs}
        path.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in [o for o in old if o["sign_id"] not in ids] + xs)
                        + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
