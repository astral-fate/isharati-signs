"""run_plan.py for sources too big for the disk: one video at a time, nothing kept but the per-sign face files.

Every drive here is nearly full, and a full run keeps ~11 GB of tracks plus the videos pulled out of zip archives.
This runs the same steps per video (extract it if the plan row has "zip" and "member", track it, match its signs, save
their faces) and then deletes the track and the temporary copy of any video it extracted from a zip (the archive is
untouched; source videos already on disk are only read, never deleted), with --workers videos in flight. Results are
appended to <plan>.partial.jsonl as each video finishes, so an interrupted run resumes where it stopped; at the end
they become <plan>.results.jsonl and the lexicon's face manifest, exactly as run_plan.py writes them.

  PYTHONPATH=src python scripts/face/stream_plan.py PLAN.jsonl [--workers 4] [--limit N] [--keep-tracks]
"""
import argparse
import json
import shutil
import sys
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_plan as R  # noqa: E402
from run_plan import DATA  # noqa: E402

LOCK = threading.Lock()  # attach_row shares `best` (candidate groups) and writes face files


def available(r):
    return Path(r["clip"]).exists() and (Path(r["video"]).exists() or ("zip" in r and "member" in r))


def do_video(rows, best, keep_tracks):
    first, video = rows[0], Path(rows[0]["video"])
    extracted = False
    if not video.exists():  # pull just this member out of its archive
        video.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(first["zip"]) as z, z.open(first["member"]) as src, open(video, "wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted = True
    err = R.run_track(first)
    results, cache = [], {}
    for r in rows:
        if not R.track_path(r).exists():
            results.append({"sign_id": r["sign_id"], "found": False, "match": None, "reason": f"no track {err or ''}"})
            continue
        try:
            with LOCK:
                results.append(R.attach_row(r, cache, best))
        except Exception as e:  # one bad clip or track must not stop a 4,000-sign run
            results.append({"sign_id": r["sign_id"], "found": False, "match": None, "reason": repr(e)[:200]})
    if not keep_tracks:
        R.track_path(first).unlink(missing_ok=True)
    if extracted:  # only the copy this run pulled out of its zip; the archive keeps the original. Videos that were
        video.unlink(missing_ok=True)  # already on disk are never deleted
    return [(r, x) for r, x in zip(rows, results)]


def finalize(plan, pairs):
    """run_plan.py's ending: a candidate group keeps its closest row; results file; lexicon manifests."""
    pick = {}
    for k, (r, x) in enumerate(pairs):
        key = r["sign_id"] if r.get("candidate_group") else ("row", k)
        d = x["match"] if x.get("match") is not None else np.inf
        if key not in pick or d < pick[key][0]:
            pick[key] = (d, k)
    pairs = [pairs[k] for k in sorted(k for _, k in pick.values())]
    results = [x for _, x in pairs]
    report = plan.with_suffix(".results.jsonl")
    report.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    ok = [x["match"] for x in results if x["found"]]
    print(f"{len(ok)}/{len(results)} signs got a face; median match {np.median(ok) if ok else float('nan'):.3f}; "
          f"details in {report}", flush=True)
    by_lex = defaultdict(list)
    for r, x in pairs:
        by_lex[r["lexicon"]].append({**x, "dataset": r.get("dataset")})
    for lex, xs in by_lex.items():
        path = DATA / "face" / lex / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        old = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
        ids = {x["sign_id"] for x in xs}
        path.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in [o for o in old if o["sign_id"] not in ids]
                                  + xs) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N signs (a trial run)")
    ap.add_argument("--keep-tracks", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.plan.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if available(r)]
    if a.limit:
        keep = set(list(dict.fromkeys(r["sign_id"] for r in rows))[: a.limit])
        rows = [r for r in rows if r["sign_id"] in keep]
    groups = defaultdict(list)
    for r in rows:
        groups[str(R.track_path(r))].append(r)
    partial = a.plan.with_suffix(".partial.jsonl")
    done, pairs = set(), []
    if partial.exists():  # resume: videos already finished
        for line in partial.read_text(encoding="utf-8").splitlines():
            p = json.loads(line)
            done.add(p["track"])
            pairs += [(r, x) for r, x in p["pairs"]]
    todo = [g for k, g in groups.items() if k not in done]
    best = {x["sign_id"]: x["match"] for r, x in pairs if x.get("found") and r.get("candidate_group")}
    print(f"{len(rows)} signs in {len(groups)} videos; {len(todo)} videos to do with {a.workers} workers", flush=True)
    R.TRACKS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex, open(partial, "a", encoding="utf-8") as log:
        futures = {ex.submit(do_video, g, best, a.keep_tracks): g for g in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            g = futures[fut]
            try:
                got = fut.result()
            except Exception as e:
                got = [(r, {"sign_id": r["sign_id"], "found": False, "match": None, "reason": repr(e)[:200]}) for r in g]
            pairs += got
            log.write(json.dumps({"track": str(R.track_path(g[0])), "pairs": got}, ensure_ascii=False) + "\n")
            log.flush()
            if n % 50 == 0:
                found = sum(1 for _, x in pairs if x.get("found"))
                print(f"  {n}/{len(todo)} videos, {found} faces so far", flush=True)
    finalize(a.plan, pairs)


if __name__ == "__main__":
    main()
