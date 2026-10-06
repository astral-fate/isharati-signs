"""ASL Citizen (zip) -> data/asl/lexicon/{lexicon.jsonl, templates.jsonl, signs/, templates/} in the Isharati pose contract.

For every gloss, up to --clips training clips from different Deaf signers are read straight from the zip (no full
extraction), converted with isharati.pose.keypoints.extract (MediaPipe Holistic -> [T,50,3], 25 fps), and the DTW medoid
of the clean ones becomes the canonical sign. One clip from another signer is kept as the back-translation template.
Single-letter glosses (A-Z) become letter signs for fingerspelling. Resumable: signs already built are skipped.

  python -m isharati.lexicon.asl_citizen --zip D:/islam/asl/ASL_Citizen.zip [--clips 3] [--workers 4] [--limit N]
"""
import argparse
import csv
import io
import json
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from isharati.config import DATA

OUT = DATA / "asl" / "lexicon"
MAX_MISSING = 0.3
# signs a sharia/Deaf reviewer should check before religious content uses them
RELIGIOUS = {"god", "pray", "prayer", "church", "bible", "jesus", "heaven", "hell", "angel", "worship", "faith",
             "believe", "religion", "holy", "bless", "sin", "soul", "spirit", "fast", "charity", "mosque", "prophet"}


def read_splits(zf):
    """Rows of the split CSVs inside the zip: {split, participant, file, gloss}."""
    rows = []
    for name in zf.namelist():
        low = name.lower()
        if low.endswith(".csv") and any(s in low for s in ("train", "val", "test")):
            split = "train" if "train" in low else "val" if "val" in low else "test"
            with zf.open(name) as f:
                for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                    keys = {k.lower().strip(): v for k, v in r.items()}
                    file = keys.get("video file") or keys.get("video_file") or keys.get("filename")
                    gloss = keys.get("gloss") or keys.get("label")
                    part = keys.get("participant id") or keys.get("participant_id") or keys.get("user", "")
                    if file and gloss:
                        rows.append({"split": split, "participant": part, "file": file, "gloss": gloss.strip()})
    if not rows:
        raise SystemExit("no split CSVs with video file + gloss columns found in the zip")
    return rows


def member_index(zf):
    return {Path(n).name: n for n in zf.namelist() if n.lower().endswith((".mp4", ".mov", ".webm"))}


def extract_trimmed(video_path, fps=25, pad=2):
    """MediaPipe Holistic -> [T,50,3] like isharati.pose.keypoints.extract, but trimmed to the stretch where a hand is
    visible. ASL Citizen clips are webcam recordings that start and end with the hands down, out of frame; the
    missing-hand ratio is measured inside the signing, not over the resting ends."""
    import cv2
    from isharati.pose.keypoints import LH, R_SH, RH, Holistic, frame_from_holistic, interpolate_missing, normalize

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
        return None
    picks = np.linspace(0, len(frames) - 1, max(1, int(round(len(frames) * fps / src_fps)))).round().astype(int)
    with Holistic() as holo:
        raw = np.stack([frame_from_holistic(holo.process(frames[i], i * 1000 / src_fps)) for i in picks])
    hand = ~(np.isnan(raw[:, LH, 0]).all(axis=1) & np.isnan(raw[:, RH, 0]).all(axis=1))
    if hand.sum() < 3 or np.isnan(raw[:, R_SH, 0]).all():
        return None
    idx = np.where(hand)[0]
    a, b = max(0, idx[0] - pad), min(len(raw), idx[-1] + 1 + pad)
    filled, missing = interpolate_missing(raw[a:b])
    if missing > MAX_MISSING:
        return None
    return normalize(filled)


def extract_clip(zip_path, member):
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / Path(member).name
        dst.write_bytes(zf.read(member))
        return extract_trimmed(dst)


def build_sign(zip_path, gloss, members, template_member, sign_id):
    from isharati.lexicon.medoid import medoid
    clips = [c for c in (extract_clip(zip_path, m) for m in members) if c is not None]
    if not clips:
        return gloss, None
    canon = clips[medoid(clips)]
    np.save(OUT / "signs" / f"{sign_id}.npy", canon)
    tpl = extract_clip(zip_path, template_member) if template_member else None
    if tpl is not None:
        np.save(OUT / "templates" / f"{sign_id}.npy", tpl)
    return gloss, (len(clips), tpl is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path(r"D:\islam\asl\ASL_Citizen.zip"))
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N glosses (quick test)")
    ap.add_argument("--priority-text", type=Path, default=DATA / "asl" / "corpus.jsonl",
                    help="build signs in order of how often the approved texts use them")
    args = ap.parse_args()
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    (OUT / "templates").mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        rows, members = read_splits(zf), member_index(zf)
    by_gloss = defaultdict(list)
    for r in rows:
        if Path(r["file"]).name in members:
            by_gloss[r["gloss"]].append(r)
    glosses = sorted(by_gloss)
    if args.priority_text and args.priority_text.exists():
        # build first the signs the approved texts actually use, so a partial lexicon is already useful
        from collections import Counter
        from isharati.text.english import candidates, normalize_en
        from isharati.lexicon.asl import gloss_key
        freq = Counter()
        for line in args.priority_text.read_text(encoding="utf-8").splitlines():
            for w in normalize_en(json.loads(line).get("text", "")).split():
                for c in candidates(w):
                    freq[c] += 1
        glosses.sort(key=lambda g: -freq.get(gloss_key(g), 0))
    glosses = glosses[: args.limit] if args.limit else glosses
    print(f"{len(rows)} clips, {len(by_gloss)} glosses; building {len(glosses)}", flush=True)

    ids = {g: f"asl_{i:04d}" for i, g in enumerate(sorted(by_gloss))}
    jobs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for g in glosses:
            sid = ids[g]
            if (OUT / "signs" / f"{sid}.npy").exists():
                continue
            train = [r for r in by_gloss[g] if r["split"] == "train"] or by_gloss[g]
            seen, picks = set(), []
            for r in train:  # one clip per signer, up to --clips signers
                if r["participant"] not in seen:
                    seen.add(r["participant"])
                    picks.append(members[Path(r["file"]).name])
                if len(picks) == args.clips:
                    break
            held = next((members[Path(r["file"]).name] for r in by_gloss[g]
                         if r["split"] != "train" and r["participant"] not in seen), None)
            jobs[pool.submit(build_sign, str(args.zip), g, picks, held, sid)] = g
        done = 0
        for fut in as_completed(jobs):
            done += 1
            g, res = fut.result()
            if done % 100 == 0 or res is None:
                print(f"  {done}/{len(jobs)} {g}: {res}", flush=True)

    write_index(by_gloss, ids)


def write_index(by_gloss=None, ids=None, zip_path=Path(r"D:\islam\asl\ASL_Citizen.zip")):
    """lexicon.jsonl + templates.jsonl from the signs built so far (safe to call while a build is running)."""
    if by_gloss is None:
        with zipfile.ZipFile(zip_path) as zf:
            rows = read_splits(zf)
        by_gloss = defaultdict(list)
        for r in rows:
            by_gloss[r["gloss"]].append(r)
        ids = {g: f"asl_{i:04d}" for i, g in enumerate(sorted(by_gloss))}
    lex, tpl = [], []
    for g in sorted(by_gloss):
        sid = ids[g]
        if not (OUT / "signs" / f"{sid}.npy").exists():
            continue
        key = g.lower().rstrip("0123456789").strip()
        base = {"gloss": g, "sign_id": sid, "dataset": "asl_citizen", "review_status": "approved",
                "is_religious": key in RELIGIOUS, "is_letter": len(key) == 1 and key.isalpha()}
        if base["is_religious"]:
            base["review_status"] = "pending"
        lex.append({**base, "keypoints_path": f"signs/{sid}.npy"})
        if (OUT / "templates" / f"{sid}.npy").exists():
            tpl.append({**base, "keypoints_path": f"templates/{sid}.npy"})
    (OUT / "lexicon.jsonl").write_text("\n".join(json.dumps(r) for r in lex), encoding="utf-8")
    (OUT / "templates.jsonl").write_text("\n".join(json.dumps(r) for r in tpl), encoding="utf-8")
    letters = sorted(r["gloss"].lower() for r in lex if r["is_letter"])
    print(f"lexicon: {len(lex)} signs ({sum(r['is_religious'] for r in lex)} religious, pending review); "
          f"templates: {len(tpl)}; letter signs: {''.join(letters) or 'none'}")


if __name__ == "__main__":
    main()
