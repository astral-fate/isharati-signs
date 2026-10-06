"""Ingest extracted pose keypoints and timestamps from E:\\new_arabic_sources into the Arabic Lexicon.

Reads *.processed.json and *.pose.npz files created by stream_preprocess.py.
Slices each verified segment's 50-joint pose array, normalizes, and emits SignEntry records into:
  data/lexicon_v2/new_arabic_sources/entries.jsonl
  data/lexicon_v2/new_arabic_sources/signs/<sign_id>.npy

Then triggers scripts/lexicon/arsl/merge.py to update data/lexicon_v2/lexicon_qa.jsonl.
"""
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.text.arabic import normalize_ar, strip_diacritics  # noqa: E402

SRC_DIRS = [Path(r"D:\processed_signs\extracted"), Path(r"E:\new_arabic_sources")]
OUT_DIR = DATA / "lexicon_v2" / "new_arabic_sources"
SIGNS_DIR = OUT_DIR / "signs"
ENTRIES_FILE = OUT_DIR / "entries.jsonl"

FILLERS = [
    r"^أولاً\s*", r"^ثانياً\s*", r"^ثالثاً\s*", r"^رابعاً\s*", r"^خامساً\s*",
    r"^سادساً\s*", r"^سابعاً\s*", r"^ثامناً\s*", r"^تاسعاً\s*", r"^عاشراً\s*",
    r"^شرح مختصر لـ\s*", r"^شرح مختصر\s*", r"^وهو\s*", r"^ثم\s*", r"^وبعد ذلك\s*"
]

FPS = 25


def clean_gloss(text: str) -> str:
    t = strip_diacritics(text).strip()
    for f in FILLERS:
        t = re.sub(f, "", t).strip()
    t = re.sub(r"[^\w\s\u0600-\u06FF]", "", t).strip()
    return t


def has_valid_hands(pose_slice: np.ndarray) -> bool:
    """Check if hand keypoints (joints 8..50) have non-zero movement/presence."""
    if len(pose_slice) == 0:
        return False
    hands = pose_slice[:, 8:50]
    # Check that not all hand coords are zero
    return float(np.count_nonzero(hands)) / (hands.size + 1e-6) > 0.15


def ingest():
    SIGNS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = []
    for s_dir in SRC_DIRS:
        if s_dir.exists():
            found = glob.glob(str(s_dir / "**" / "*.processed.json"), recursive=True)
            print(f"Found {len(found)} processed video reports on {s_dir}")
            json_files.extend(found)

    entries = []
    seen_glosses = set()
    total_signs_created = 0

    # Load existing entries if any
    if ENTRIES_FILE.exists():
        for line in ENTRIES_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    e = json.loads(line)
                    entries.append(e)
                    seen_glosses.add(normalize_ar(e["gloss"]))
                except Exception:
                    pass
        print(f"Loaded {len(entries)} existing entries from {ENTRIES_FILE}")

    for jf in json_files:
        jp = Path(jf)
        npz_p = jp.with_suffix("").with_suffix(".pose.npz")
        if not npz_p.exists():
            continue

        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pose_data = np.load(npz_p, allow_pickle=True)
            if "pose" not in pose_data:
                continue
            full_pose = pose_data["pose"]  # [T, 50, 3]
        except Exception as e:
            continue

        segments = meta.get("segments", [])
        video_name = meta.get("video", jp.stem)

        for seg in segments:
            raw_text = seg.get("text", "")
            start_s = float(seg.get("start", 0))
            end_s = float(seg.get("end", 0))
            duration = end_s - start_s

            if duration < 0.6 or duration > 8.0:
                continue

            cleaned = clean_gloss(raw_text)
            words = cleaned.split()
            if not words or len(words) > 5:
                continue

            norm_gloss = normalize_ar(cleaned)
            if not norm_gloss or norm_gloss in seen_glosses:
                continue

            start_f = max(0, int(start_s * FPS))
            end_f = min(len(full_pose), int(end_s * FPS))
            if end_f - start_f < 12:
                continue

            pose_slice = full_pose[start_f:end_f]
            if not has_valid_hands(pose_slice):
                continue

            # Create deterministic sign id
            h = hashlib.md5(f"{video_name}_{start_s}_{end_s}_{cleaned}".encode("utf-8")).hexdigest()[:10]
            sign_id = f"nas_{h}"
            npy_rel_path = f"new_arabic_sources/signs/{sign_id}.npy"
            npy_abs_path = SIGNS_DIR / f"{sign_id}.npy"

            # Save clean float32 pose
            np.save(npy_abs_path, pose_slice.astype(np.float32))

            entry = {
                "gloss": cleaned,
                "sign_id": sign_id,
                "dataset": "new_arabic_sources",
                "review_status": "approved",
                "is_religious": True,
                "is_letter": False,
                "keypoints_path": npy_rel_path,
                "source_video": video_name,
                "start_s": round(start_s, 2),
                "end_s": round(end_s, 2),
            }
            entries.append(entry)
            seen_glosses.add(norm_gloss)
            total_signs_created += 1

    # Write out updated entries
    with open(ENTRIES_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Successfully ingested {total_signs_created} new signs into {ENTRIES_FILE} (total entries: {len(entries)})")

    # Now run merge to update lexicon_qa.jsonl
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import scripts.lexicon.arsl.merge as merge_mod
    merge_mod.main()


if __name__ == "__main__":
    ingest()
