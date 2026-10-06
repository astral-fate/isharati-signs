"""manifest.csv + labels.csv -> lexicon.jsonl (canonical clips from lexicon signers) and templates.jsonl (held-out signers)."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from isharati.pose.dtw import dtw_distance

# every character shape fingerspelling may need; missing shapes are reported, never silently tolerated
LETTER_SHAPES = "ابتثجحخدذرزسشصضطظعغفقكلمنهويةأؤئءإآى"


def medoid(clips: list[np.ndarray]) -> int:
    n = len(clips)
    if n <= 2:
        return 0
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = dtw_distance(clips[i], clips[j])
    return int(d.sum(1).argmin())


def _pick(rows, root, signers, max_missing, max_candidates):
    """Medoid of the clips within the quality threshold; if none pass, the least-missing clip (flagged low quality).
    Returns (clip, low_quality) or (None, False) when the signers have no usable clip at all."""
    usable = [r for r in rows if r["signer"] in signers and r["ok"] == "True"]
    if not usable:
        return None, False
    good = [r for r in usable if float(r["missing_ratio"]) <= max_missing][:max_candidates]
    if not good:
        best = min(usable, key=lambda r: float(r["missing_ratio"]))
        return np.load(root / best["npy_path"]), True
    clips = [np.load(root / r["npy_path"]) for r in good]
    return clips[medoid(clips)], False


def build(manifest_csv, labels_csv, out_dir, lexicon_signers: set[str], template_signers: set[str],
          max_missing: float = 0.3, max_candidates: int = 10) -> dict:
    manifest_csv, out_dir = Path(manifest_csv), Path(out_dir)
    root = manifest_csv.parent
    by_sign = defaultdict(list)
    with open(manifest_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_sign[r["sign_id"]].append(r)
    with open(labels_csv, encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    (out_dir / "signs").mkdir(parents=True, exist_ok=True)
    (out_dir / "templates").mkdir(parents=True, exist_ok=True)
    lex, tpl, skipped, low_quality = [], [], 0, 0
    for lab in labels:
        sid, gloss, chapter = lab["sign_id"], lab["gloss_ar"].strip(), lab["chapter"].strip().lower()
        if chapter == "letters" and len(gloss) > 2:
            raise ValueError(f"letter sign {sid} has gloss '{gloss}': letter glosses must be the character itself")
        canon, low = _pick(by_sign.get(sid, []), root, lexicon_signers, max_missing, max_candidates)
        if canon is None:
            skipped += 1
            continue
        low_quality += low
        religious = chapter == "religion"
        base = {"gloss": gloss, "sign_id": sid, "dataset": "karsl",
                # religious signs and low-quality fallback clips both need a human reviewer before publishing
                "review_status": "pending" if religious or low else "approved",
                "is_religious": religious, "is_letter": chapter == "letters"}
        np.save(out_dir / "signs" / f"{sid}.npy", canon)
        lex.append({**base, "keypoints_path": f"signs/{sid}.npy"})
        held, _ = _pick(by_sign.get(sid, []), root, template_signers, max_missing, max_candidates)
        if held is not None:
            np.save(out_dir / "templates" / f"{sid}.npy", held)
            tpl.append({**base, "keypoints_path": f"templates/{sid}.npy"})
    for name, rows in (("lexicon.jsonl", lex), ("templates.jsonl", tpl)):
        if rows:
            (out_dir / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    present = {r["gloss"] for r in lex if r["is_letter"]}
    return {"signs": len(lex), "skipped": skipped, "low_quality": low_quality, "religious": sum(r["is_religious"] for r in lex),
            "missing_letter_shapes": "".join(c for c in LETTER_SHAPES if c not in present)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon-signers", default="01,02")
    ap.add_argument("--template-signers", default="03")
    a = ap.parse_args()
    print(build(a.manifest, a.labels, a.out, set(a.lexicon_signers.split(",")), set(a.template_signers.split(","))))
