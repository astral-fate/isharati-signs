"""A sentence-final Isharah sign, found with a verified template: رمضان from «انا حب شهر رمضان».

Isharah has no per-sign timing, and cutting at pauses or by CTC alignment is no better than random cuts
(isharah/report.md). A sign that always ends its sentence is easier: it lies between the last pause and the return
to rest. Its span in each sentence is the stretch in the final part that best matches the verified Tawasol clip
(subsequence DTW); the medoid across signers becomes the entry. Validation: the matched spans must be much closer to
each other, and to the template, than same-length spans taken from the sentence start («انا حب») are.

Isharah is CC BY-NC-ND: entries are credited to Isharah and used for non-commercial research.
Output: data/lexicon_v2/isharah_final/{signs/*.npy, entries.jsonl, report.md}
  .venv/Scripts/python scripts/lexicon/arsl/isharah/last_sign.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from gloss_clips import ISHARAH, PKL, hand_speed, to_contract  # noqa: E402

from isharati.text.arabic import normalize_ar  # noqa: E402
from isharati.pose.dtw import dtw_distance  # noqa: E402
from isharati.pose.proportions import fix  # noqa: E402

OUT = DATA / "lexicon_v2" / "isharah_final"
TARGETS = {"رمضان": DATA / "lexicon_v2" / "tawasol" / "signs" / "taw109_08.npy"}  # verified Tawasol cut


def load_glosses():
    g = {}
    for split in ("train.csv", "dev.csv"):
        for line in (ISHARAH / split).read_text(encoding="utf-8").splitlines()[1:]:
            if "|" in line:
                sid, x = line.split("|", 1)
                g[sid] = [t.replace("_", " ") for t in x.split()]
    return g


def best_final_span(p, template):
    """The span in the last half of the signing that best matches the template, by DTW."""
    s = hand_speed(p)
    active = np.where(s > 0.2 * s.max())[0]
    a, b = (int(active[0]), int(active[-1]) + 1) if len(active) else (0, len(p))
    L = len(template)
    best = None
    for end in range(max(a + 8, b - 6), min(len(p), b + 4) + 1):
        for length in range(max(8, int(L * 0.5)), int(L * 1.6) + 1, 2):
            start = end - length
            if start < a + (b - a) // 3:
                continue
            d = dtw_distance(p[start:end], template)
            if best is None or d < best[0]:
                best = (d, start, end)
    return best, (a, b)


def medoid(clips):
    D = np.array([[dtw_distance(x, y) if i != j else 0.0 for j, y in enumerate(clips)] for i, x in enumerate(clips)])
    k = int(D.sum(axis=1).argmin())
    return k, D


def main():
    glosses = load_glosses()
    wanted = {normalize_ar(w): w for w in TARGETS}
    sids = {sid: g for sid, g in glosses.items() if g and normalize_ar(g[-1]) in wanted}
    print(f"{len(sids)} sentences end with a target sign; loading {PKL.name}", flush=True)
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries, report = [], ["# Isharah sentence-final signs\n",
                           "Matched = span found by DTW to the verified Tawasol clip; control = a span of the same "
                           "length from the start of the same sentence. Lower DTW = more similar.\n",
                           "| sign | sentences | signers | matched vs template | control vs template | "
                           "matched vs each other | control vs each other |", "|---|---|---|---|---|---|---|"]
    for key, word in wanted.items():
        template = fix(np.load(TARGETS[word]))  # one body shape: Isharah's pixel data is squashed flat
        spans, ctrl, signers = [], [], set()
        for sid, g in sids.items():
            if normalize_ar(g[-1]) != key or sid not in data:
                continue
            try:
                p = fix(to_contract(data[sid]["keypoints"]))
            except ValueError:
                continue
            found, (a, b) = best_final_span(p, template)
            if found is None:
                continue
            d, s, e = found
            spans.append(p[s:e])
            ctrl.append(p[a:a + (e - s)])
            signers.add(sid.split("_")[0])
        if len(spans) < 3:
            print(f"{word}: only {len(spans)} usable sentences, skipped")
            continue
        k, D = medoid(spans)
        _, C = medoid(ctrl)
        off = ~np.eye(len(spans), dtype=bool)
        row = (word, len(spans), len(signers), np.mean([dtw_distance(x, template) for x in spans]),
               np.mean([dtw_distance(x, template) for x in ctrl]), D[off].mean(), C[off].mean())
        report.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(*row))
        sid = f"ishf_{len(entries):03d}"
        np.save(OUT / "signs" / f"{sid}.npy", spans[k])
        entries.append({"gloss": word, "sign_id": sid, "dataset": "isharah", "keypoints_path":
                        f"isharah_final/signs/{sid}.npy", "review_status": "pending", "is_religious": True,
                        "is_letter": False})
        print(report[-1])
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
