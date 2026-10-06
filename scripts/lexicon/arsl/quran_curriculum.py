"""Religious sign entries from the Quran curriculum videos, in the Isharati pose contract [T,50,3].

Input: D:/islam/gathring data/gloss_analysis/word_glosses.jsonl (every Quran word signed in the Al-Kharj
curriculum, with its lemma gloss and its frames) and each video's MediaPipe keypoints. In that series the
signer signs in sync with the recitation, so a word's frames are its sign.

Every lemma signed at least once becomes an entry: its DTW medoid when it was signed several times,
else its only clip. All entries are is_religious and review_status "pending" (Deaf + sharia review).

Outputs in data/lexicon_v2/religious/:
  signs/rel_NNNN.npy   the clip
  entries.jsonl        SignEntry rows (dataset "quran_curriculum")
  glosses.csv          gloss -> occurrences, example ayah, mean DTW spread
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.text.arabic import normalize_ar, strip_diacritics  # noqa: E402
from isharati.pose.dtw import dtw_distance  # noqa: E402
from isharati.pose.keypoints import interpolate_missing, normalize  # noqa: E402
from isharati.types import N_JOINTS  # noqa: E402

GATHER = Path(r"D:\islam\gathring data")
OUT = DATA / "lexicon_v2" / "religious"
# Isharati slot -> MediaPipe pose landmark (same mapping as isharati.pose.keypoints)
BODY = {0: 0, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15}
MAX_CANDIDATES = 12


def to_contract(kp, i0, i1):
    """Frames [i0, i1) of an extract_keypoints.py npz -> normalised [T,50,3]."""
    t = i1 - i0
    out = np.full((t, N_JOINTS, 3), np.nan, np.float32)
    pose = kp["pose"][i0:i1, :, :3]
    for slot, mp in BODY.items():
        out[:, slot] = pose[:, mp]
    out[:, 1] = (out[:, 2] + out[:, 5]) / 2
    out[:, 8:29] = kp["left_hand"][i0:i1]
    out[:, 29:50] = kp["right_hand"][i0:i1]
    filled, _ = interpolate_missing(out)
    return normalize(filled)


def main():
    occurrences = defaultdict(list)
    cache = {}
    for line in (GATHER / "gloss_analysis" / "word_glosses.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        folder = next((GATHER / "quran_sign_language_videos").glob(f"{row['surah']:03d} *"))
        if folder not in cache:
            cache[folder] = dict(np.load(folder / f"{folder.name}.keypoints.npz"))
        for w in row["words"]:
            if not w.get("frames") or w["frames"][1] - w["frames"][0] < 3:
                continue
            try:
                clip = to_contract(cache[folder], *w["frames"])
            except ValueError:
                continue  # no shoulders in these frames
            gloss = strip_diacritics(w["lemma"])
            occurrences[normalize_ar(gloss)].append(
                {"gloss": gloss, "clip": clip, "ref": f"{row['surah']}:{row['ayah']}", "word": w["word"]})

    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    entries, table = [], []
    for n, (key, occ) in enumerate(sorted(occurrences.items(), key=lambda kv: -len(kv[1]))):
        cand = occ[:MAX_CANDIDATES]
        spread = 0.0
        best = cand[0]
        if len(cand) > 2:
            d = np.zeros((len(cand), len(cand)))
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    d[i, j] = d[j, i] = dtw_distance(cand[i]["clip"], cand[j]["clip"])
            best = cand[int(d.sum(1).argmin())]
            spread = float(d[np.triu_indices(len(cand), 1)].mean())
        sign_id = f"rel_{n:04d}"
        np.save(OUT / "signs" / f"{sign_id}.npy", best["clip"])
        entries.append({"gloss": best["gloss"], "sign_id": sign_id, "dataset": "quran_curriculum",
                        "keypoints_path": f"signs/{sign_id}.npy", "review_status": "pending",
                        "is_religious": True, "is_letter": False})
        table.append({"gloss": key, "surface": best["gloss"], "sign_id": sign_id, "occurrences": len(occ),
                      "example": f"{best['ref']} {best['word']}", "mean_dtw": round(spread, 4)})
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    with open(OUT / "glosses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"{len(entries)} religious entries ({sum(1 for t in table if t['occurrences'] >= 3)} from 3+ occurrences)")


if __name__ == "__main__":
    main()
