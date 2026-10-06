"""Data for the weakly supervised sign spotter (train.py): YouTube-ASL sentence clips labelled with the words of their
English caption, plus the lexicon's isolated signs as clean examples.

Each clip becomes per-frame features at ~15 fps (YouTube-ASL ~30 fps halved; lexicon 25 fps resampled): elbow, wrist
and hand shape of the dominant (more active) hand first, then the other, mirrored for left-handed signers
(scripts/mining/motion_spot.features).

  vocabulary  content words with >= MIN_CLIPS training clips; every uncovered Qur'an/hadith word
              (results/coverage_en.json) with >= MIN_CLIPS_MISSING clips is kept
  held out    EVAL_WORDS known lexicon words (the most frequent, as in motion_spot.py eval) get no lexicon
              examples, so they are learnt from captions only, like the missing words: the honest test
  split       by video: test = hash % 10 == 0, val = 1, train = the rest

  .venv/Scripts/python scripts/mining/mil_prepare.py
Output: data/youtube_asl/mil/{feats.npy (float16, all frames), meta.json}
"""
import csv
import glob
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from isharati.text.english import STOPWORDS, normalize_en  # noqa: E402
from isharati.pose.dtw import resample  # noqa: E402
from motion_spot import CAPTIONS, LEX, features  # noqa: E402

YT = DATA / "youtube_asl"
OUT = YT / "mil"
MIN_CLIPS, MIN_CLIPS_MISSING, MAX_VOCAB, EVAL_WORDS = 40, 15, 1500, 80
MAX_FRAMES = 300          # source frames; longer clips are skipped


def feats(p, step=2):
    act = [np.nanmean(np.abs(np.diff(p[:, w, :2], axis=0))) if np.isfinite(p[:, w, 0]).any() else 0 for w in (4, 7)]
    return features(p, mirror=act[1] > act[0])[::step].astype(np.float16)


def split_of(video):
    h = zlib.crc32(video.encode()) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def main():
    caps = json.loads(CAPTIONS.read_text(encoding="utf-8"))
    text = {c: v[c]["translation"] for v in caps.values() for c in v["clip_order"]}
    clips = []
    for man in sorted(glob.glob(str(YT / "manifest" / "raw_keypoints_*.csv"))):
        part = Path(man).stem
        for r in csv.DictReader(open(man, encoding="utf-8")):
            if r["clip_id"] in text and int(r["frames"]) <= MAX_FRAMES and \
                    max(float(r["left_hand"]), float(r["right_hand"])) >= 0.5:
                words = sorted({w for w in normalize_en(text[r["clip_id"]]).split() if w not in STOPWORDS and len(w) > 2})
                clips.append((part, r["clip_id"], r["video_id"], words))
    print(len(clips), "clips", flush=True)

    from isharati.lexicon.asl import ASLLexicon
    lex = ASLLexicon.load(LEX / "lexicon_all.jsonl")
    lex_words = {e.gloss.rstrip("0123456789").lower() for e in lex.sign_entries()}
    missing = [w for w, _ in json.loads((ROOT / "results" / "coverage_en.json").read_text(encoding="utf-8"))["top_missing"]]
    n = Counter(w for part, c, v, ws in clips if split_of(v) == "train" for w in ws)
    vocab = [w for w, k in n.most_common() if k >= MIN_CLIPS][:MAX_VOCAB]
    vocab += [w for w in missing if n[w] >= MIN_CLIPS_MISSING and w not in vocab]
    held = [w for w, k in n.most_common() if w in lex_words and len(w) > 3 and k >= 25][:EVAL_WORDS]
    vocab += [w for w in held if w not in vocab]
    index = {w: i for i, w in enumerate(vocab)}
    print(len(vocab), "words;", sum(w in index for w in missing), "missing words;", len(held), "held out", flush=True)

    arrays, items = [], []
    off = 0
    zs = {}
    for k, (part, c, v, ws) in enumerate(clips):
        if part not in zs:
            zs[part] = np.load(YT / "poses" / f"{part}.npz")
        f = feats(zs[part][c].astype(np.float32))
        arrays.append(f)
        items.append({"id": c, "video": v, "split": split_of(v), "start": off, "len": len(f),
                      "labels": [index[w] for w in ws if w in index], "kind": "yt"})
        off += len(f)
        if k % 5000 == 0:
            print(f"{k}/{len(clips)}", flush=True)
    # clean examples: lexicon signs of vocabulary words that are not held out (25 fps -> 15 fps)
    templates = ASLLexicon.load(LEX / "templates.jsonl") if (LEX / "templates.jsonl").exists() else None
    n_iso = 0
    for L in [lex] + ([templates] if templates else []):
        for e in L.sign_entries():
            g = e.gloss.rstrip("0123456789").lower()
            if g in index and g not in held:
                try:
                    p = L.keypoints(e).astype(np.float32)
                except Exception:
                    continue
                p = resample(p, max(6, round(len(p) * 15 / 25)))
                f = feats(p, step=1)
                arrays.append(f)
                items.append({"id": e.sign_id, "video": "lexicon", "split": "train", "start": off, "len": len(f),
                              "labels": [index[g]], "kind": "iso"})
                off += len(f)
                n_iso += 1
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "feats.npy", np.concatenate(arrays))
    meta = {"vocab": vocab, "held_out": held, "missing": [w for w in missing if w in index], "items": items,
            "fps": 15, "dim": int(arrays[0].shape[1])}
    (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    sp = Counter(i["split"] for i in items)
    print(f"{off} frames, {len(items)} items ({n_iso} lexicon signs), splits {dict(sp)} -> {OUT}")


if __name__ == "__main__":
    main()
