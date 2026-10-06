"""Sign spotting in YouTube-ASL sentence clips: the sign for a word is the stretch of motion that recurs in the clips
whose English caption contains the word, and not in clips without it.

YouTube-ASL (Uthus et al., NeurIPS 2023) keypoints, converted by notebooks/youtube_asl_keypoints_colab.ipynb into the
Isharati format ([T,50,3], neck-centred, shoulder-scaled; data/youtube_asl from the HF dataset; CC BY 4.0).

For a word w:
  positives  up to MAX_POS clips whose caption contains w (fewest caption words first, a hand visible in >= half the frames)
  negatives  as many random clips whose caption does not contain w
  windows    every 16/24/32/44-frame stretch (step 4) with a hand raised, resampled to 12 frames; features: wrists and
             elbows relative to the neck, both hands' 21 points relative to their wrist
  score(u)   mean distance from u to its nearest window in each other positive clip (closest half of the clips, since
             a caption is a translation and the sign may be absent), divided by the same statistic over the negatives
The window with the lowest score is the candidate sign; its frames (source rate ~30 fps) are resampled to 25 fps.

  eval  words the lexicon already has: is the spotted sign nearer to the true lexicon sign than to the other signs?
  spot  the uncovered Qur'an/hadith words (results/coverage_en.json top_missing) -> data/asl/lexicon/youtube_asl/

  .venv/Scripts/python scripts/mining/motion_spot.py eval [n_words] | spot [n_words]
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.text.english import STOPWORDS, normalize_en  # noqa: E402
from isharati.pose.dtw import resample  # noqa: E402

YT = DATA / "youtube_asl"
CAPTIONS = Path(r"D:\islam\youtube-asl\YT.translations.all.json")
LEX = DATA / "asl" / "lexicon"
OUT = LEX / "youtube_asl"
LENGTHS, STEP, NORM_T = (16, 24, 32, 44), 4, 12
MAX_POS, MIN_POS, MAX_FRAMES = 60, 12, 240
SRC_FPS, FPS = 30, 25
RAISED = 1.2   # a wrist above this height (shoulder units below the neck) counts as signing


def features(p, mirror=False):
    """[T,50,3] -> [T,F]: the dominant (more active) arm and hand first, then the other: elbow and wrist relative to
    the neck, hand points relative to their wrist (hand scaled up, it is small); missing values -> 0. mirror=True
    swaps the sides and flips x, so a left-handed signer's sign matches a right-handed one."""
    p = np.nan_to_num(p[..., :2].astype(np.float32))
    R = (p[:, 3], p[:, 4], p[:, 29:50])
    L = (p[:, 6], p[:, 7], p[:, 8:29])
    a, b = (L, R) if mirror else (R, L)
    sx = np.array([-1, 1], np.float32) if mirror else np.ones(2, np.float32)
    parts = []
    for el, wr, hand in (a, b):
        parts += [el * sx, wr * sx, ((hand - hand[:, :1]) * sx * 3).reshape(len(p), -1)]
    return np.concatenate(parts, 1)


def windows(p):
    """All candidate windows of a clip: (vectors [N, NORM_T*F], spans [(start, end)]). The clip is read with its more
    active hand as the dominant one."""
    act = [np.nanmean(np.abs(np.diff(p[:, w, :2], axis=0))) if np.isfinite(p[:, w, 0]).any() else 0 for w in (4, 7)]
    f = features(p, mirror=act[1] > act[0])
    wy = np.nan_to_num(p[:, [4, 7], 1], nan=9.0).min(1)       # the higher wrist (y grows downwards)
    vecs, spans = [], []
    for L in LENGTHS:
        for s in range(0, len(p) - L + 1, STEP):
            if np.mean(wy[s:s + L] < RAISED) < 0.6:
                continue
            vecs.append(resample(f[s:s + L][:, :, None], NORM_T)[..., 0].ravel())
            spans.append((s, s + L))
    return (np.asarray(vecs, np.float32) if vecs else np.zeros((0, NORM_T * f.shape[1]), np.float32)), spans


def nearest(U, V):
    """For each row of U, the distance to its nearest row of V."""
    if len(V) == 0:
        return np.full(len(U), np.inf, np.float32)
    d = (U ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None] - 2 * U @ V.T
    return np.sqrt(np.maximum(d.min(1), 0))


class Data:
    def __init__(self):
        self.z = np.load(YT / "poses" / "raw_keypoints_1.npz")
        caps = json.loads(CAPTIONS.read_text(encoding="utf-8"))
        text = {c: v[c]["translation"] for v in caps.values() for c in v["clip_order"]}
        self.rows = {}
        for r in csv.DictReader(open(YT / "manifest" / "raw_keypoints_1.csv", encoding="utf-8")):
            if r["clip_id"] in text and int(r["frames"]) <= MAX_FRAMES and \
                    max(float(r["left_hand"]), float(r["right_hand"])) >= 0.5:
                self.rows[r["clip_id"]] = (int(r["frames"]), set(normalize_en(text[r["clip_id"]]).split()))
        self.by_word = defaultdict(list)
        for c, (n, words) in self.rows.items():
            for w in words:
                self.by_word[w].append(c)
        self.cache = {}

    def win(self, clip):
        if clip not in self.cache:
            self.cache[clip] = windows(self.z[clip].astype(np.float32))
        return self.cache[clip]

    def spot(self, word, rng):
        pos = sorted(self.by_word.get(word, []), key=lambda c: (len(self.rows[c][1]), self.rows[c][0]))[:MAX_POS]  # shortest captions
        if len(pos) < MIN_POS:
            return None
        others = [c for c in rng.sample(list(self.rows), min(len(self.rows), len(pos) * 3)) if word not in self.rows[c][1]]
        P = [self.win(c) for c in pos]
        N = [V for V in (self.win(c)[0] for c in others) if len(V)][:len(pos)]   # as many negatives as positives
        best = None
        for i, (U, spans) in enumerate(P):
            if len(U) == 0:
                continue
            d = np.stack([nearest(U, V) for j, (V, _) in enumerate(P) if j != i], 1)   # [windows, other positives]
            d.sort(1)
            pos_d = d[:, : max(3, d.shape[1] // 2)].mean(1)
            n = np.stack([nearest(U, V) for V in N], 1)                                   # the same statistic
            n.sort(1)
            neg_d = n[:, : max(3, n.shape[1] // 2)].mean(1)
            score = pos_d / np.maximum(neg_d, 1e-6)
            k = int(score.argmin())
            if best is None or score[k] < best[0]:
                best = (float(score[k]), pos[i], spans[k], float(pos_d[k]), float(neg_d[k]))
        if best is None:
            return None
        score, clip, (s, e), pd_, nd = best
        seg = self.z[clip][s:e].astype(np.float32)
        seg = resample(seg, max(6, round(len(seg) * FPS / SRC_FPS)))
        return {"word": word, "score": round(score, 4), "clip": clip, "start": s, "end": e, "n_pos": len(pos),
                "pos_dist": round(pd_, 3), "neg_dist": round(nd, 3), "pose": seg}


def sign_vec(p):
    """A whole sign as one vector, comparable with the spotted windows (same dominant-hand rule)."""
    act = [np.nanmean(np.abs(np.diff(p[:, w, :2], axis=0))) if np.isfinite(p[:, w, 0]).any() else 0 for w in (4, 7)]
    return resample(features(p, mirror=act[1] > act[0])[:, :, None], NORM_T)[..., 0].ravel()


def lexicon():
    from isharati.lexicon.asl import ASLLexicon
    lex = ASLLexicon.load(LEX / "lexicon_all.jsonl")
    glosses, vecs = [], []
    for e in lex.sign_entries():
        try:
            p = lex.keypoints(e)
        except Exception:
            continue
        glosses.append(e.gloss.rstrip("0123456789").lower())
        vecs.append(sign_vec(p))
    return glosses, np.asarray(vecs, np.float32)


def evaluate(n_words):
    """Known words: rank of the true lexicon sign among all lexicon signs, for the spotted window and for a random
    raised-hand window of the same clips (the chance baseline)."""
    rng = random.Random(0)
    d = Data()
    glosses, V = lexicon()
    gset = set(glosses)
    words = [w for w in sorted(d.by_word, key=lambda w: -len(d.by_word[w]))
             if w in gset and w not in STOPWORDS and len(w) > 3 and len(d.by_word[w]) >= 25][:n_words]
    ranks, base, per = [], [], []
    for w in words:
        r = d.spot(w, rng)
        if r is None:
            continue
        def rank(vec):
            dist = np.sqrt(((V - vec) ** 2).sum(1))
            true = min(dist[i] for i, g in enumerate(glosses) if g == w)
            return int((dist < true).sum()) + 1
        ranks.append(rank(sign_vec(r["pose"])))
        per.append({"word": w, "score": r["score"], "rank": ranks[-1]})
        c = rng.choice(d.by_word[w][:MAX_POS])
        U, spans = d.win(c)
        if len(U):
            s, e = spans[rng.randrange(len(spans))]
            base.append(rank(sign_vec(d.z[c][s:e].astype(np.float32))))
        print(f"{w:14s} pos {r['n_pos']:3d} score {r['score']:.3f} rank {ranks[-1]:5d} / {len(glosses)}", flush=True)
    ranks, base = np.array(ranks), np.array(base)
    res = {"words": len(ranks), "lexicon_signs": len(glosses),
           "spotted": {"top1": float((ranks <= 1).mean()), "top10": float((ranks <= 10).mean()),
                       "top50": float((ranks <= 50).mean()), "median_rank": float(np.median(ranks))},
           "random_window": {"top1": float((base <= 1).mean()), "top10": float((base <= 10).mean()),
                             "top50": float((base <= 50).mean()), "median_rank": float(np.median(base))},
           "per_word": per}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "motion_spot_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def spot(n_words):
    rng = random.Random(0)
    d = Data()
    missing = [w for w, _ in json.loads((ROOT / "results" / "coverage_en.json").read_text(encoding="utf-8"))["top_missing"]]
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    found = []
    for w in missing[:n_words]:
        r = d.spot(w, rng)
        if r is None:
            continue
        sid = f"yt_{w}"
        np.save(OUT / "signs" / f"{sid}.npy", r.pop("pose"))
        r["sign_id"] = sid
        found.append(r)
        print(f"{w:14s} pos {r['n_pos']:3d} score {r['score']:.3f}  {r['clip']} [{r['start']}:{r['end']}]", flush=True)
    (OUT / "candidates.json").write_text(json.dumps(found, indent=1), encoding="utf-8")
    print(len(found), "candidates ->", OUT)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    evaluate(n) if mode == "eval" else spot(n)
