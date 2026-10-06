"""Jordanian Sign Language signs spotted in continuous news videos (SignVerse-2M metadata list, CC-BY-NC-4.0).

The 132 YouTube videos (F:/signverse/jordanian_jos_video_ids.csv) are almost all one weekly news programme
(«أهلا بالخميس», Mind Rockets / Mimix) and a few short reports: continuous signing next to news pictures, with
Arabic captions that are whole sentences (one cue = 5-30 s). No video names its signs, so the signs are found by weakly
supervised cross-occurrence spotting: a word's sign is the segment that recurs inside every caption that has the word.

  prep   per video: the Holistic track (scripts/face/track.py, whole frame, 25 fps) -> the signer's frames only (the
         biggest person; frames where the tracker is on a news photo or a credits screen are dropped) -> sign
         segments from the pretrained RoPE Transformer of sign-language-processing/segmentation, run exactly as its
         own bin.run_inference (pose_format holistic pose, its normalisation), decoded so that a B frame starts a sign
         -> F:/signverse/jordanian/work/<id>.npz
  spot   words: captions normalised as the Arabic lexicon does (normalize_ar), stopwords dropped, keyed by CAMeL
         lemma (a private cache, the lexicon's lemma_cache.json is not touched); each caption's window
         [start + lag - 0.5 s, end + lag + 1 s] collects the segments inside it. For a word in >= 4 captions every
         candidate segment s is tested: tau(s) = the distance within which s finds a match in only 10% of random
         caption windows without the word, support(s) = the share of the word's other captions with a match within
         tau(s). The best s must reach support >= 0.6 and beat the 95th percentile of the same maximum over random
         pseudo-words with as many captions (the margin is reported). The clip is the medoid of s and its matches.
         The lag is the one that gives the most support over all words.
  validate  the same spotting on words that already have a sign (jordan_shorts first, the same dialect; then the
         rest of lexicon_qa): is the spotted segment closer to the known sign than the word's other candidates?
  build  signs/*.npy + entries.jsonl (new words; extra recordings of words the lexicon has become "variant_of"
         rows), faces in data/face/ar/ with manifest rows (locked append); only if validation passed.

Segment distance: a hands-dominant feature (the dominant hand's shape around its wrist, both wrists around the neck,
the other hand's shape at half weight; mirrored when the signer is left-dominant), resampled to 12 frames, Euclidean.

  .venv/Scripts/python scripts/lexicon/arsl/signverse_jordanian.py prep|spot|validate|build|qa
Output: data/lexicon_v2/signverse_jordanian/{signs/*.npy, entries.jsonl}, F:/signverse/jordanian/{work,qa,report}
"""
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "face"))
from isharati.config import DATA  # noqa: E402
from isharati.pose.keypoints import L_SH, L_WR, NECK, R_SH, R_WR, interpolate_missing, normalize  # noqa: E402
from isharati.text.arabic import normalize_ar, strip_diacritics  # noqa: E402
from isharati.text.arabic_morph import STOPWORDS  # noqa: E402

FPS = 25
BASE_F = Path(r"F:\signverse\jordanian")
VIDEOS, TRACKS, META = BASE_F / "videos", BASE_F / "tracks", BASE_F / "meta"
WORK, QA, REPORT = BASE_F / "work", BASE_F / "qa", BASE_F / "report"
LEMMA_CACHE = BASE_F / "lemma_cache.json"
IDS_CSV = Path(r"F:\signverse\jordanian_jos_video_ids.csv")
OUT = DATA / "lexicon_v2" / "signverse_jordanian"
FACE = DATA / "face" / "ar"
SEG_MODEL = Path(r"D:\islam\sign-language-segmentation\sign_language_segmentation\dist\2026")
NLP_PYTHON = Path(r"D:\islam\models\.venv-nlp\Scripts\python.exe")
DATASET = "signverse_jordanian"
L_RS = 12              # resampled length of a segment for the distance
MIN_OCC = 4            # captions a word must occur in
SUPPORT = 0.6          # share of a word's captions that must contain the sign
NEG_Q = 0.10           # tau: the distance at which a segment matches this share of random windows
MIN_LEN, MAX_LEN = 4, 62   # model spans of 0.16-2.5 s (padded by PAD frames each side)
PAD = 2


def video_ids():
    return [l.split(",")[0] for l in IDS_CSV.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]


# ---------------------------------------------------------------- prep: signer frames + model segmentation
def signer_mask(track):
    """Frames where the tracked person is the signer: shoulder width near the video's median (a news photo or the
    credits' portraits are much smaller), and no jump of the neck (the tracker switching to someone else)."""
    p = track["pose"]
    sw = np.hypot(p[:, 11, 0] - p[:, 12, 0], p[:, 11, 1] - p[:, 12, 1])
    med = np.nanmedian(sw)
    ok = np.isfinite(sw) & (sw > 0.6 * med) & (sw < 1.6 * med)
    neck = (p[:, 11, :2] + p[:, 12, :2]) / 2
    jump = np.r_[0, np.linalg.norm(np.diff(neck, axis=0), axis=1)]
    ok &= ~(jump > 0.5 * med)
    return ok


def model_segments(track, meta, ok):
    """Sign spans from the RoPE Transformer, its own preprocessing (bin.run_inference) on a pose_format pose built
    from the Holistic track (pixel units; the signer's frames only, the rest masked)."""
    import torch  # noqa: F401
    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody
    from pose_format.pose_header import PoseHeader, PoseHeaderDimensions
    from pose_format.utils.holistic import holistic_components
    from sign_language_segmentation.bin import load_model, run_inference
    w, h = meta["width"], meta["height"]
    comps = [c for c in holistic_components("XYZC", 10) if c.name != "POSE_WORLD_LANDMARKS"]
    header = PoseHeader(version=0.2, dimensions=PoseHeaderDimensions(width=w, height=h, depth=1000), components=comps)
    T = len(track["time"])
    face = track["face"].astype(np.float32)
    parts = [track["pose"][:, :, :3], face, track["left_hand"], track["right_hand"]]
    data = np.concatenate(parts, axis=1).astype(np.float32)
    data[~ok] = np.nan
    data = data * np.array([w, h, 1], np.float32)   # pixels; z as MediaPipe gives it (pose_format holistic files)
    miss = np.isnan(data)
    conf = (~miss.any(-1)).astype(np.float32)
    body = NumPyPoseBody(fps=FPS, data=np.ma.masked_array(np.nan_to_num(data)[:, None], mask=miss[:, None]),
                         confidence=conf[:, None])
    pose = Pose(header, body)
    model = load_model(model_dir=str(SEG_MODEL), device="cpu")
    lp = run_inference(model, pose, "cpu")
    sign = lp["sign"][0].cpu().numpy()            # (T, 4) log-probs: O, B, I, UNK
    sent = lp["sentence"][0].cpu().numpy()
    return decode(sign.argmax(1)), decode(sent.argmax(1)), sign


def decode(pred):
    """BIO -> spans [a, b); unlike the package's decoder a B frame always starts a new sign, so signs without a
    pause between them stay apart."""
    spans, a = [], None
    for i, p in enumerate(pred):
        if p == 2 or (p == 3 and a is None):     # package BIO: UNK 0, O 1, B 2, I 3
            if a is not None:
                spans.append((a, i))
            a = i
        elif p not in (2, 3) and a is not None:
            spans.append((a, i))
            a = None
    if a is not None:
        spans.append((a, len(pred)))
    return spans


def captions(vid):
    path = META / f"{vid}.ar.vtt"
    if not path.exists():
        return []
    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        m = re.search(r"(\d+):(\d+):([\d.]+)\s*-->\s*(\d+):(\d+):([\d.]+)", block)
        if not m:
            continue
        g = [float(x) for x in m.groups()]
        text = " ".join(block.split("\n")[1:]) if block.startswith(m.group(0)[:2]) else block[m.end():]
        text = block[m.end():].strip()
        cues.append((g[0] * 3600 + g[1] * 60 + g[2], g[3] * 3600 + g[4] * 60 + g[5], re.sub(r"\s+", " ", text)))
    return cues


def prep(vid):
    out = WORK / f"{vid}.npz"
    if out.exists():
        return "cached"
    tp = TRACKS / f"{vid}.npz"
    if not tp.exists():
        return "no track"
    from attach import fifty
    z = np.load(tp)
    meta = json.loads(str(z["meta"]))
    track = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    ok = signer_mask(track)
    raw = fifty(track)
    raw[~ok] = np.nan
    signs, sents, logp = model_segments(track, meta, ok)
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, raw=raw.astype(np.float32), ok=ok, signs=np.array(signs, int).reshape(-1, 2),
                        sents=np.array(sents, int).reshape(-1, 2), logp=logp.astype(np.float16),
                        aspect=meta["aspect"], meta=json.dumps(meta))
    os.replace(tmp, out)
    return f"{len(signs)} signs, signer in {ok.mean():.0%} of {len(ok)} frames"


# ---------------------------------------------------------------- segment features
def dominant(p):
    """True if the right hand (29:50) moves more than the left (8:29)."""
    def v(s):
        d = np.linalg.norm(np.diff(p[:, s, :2], axis=0), axis=-1)
        return float(np.nanmean(d)) if np.isfinite(d).any() else 0.0
    return v(slice(29, 50)) >= v(slice(8, 29))


def feature(p, right_dom=True):
    """[L_RS * D] hands-dominant feature of a normalised clip [T,50,3] (neck-centred, shoulder widths)."""
    x = p[..., :2].astype(np.float32)
    if not right_dom:  # mirror, so that the dominant hand always sits in 29:50
        x = x.copy()
        x[..., 0] *= -1
        x = x[:, list(range(8)) + list(range(29, 50)) + list(range(8, 29))]
        x[:, [R_SH, R_WR, L_SH, L_WR]] = x[:, [L_SH, L_WR, R_SH, R_WR]]
    dom, oth = x[:, 29:50], x[:, 8:29]
    f = np.concatenate([(dom - dom[:, :1]).reshape(len(x), -1) * 1.5,       # dominant hand shape
                        (oth - oth[:, :1]).reshape(len(x), -1) * 0.5,       # other hand shape
                        dom[:, 0] * 2.0, oth[:, 0] * 1.0], axis=1)          # wrists around the neck
    idx = np.linspace(0, len(f) - 1, L_RS)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(f) - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * f[lo] + w * f[hi]).astype(np.float32).ravel()


def clip_of(raw, a, b):
    filled, _ = interpolate_missing(raw[a:b])
    return normalize(filled)


def video_segments(vid):
    """The usable sign segments of one video: model spans of MIN_LEN-MAX_LEN frames, padded by PAD frames (and to
    at least 0.4 s), the signer tracked in >= 70% of their frames and the dominant hand in >= 70%
    (interpolate_missing's ratio). Returns padded spans, feature rows, model spans; cached beside the work file."""
    cache = WORK / f"{vid}.segs.npz"
    if cache.exists():
        z = np.load(cache)
        return [tuple(x) for x in z["spans"]], z["feats"], z["core"]
    z = np.load(WORK / f"{vid}.npz")
    raw, T = z["raw"], len(z["raw"])
    filled, _ = interpolate_missing(raw)
    try:
        rd = dominant(normalize(filled))
    except ValueError:
        return [], np.zeros((0, L_RS * 88), np.float32), np.zeros((0, 2), int)
    spans, feats, core = [], [], []
    for a, b in z["signs"]:
        if not (MIN_LEN <= b - a <= MAX_LEN):
            continue
        pa, pb = max(0, a - PAD), min(T, b + PAD)
        while pb - pa < 10 and (pa > 0 or pb < T):   # at least 0.4 s
            pa, pb = max(0, pa - 1), min(T, pb + 1)
        seg = raw[pa:pb]
        if np.isnan(seg[:, NECK, 0]).mean() > 0.3:
            continue
        _, missing = interpolate_missing(seg)
        if missing > 0.3:
            continue
        try:
            c = clip_of(raw, pa, pb)
        except ValueError:
            continue
        spans.append((int(pa), int(pb)))
        core.append((int(a), int(b)))
        feats.append(feature(c, rd))
    feats = np.stack(feats) if feats else np.zeros((0, L_RS * 88), np.float32)
    core = np.array(core, int).reshape(-1, 2)
    np.savez_compressed(cache, spans=np.array(spans, int).reshape(-1, 2), feats=feats, core=core)
    return spans, feats, core


# ---------------------------------------------------------------- words
def tokens(text):
    return [w for w in normalize_ar(text).split() if re.fullmatch(r"[ء-ي]+", w) and w not in STOPWORDS and len(w) >= 2]


def _cache():
    cache = {}
    shared = DATA / "lexicon_v2" / "lemma_cache.json"
    if shared.exists():  # read only: the lexicon's own cache is never written here
        cache.update(json.loads(shared.read_text(encoding="utf-8")))
    if LEMMA_CACHE.exists():
        cache.update(json.loads(LEMMA_CACHE.read_text(encoding="utf-8")))
    return cache


def private_lemmatize(words):
    """CAMeL lemmas for words not cached yet, saved in F:/signverse/jordanian/lemma_cache.json; also installed as
    isharati.text.arabic_morph's cache and lemmatizer, so the lexicon matcher never writes the shared cache."""
    from isharati.text import arabic_morph
    cache = arabic_morph._lemmas
    if not (isinstance(cache, dict) and cache.get("__sv__")):
        cache = _cache()
        cache["__sv__"] = True
        arabic_morph._lemmas = cache
        arabic_morph.lemmatize = private_lemmatize
    todo = sorted({w for w in words if w not in cache})
    if todo and NLP_PYTHON.exists():
        for i in range(0, len(todo), 2000):
            r = subprocess.run([str(NLP_PYTHON), str(ROOT / "scripts" / "lexicon" / "arsl" / "lemmatize_ar.py")],
                               input=json.dumps(todo[i:i + 2000], ensure_ascii=False), capture_output=True, text=True,
                               encoding="utf-8", timeout=900)
            try:
                cache.update(json.loads(r.stdout.strip().splitlines()[-1]))
            except (ValueError, IndexError):
                cache.update({w: [] for w in todo[i:i + 2000]})
                print("lemmatizer failed:", (r.stderr or "")[-300:])
        mine = json.loads(LEMMA_CACHE.read_text(encoding="utf-8")) if LEMMA_CACHE.exists() else {}
        mine.update({w: cache[w] for w in todo})
        LEMMA_CACHE.write_text(json.dumps(mine, ensure_ascii=False), encoding="utf-8")
    return cache


def _lemmas_of(w, cache):
    v = cache.get(w, [])
    return v.get("l", []) if isinstance(v, dict) else v


def lemma_key(w, cache):
    """A caption word's key: its first CAMeL lemma (normalised); a word the analyser does not know keeps its light
    stem (one of «وال», «بال», «فال», «لل», «ال», «و» off)."""
    ls = _lemmas_of(w, cache)
    key = normalize_ar(strip_diacritics(ls[0])) if ls else ""
    if not key:
        key = w
        for p in ("وال", "بال", "فال", "لل", "ال", "و"):
            if key.startswith(p) and len(key) - len(p) >= 3:
                key = key[len(p):]
                break
    return key


def lemma_gloss(w, cache):
    """The lemma with its letter shapes (diacritics off) as the gloss text."""
    ls = _lemmas_of(w, cache)
    return strip_diacritics(ls[0]) if ls else w


# ---------------------------------------------------------------- corpus of windows
class Corpus:
    """All tracked videos: segments (feature rows, spans) and caption windows with their word keys."""

    def __init__(self):
        self.seg_vid, self.seg_span, self.seg_core, feats = [], [], [], []
        self.vid_segs, self.cues = {}, {}
        for vid in video_ids():
            if not (WORK / f"{vid}.npz").exists():
                continue
            spans, f, core = video_segments(vid)
            self.vid_segs[vid] = list(range(len(self.seg_vid), len(self.seg_vid) + len(spans)))
            self.seg_vid += [vid] * len(spans)
            self.seg_span += [tuple(int(v) for v in s) for s in spans]
            self.seg_core += [tuple(int(v) for v in c) for c in core]
            if len(spans):
                feats.append(f)
            self.cues[vid] = captions(vid)
        self.F = np.concatenate(feats).astype(np.float32)
        self.sq = (self.F * self.F).sum(1)
        self.mid = np.array([(a + b) / 2 / FPS for a, b in self.seg_core])
        words = [w for c in self.cues.values() for _, _, t in c for w in tokens(t)]
        self.cache = private_lemmatize(words)
        self.key = {w: lemma_key(w, self.cache) for w in set(words)}
        self.surface = defaultdict(Counter)
        for w in words:
            self.surface[self.key[w]][w] += 1
        self.set_lag(1.0)

    def set_lag(self, lag):
        self.lag = lag
        self.windows = []     # (vid, cue index, text, segment ids, set of keys)
        for vid, cues in self.cues.items():
            ids = np.array(self.vid_segs[vid], int)
            if not len(ids):
                continue
            mid = self.mid[ids]
            for k, (s, e, t) in enumerate(cues):
                sel = ids[(mid >= s + lag - 0.5) & (mid <= e + lag + 1.0)]
                keys = {self.key[w] for w in tokens(t)}
                if len(sel) and keys:
                    self.windows.append((vid, k, t, sel, keys))
        self.by_key = defaultdict(list)
        for i, wdw in enumerate(self.windows):
            for kk in wdw[4]:
                self.by_key[kk].append(i)

    def dist(self, A, B):
        """RMS distance between segment feature rows A and B (index arrays)."""
        d = self.sq[A][:, None] + self.sq[B][None] - 2 * self.F[A] @ self.F[B].T
        return np.sqrt(np.maximum(d, 0) / self.F.shape[1])

    def dist_to(self, f, B):
        d = float((f * f).sum()) + self.sq[B] - 2 * self.F[B] @ f
        return np.sqrt(np.maximum(d, 0) / self.F.shape[1])


def window_min(D, owner, n):
    """M[s, k] = min over the segments of window k of D[s, :]."""
    M = np.full((D.shape[0], n), np.inf, np.float32)
    for k in range(n):
        cols = owner == k
        if cols.any():
            M[:, k] = D[:, cols].min(1)
    return M


def spot(C, win_ids, neg_ids):
    """Cross-occurrence spotting over the windows win_ids (see the module doc)."""
    segs = [C.windows[i][3] for i in win_ids]
    owner = np.concatenate([[k] * len(s) for k, s in enumerate(segs)])
    allp = np.concatenate(segs)
    negsegs = [C.windows[i][3] for i in neg_ids]
    negowner = np.concatenate([[k] * len(s) for k, s in enumerate(negsegs)])
    n, m = len(segs), len(negsegs)
    Dp = C.dist(allp, allp)
    Mp = window_min(Dp, owner, n)
    Mn = window_min(C.dist(allp, np.concatenate(negsegs)), negowner, m)
    tau = np.quantile(Mn, NEG_Q, axis=1)
    Mp[np.arange(len(allp)), owner] = np.inf          # a segment's own caption does not count
    # support counts videos, not captions: one story is often captioned twice in the same video
    vids = np.array([C.windows[i][0] for i in win_ids])
    uv, vidx = np.unique(vids, return_inverse=True)
    hit = Mp <= tau[:, None]
    hit_v = np.zeros((len(allp), len(uv)), bool)
    for k in range(n):
        hit_v[:, vidx[k]] |= hit[:, k]
    own_v = vidx[owner]
    # a video counts as "other" unless all its windows are the segment's own caption
    per_v = np.bincount(vidx, minlength=len(uv))
    others = np.full(len(allp), len(uv), float) - (per_v[own_v] == 1)
    support = hit_v.sum(1) / np.maximum(1, others)
    fin = np.where(np.isfinite(Mp), Mp, np.nan)
    rel = np.nanmedian(fin, 1) / np.maximum(np.median(Mn, 1), 1e-6)   # tighter to the word than to random windows
    score = support - 0.01 * rel
    s = int(np.nanargmax(score))
    hits = [k for k in range(n) if k != owner[s] and hit[s, k]]
    matches = [int(allp[s])]
    for k in hits:
        cols = np.where(owner == k)[0]
        matches.append(int(allp[cols[np.argmin(Dp[s, cols])]]))
    if len(matches) >= 2:
        Dm = C.dist(np.array(matches), np.array(matches))
        med = matches[int(np.argmin(Dm.sum(1)))]
        spread = float(np.median(Dm[np.triu_indices(len(matches), 1)]))
    else:
        med, spread = matches[0], float("nan")
    return {"seg": int(allp[s]), "medoid": med, "support": float(support[s]), "tau": float(tau[s]),
            "n": n, "videos": int(len(uv)), "matches": matches, "spread": spread,
            "pos_median": float(np.nanmedian(fin[s])), "neg_median": float(np.median(Mn[s])), "rel": float(rel[s]),
            "cands": allp}


def neg_windows(C, keys, rng, count=60):
    pool = [i for i, w in enumerate(C.windows) if not (w[4] & keys)]
    return list(rng.choice(pool, size=min(count, len(pool)), replace=False))


def null_max_support(C, n, rng, trials=30):
    """The best support random pseudo-words reach: n random windows from n different videos."""
    out = []
    by_vid = defaultdict(list)
    for i, w in enumerate(C.windows):
        by_vid[w[0]].append(i)
    vids = list(by_vid)
    for _ in range(trials):
        pick = rng.choice(len(vids), size=min(n, len(vids)), replace=False)
        wins = [int(rng.choice(by_vid[vids[j]])) for j in pick]
        keys = set().union(*(C.windows[i][4] for i in wins))
        neg = [int(i) for i in rng.choice(len(C.windows), size=min(150, len(C.windows)), replace=False)
               if i not in wins and not (C.windows[i][4] & keys)][:60]
        if len(neg) < 20:
            neg = [int(i) for i in rng.choice(len(C.windows), size=min(60, len(C.windows)), replace=False) if i not in wins]
        out.append(spot(C, wins, neg)["support"])
    return np.array(out)


# ---------------------------------------------------------------- known signs
def known_index(C):
    """key -> the existing sign of that word: jordan_shorts (same dialect) first, then the QA lexicon's matcher
    on the word's commonest surface form."""
    from isharati.lexicon import arabic as ar_lexicon
    base = DATA / "lexicon_v2"
    rows = [json.loads(l) for l in (base / "jordan_shorts" / "entries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    private_lemmatize([normalize_ar(e["gloss"]) for e in rows if " " not in e["gloss"].strip()])
    js = {}
    for e in rows:
        g = normalize_ar(e["gloss"])
        js.setdefault(g, e)
        if " " not in g:
            js.setdefault(lemma_key(g, C.cache), e)
    qa = ar_lexicon.qa_lexicon(base / "lexicon_qa.jsonl")
    out = {}
    for key, forms in C.surface.items():
        e = None
        for w, _ in forms.most_common(5):
            for c in (w, w[2:] if w.startswith("ال") else None, key):
                if c and c in js:
                    e = js[c]
                    break
            if e:
                break
        if e is None:
            m = qa.match_token(forms.most_common(1)[0][0])
            if m is not None and not m.is_letter:
                e = {"dataset": m.dataset, "sign_id": m.sign_id, "gloss": m.gloss, "keypoints_path": m.keypoints_path}
        if e is not None:
            out[key] = {"dataset": e["dataset"], "sign_id": e["sign_id"], "gloss": e["gloss"],
                        "path": str(base / e["keypoints_path"])}
    return out


def known_feature(path):
    p = np.load(path)
    return feature(p, dominant(p))


# ---------------------------------------------------------------- spot + validate
def run_spot():
    t0 = time.time()
    C = Corpus()
    rng = np.random.default_rng(0)
    print(f"{len(C.seg_vid)} segments in {len(C.vid_segs)} videos, {sum(len(c) for c in C.cues.values())} captions "
          f"({time.time() - t0:.0f}s)", flush=True)
    lag_scores = {}  # the caption-to-sign lag: the one with the most support over the 120 commonest words
    for lag in (0.0, 0.5, 1.0, 1.5, 2.0):
        C.set_lag(lag)
        keys = sorted((k for k, v in C.by_key.items() if len({C.windows[i][0] for i in v}) >= MIN_OCC),
                      key=lambda k: -len(C.by_key[k]))[:120]
        sup = [spot(C, C.by_key[k][:40], neg_windows(C, {k}, rng))["support"] for k in keys]
        lag_scores[lag] = float(np.mean(sup)) if sup else 0.0
        print(f"lag {lag}: mean support {lag_scores[lag]:.3f} over {len(sup)} words", flush=True)
    lag = max(lag_scores, key=lag_scores.get)
    C.set_lag(lag)
    known = known_index(C)
    # a word: >= MIN_OCC captions from >= 3 videos
    words = {k: v for k, v in C.by_key.items() if len(v) >= MIN_OCC and len({C.windows[i][0] for i in v}) >= 3}
    print(f"lag {lag}s; {len(words)} words in >= {MIN_OCC} captions; {sum(k in known for k in words)} have a sign",
          flush=True)
    nulls = {}

    def null_for(n):
        b = n if n <= 12 else (16 if n <= 16 else (24 if n <= 24 else (32 if n <= 32 else 60)))
        if b not in nulls:
            nulls[b] = null_max_support(C, min(b, len(C.vid_segs)), rng)
        return nulls[b]
    rand_segs = rng.choice(len(C.seg_vid), size=min(3000, len(C.seg_vid)), replace=False)
    win_sets = {k: frozenset(v) for k, v in words.items()}
    results = []
    for i, (k, wins) in enumerate(sorted(words.items(), key=lambda kv: -len(kv[1]))):
        wins = wins[:60]
        r = spot(C, wins, neg_windows(C, {k}, rng))
        nl = null_for(r["videos"])
        r["null95"], r["null_median"] = float(np.quantile(nl, 0.95)), float(np.median(nl))
        r["margin"] = r["support"] - r["null95"]
        twins = [k2 for k2, s2 in win_sets.items() if k2 != k and len(s2 & win_sets[k]) / len(s2 | win_sets[k]) >= 0.7]
        row = {"key": k, "gloss": lemma_gloss(C.surface[k].most_common(1)[0][0], C.cache),
               "surface": [w for w, _ in C.surface[k].most_common(4)], "captions": len(C.by_key[k]),
               **{x: r[x] for x in ("support", "tau", "n", "videos", "spread", "pos_median", "neg_median", "rel",
                                     "null95", "null_median", "margin")},
               "medoid": r["medoid"], "matches": r["matches"], "twins": twins,
               "accepted": bool(r["support"] >= SUPPORT and r["margin"] > 0 and not twins)}
        if k in known:
            kf = known_feature(known[k]["path"])
            d_med = float(C.dist_to(kf, np.array([r["medoid"]]))[0])
            d_c = C.dist_to(kf, r["cands"])
            d_r = C.dist_to(kf, rand_segs)
            row["known"] = {**known[k], "d_spotted": d_med,
                            "pct_vs_candidates": float(np.mean(d_c < d_med)),    # 0 = the closest candidate
                            "pct_vs_random": float(np.mean(d_r < d_med)),
                            "best_candidate_is_spotted": bool(d_med <= d_c.min() + 1e-6)}
        results.append(row)
        if i % 25 == 0:
            kn = f"known pct {row['known']['pct_vs_candidates']:.2f}" if "known" in row else ""
            print(f"  {i}/{len(words)} {row['gloss']} n={row['videos']} support {row['support']:.2f} "
                  f"null95 {row['null95']:.2f} {'ACCEPT' if row['accepted'] else ''} {kn}", flush=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "spot.json").write_text(json.dumps(
        {"lag": lag, "lag_scores": lag_scores, "results": results,
         "segments": {"vid": C.seg_vid, "span": C.seg_span},
         "nulls": {str(b): v.tolist() for b, v in nulls.items()}}, ensure_ascii=False), encoding="utf-8")
    summarize(results)


def summarize(results):
    def prec(rows):
        if not rows:
            return "n/a"
        top = np.mean([r["known"]["pct_vs_candidates"] <= 0.10 for r in rows])
        best = np.mean([r["known"]["best_candidate_is_spotted"] for r in rows])
        rnd = np.mean([r["known"]["pct_vs_random"] <= 0.10 for r in rows])
        med = np.median([r["known"]["pct_vs_candidates"] for r in rows])
        return (f"{len(rows)} words: in the top 10% of candidates {top:.0%} (chance 10%), the closest candidate "
                f"{best:.0%}, top 10% of random segments {rnd:.0%}, median percentile {med:.2f}")
    kn = [r for r in results if "known" in r]
    acc = [r for r in results if r["accepted"]]
    print(f"accepted {len(acc)}/{len(results)}; of them new words {sum('known' not in r for r in acc)}")
    for name, sel in (("jordan_shorts", lambda r: r["known"]["dataset"] == "jordan_shorts"),
                      ("other lexicon", lambda r: r["known"]["dataset"] != "jordan_shorts")):
        print(f"validation {name}, all spotted: {prec([r for r in kn if sel(r)])}")
        print(f"validation {name}, accepted only: {prec([r for r in kn if sel(r) and r['accepted']])}")


def precision(results):
    """Validation precision: of the accepted words that already have a sign, the share whose spotted sign is in the
    closest 10% of the word's candidate segments to that sign (chance: 10%)."""
    rows = [r for r in results if r["accepted"] and "known" in r]
    return (float(np.mean([r["known"]["pct_vs_candidates"] <= 0.10 for r in rows])) if rows else float("nan")), len(rows)


# ---------------------------------------------------------------- build
def face_manifest_append(rows):
    """Rows in data/face/ar/manifest.jsonl, appended under a lock file (scripts/lexicon/tid/tid_youtube.py's)."""
    lock = FACE / "manifest.jsonl.lock"
    fd = None
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        with open(FACE / "manifest.jsonl", "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if fd is not None:
            os.close(fd)
            lock.unlink(missing_ok=True)


def run_build(min_precision=0.70):
    from attach import fifty, save_face
    S = json.loads((REPORT / "spot.json").read_text(encoding="utf-8"))
    results, seg_vid, seg_span = S["results"], S["segments"]["vid"], S["segments"]["span"]
    p, n = precision(results)
    print(f"validation precision {p:.0%} on {n} accepted known words (gate {min_precision:.0%})")
    if not (p >= min_precision):
        print("below the gate: nothing added")
        return
    acc = sorted((r for r in results if r["accepted"]), key=lambda r: -r["support"])
    used, keep = set(), []
    for r in acc:  # one segment, one word: two words spotting the same recording keep the better supported one
        if r["medoid"] in used or set(r["matches"][:3]) & used:
            r["dropped"] = "same segment as a better supported word"
            continue
        used |= {r["medoid"], *r["matches"][:3]}
        keep.append(r)
    from isharati.lexicon import arabic as ar_lexicon  # noqa: F401  (installs nothing; the lexicon is only read)
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    FACE.mkdir(parents=True, exist_ok=True)
    entries, manifest, tracks = [], [], {}
    for r in keep:
        k = r["medoid"]
        vid, (a, b) = seg_vid[k], seg_span[k]
        sid = f"svjo_{vid}_{a:05d}"
        work = np.load(WORK / f"{vid}.npz")
        raw = work["raw"]
        clip = clip_of(raw, a, b)
        np.save(OUT / "signs" / f"{sid}.npy", clip.astype(np.float32))
        if vid not in tracks:
            z = np.load(TRACKS / f"{vid}.npz")
            tracks = {vid: ({k2: z[k2] for k2 in ("pose", "left_hand", "right_hand", "face", "blend", "time")},
                            json.loads(str(z["meta"])))}
        tr, meta = tracks[vid]
        t = tr["time"]
        row = {"sign_id": sid, "gloss": r["gloss"], "dataset": DATASET, "video": f"{vid}.mp4",
               "start": round(float(t[a]), 3), "end": round(float(t[b - 1]), 3), "frames": b - a, "match": 0.0,
               "found": True}
        save_face(FACE, sid, tr, raw, meta, a, b - a, 0.0, row, clip)
        manifest.append(row)
        note = {"dataset": DATASET, "keypoints_path": f"signverse_jordanian/signs/{sid}.npy", "review_status": "pending",
                "is_religious": False, "is_letter": False, "dialect": "Jordanian", "alignment": "weak_cross_occurrence",
                "support": round(r["support"], 3), "support_videos": r["videos"], "captions": r["captions"],
                "consistency": round(r["spread"], 4) if r["spread"] == r["spread"] else None,
                "margin": round(r["margin"], 3), "video": vid, "start": row["start"], "end": row["end"],
                "source": "SignVerse-2M metadata (CC-BY-NC-4.0), YouTube " + vid}
        if "known" in r:  # the lexicon already has the word: another recording, never a replacement
            entries.append({"gloss": r["known"]["gloss"], "sign_id": sid, **note, "variant_of": r["known"]["sign_id"],
                            "variant_matches_existing": r["known"]["pct_vs_candidates"] <= 0.10})
        else:
            names = [r["gloss"]] + [w for w in r["surface"] if normalize_ar(w) != normalize_ar(r["gloss"])][:3]
            for g in names:
                entries.append({"gloss": g, "sign_id": sid, **note})
    tmp = OUT / "entries.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8")
    os.replace(tmp, OUT / "entries.jsonl")
    face_manifest_append(manifest)
    print(f"{len(keep)} signs: {sum('known' not in r for r in keep)} new words, {sum('known' in r for r in keep)} "
          f"variants; {len(entries)} rows -> {OUT / 'entries.jsonl'}")


# ---------------------------------------------------------------- QA contact sheets
BW = dict(zip("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي", "'|>&<}AbptvjHxd*rzs$SDTZEgfqklmnhwYy"))


def run_qa(count=20):
    import cv2
    S = json.loads((REPORT / "spot.json").read_text(encoding="utf-8"))
    seg_vid, seg_span = S["segments"]["vid"], S["segments"]["span"]
    acc = [r for r in S["results"] if r["accepted"]]
    rng = np.random.default_rng(1)
    pick = [acc[i] for i in sorted(rng.choice(len(acc), size=min(count, len(acc)), replace=False))]
    QA.mkdir(parents=True, exist_ok=True)
    index = []
    for r in pick:
        rows = []
        for k in r["matches"][:6]:
            vid, (a, b) = seg_vid[k], seg_span[k]
            z = np.load(TRACKS / f"{vid}.npz")
            meta = json.loads(str(z["meta"]))
            cap = cv2.VideoCapture(str(VIDEOS / f"{vid}.mp4"))
            pose = z["pose"][a:b]
            xs, ys = pose[..., 0][np.isfinite(pose[..., 0])], pose[..., 1][np.isfinite(pose[..., 1])]
            W, H = meta["width"], meta["height"]
            x0, x1 = (int(max(0, np.percentile(xs, 2) * W - 60)), int(min(W, np.percentile(xs, 98) * W + 60))) if len(xs) else (0, W)
            ims = []
            for f in np.linspace(a, b - 1, 4).astype(int):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(z["time"][f]) * 1000)
                ok, im = cap.read()
                im = im[:, x0:x1] if ok else np.zeros((H, 200, 3), np.uint8)
                ims.append(cv2.resize(im, (int(im.shape[1] * 200 / im.shape[0]), 200)))
            row = np.hstack(ims)
            cv2.putText(row, f"{vid} {z['time'][a]:.1f}s{' MEDOID' if k == r['medoid'] else ''}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            rows.append(row)
        wmax = max(x.shape[1] for x in rows)
        sheet = np.vstack([np.pad(x, ((0, 0), (0, wmax - x.shape[1]), (0, 0))) for x in rows])
        name = "".join(BW.get(c, c) for c in normalize_ar(r["gloss"]))
        head = np.zeros((30, wmax, 3), np.uint8)
        cv2.putText(head, f"{name}  support {r['support']:.2f} (null95 {r['null95']:.2f}) videos {r['videos']}",
                    (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out = QA / f"{name}.jpg"
        cv2.imwrite(str(out), np.vstack([head, sheet]))
        index.append({"file": out.name, "gloss": r["gloss"], "support": r["support"], "videos": r["videos"],
                      "known": r.get("known", {}).get("dataset")})
    (QA / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} contact sheets -> {QA}")


def main():
    mode = sys.argv[1]
    if mode == "build":
        run_build()
        return
    if mode == "qa":
        run_qa()
        return
    if mode == "prep":
        for vid in video_ids():
            if (WORK / f"{vid}.npz").exists() or not (TRACKS / f"{vid}.npz").exists():
                continue
            t = time.time()
            print(vid, prep(vid), f"{time.time() - t:.0f}s", flush=True)
    elif mode == "spot":
        run_spot()
    elif mode == "summary":
        summarize(json.loads((REPORT / "spot.json").read_text(encoding="utf-8"))["results"])
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()


