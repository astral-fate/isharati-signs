"""Opposite pairs of the TİD dictionary -> one sign per word (data/lexicon_tr/pairs_fixed).

The TİD Sözlüğü channel teaches opposites in pairs: one video titled «İYİ / KÖTÜ» signs İYİ under an «İYİ» caption
(word and picture on the right of the frame), then KÖTÜ under a «KÖTÜ» caption. The first TİD build
(scripts/lexicon/youtube_dictionaries.py) cut ONE stretch of signing from each such video and gave it to both words, so
66 clips of data/lexicon_tr/dictionaries are shared by two opposite glosses and one word of each pair is signed with
its opposite's sign (iyi and kötü had the same clip).

Here each pair video is re-cut into its two signs:
  1. the video goes once through the MediaPipe Holistic task (scripts/face/track.py: body, hands and face on the
     builders' 25 fps grid, whole frame), kept on F:/tid_fix/tracks;
  2. the caption change splits the video into the first word's half and the second's: the largest jump of the right
     part of the frame (where the caption and picture sit) away from the very start and end;
  3. in each half the first signing run (signing_runs: a wrist raised above its resting height, the arms hanging) of
     at least 0.4 s is that word's sign, cut to its first performance (tid_youtube.first_repetition) and its final
     hold (trim_hold), PAD frames either side but never across the caption change. A run that flows on across the caption change (hands not
     lowered between the two signs) is split at the change. Run 1 -> the first word of the title, run 2 -> the second;
  4. checks: 0.4-4 s, dominant hand seen in >= 70 % of the frames, missing hand <= 45 % (the dictionary's limit);
  5. verification against AUTSL (data/lexicon_tr/autsl, incl. its variant rows): where AUTSL records either word of
     a pair, does it agree with the assignment? DTW over both wrists' paths (motion), clips proportion-fixed; with
     both words in AUTSL the assignment agrees when d(r1, A1) + d(r2, A2) < d(r1, A2) + d(r2, A1), with one word w
     when w's recut is closer to AUTSL's w than its opposite's recut is (autsl_check; the handshape version, per-finger
     openness, is reported beside it). A word whose recut fails the checks in every video of its pair while AUTSL
     records it takes the AUTSL medoid clip instead ("autsl_fallback" in the entry).

Output: data/lexicon_tr/pairs_fixed/{signs/<sign_id>.npy, entries.jsonl, report.json}, one entry per corrected word
(dataset "tid_dictionary_pairfix", "fixed_from": the old shared clip, "pair": [first, second]); a second video of the
same pair gives a "variant_of" row. Faces of exactly the clip frames go to data/face/tr/<sign_id>.npz
(scripts/face/attach.py's save_face format). tid_lexicon() loads the entries with override_words, so they replace the
dictionary's shared clip for these glosses. QA contact sheets (both signs side by side, labelled) on F:/tid_fix/qa.

  "D:/islam/gathring data/.venv/Scripts/python" scripts/lexicon/tid/pairs_fix.py track <k> <n>   # MediaPipe venv
  PYTHONPATH=src python scripts/lexicon/tid/pairs_fix.py cut       # recut + checks + faces + AUTSL check + entries
  PYTHONPATH=src python scripts/lexicon/tid/pairs_fix.py qa        # contact sheets on F:/tid_fix/qa
"""
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "face"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isharati.config import DATA  # noqa: E402

WORK = Path(os.environ.get("TID_FIX", r"F:\tid_fix"))
VIDEOS, TRACKS, QA = WORK / "videos", WORK / "tracks", WORK / "qa"
SRC_VIDEOS = Path(r"D:\islam\gathring data\turkish_dictionaries\tid_sozluk")  # the first TİD download (read only)
LEX = DATA / "lexicon_tr"
DICT = LEX / "dictionaries"
OUT = LEX / "pairs_fixed"
AUTSL = LEX / "autsl" / "entries.jsonl"
FACE = DATA / "face" / "tr"
DATASET, PREFIX = "tid_dictionary_pairfix", "tidpf"
CAPTION_X = 0.6     # the caption and its picture sit right of 60 % of the frame width
EDGE_S = (0.8, 0.6)  # caption changes this close to the start / end are title or end cards, not the word change
ASCII = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOU")  # cv2.putText draws ASCII only


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def pairs():
    """The shared clips: [{id, old (keypoints_path), words [first, second] in title order, title}], entries order."""
    rows = load(DICT / "entries_tid_dictionary.jsonl")
    man = {r["id"]: r for r in csv.DictReader(open(DICT / "manifest_tid_sozluk.csv", encoding="utf-8"))}
    by = defaultdict(list)
    for r in rows:
        by[r["keypoints_path"]].append(r["gloss"])
    out = []
    for kp, gl in by.items():
        if len(gl) < 2:
            continue
        vid = Path(kp).stem
        m = man[vid]
        words = [m["gloss"]] + [a for a in m["aliases"].split("|") if a]  # glosses_tr keeps the title's order
        assert sorted(words) == sorted(gl), (vid, words, gl)
        out.append({"id": vid, "old": kp, "old_sign_id": f"tr_{vid}", "words": words, "title": m["title"]})
    return out


def video_of(vid):
    for p in (VIDEOS, SRC_VIDEOS):
        hit = next(iter(sorted(p.glob(f"{vid}.*mp4"))), None)
        if hit is not None:
            return hit
    return None


def track_all(k, n):
    """Holistic task tracks of the pair videos (worker k of n), copied to F:/tid_fix/videos first."""
    from track import track
    VIDEOS.mkdir(parents=True, exist_ok=True)
    TRACKS.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pairs()):
        if i % n != k:
            continue
        out = TRACKS / f"{p['id']}.npz"
        if out.exists():
            continue
        src = video_of(p["id"])
        dst = VIDEOS / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        data, meta = track(dst)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **data)
        os.replace(tmp, out)
        print(p["id"], meta["frames"], "frames", flush=True)


def caption_change(video):
    """Seconds of the caption change (the largest jump of the frame's caption region, away from the edges), its
    strength, and the next-largest jump's strength inside the same window (for the report)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev, d = None, []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        w = img.shape[1]
        r = cv2.resize(cv2.cvtColor(img[:, int(w * CAPTION_X):], cv2.COLOR_BGR2GRAY), (64, 72)).astype(np.float32)
        if prev is not None:
            d.append(float(np.abs(r - prev).mean()))
        prev = r
    cap.release()
    d = np.array(d)
    t = (np.arange(len(d)) + 1) / fps  # jump d[i] is between frames i and i+1: the new caption shows at i+1
    inside = (t >= EDGE_S[0]) & (t <= len(d) / fps - EDGE_S[1])
    dd = np.where(inside, d, -1)
    i = int(np.argmax(dd))
    rest = dd.copy()
    rest[max(0, i - 3):i + 4] = -1
    return float(t[i]), float(d[i]), float(max(rest.max(), 0))


def load_track(vid):
    from attach import fifty
    z = np.load(TRACKS / f"{vid}.npz")
    tr = {k: z[k] for k in ("pose", "left_hand", "right_hand", "face", "blend", "time")}
    return tr, fifty(tr), json.loads(str(z["meta"]))


RAISED = 0.45   # shoulder widths above the wrist's resting height: signing
MIN_RUN = 10    # frames (0.4 s): shorter raised stretches are not a sign


def signing_runs(tr, meta, min_len=MIN_RUN, max_gap=6):
    """[start, end) runs of frames where either wrist (the body model's, 5-frame median) is raised RAISED shoulder
    widths above its resting height (its 90th-percentile height in the video: these signers rest with the arms
    hanging), gaps of up to max_gap frames bridged. tid_youtube.signing_runs (a wrist above chest level, or moving)
    misses the signs these pair videos make low and slowly in front of the waist («DÜZ», «YAVAŞ»)."""
    P = tr["pose"][..., :2].astype(np.float64)
    P[..., 0] *= float(meta.get("aspect", 1.0))
    sw = np.nanmedian(np.abs(P[:, 11, 0] - P[:, 12, 0]))
    sh_y = np.nanmedian((P[:, 11, 1] + P[:, 12, 1]) / 2)
    active = np.zeros(len(P), bool)
    for wr in (15, 16):
        y = (P[:, wr, 1] - sh_y) / sw
        ok = ~np.isnan(y)
        if ok.sum() < 3:
            continue
        y = np.interp(np.arange(len(y)), np.where(ok)[0], y[ok])
        y = np.array([np.median(y[max(0, i - 2):i + 3]) for i in range(len(y))])
        active |= y < np.percentile(y, 90) - RAISED
    runs, start, gap = [], None, 0
    for i, a in enumerate(active):
        if a:
            start = i if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(active)))
    # the hand model's view of the run: a run without a seen hand is the body model's start-up noise, and the run is
    # cut to the frames from the first to the last seen hand (the raising and lowering blur the hands)
    seen = ~(np.isnan(tr["left_hand"][:, :, 0]).all(1) & np.isnan(tr["right_hand"][:, :, 0]).all(1))
    out = []
    for a, b in runs:
        idx = np.where(seen[a:b])[0]
        if len(idx) >= 0.1 * (b - a):
            out.append((a + int(idx[0]), a + int(idx[-1]) + 1))
    return [(a, b) for a, b in out if b - a >= min_len]


def word_spans(runs, kb):
    """The two words' runs: a run crossing the caption change frame kb is split there; each run to the half holding
    it. Returns ([runs of half 1], [runs of half 2])."""
    halves = ([], [])
    for a, b in runs:
        if a < kb < b:
            if kb - a >= MIN_RUN:
                halves[0].append((a, kb))
            if b - kb >= MIN_RUN:
                halves[1].append((kb, b))
        else:
            halves[0 if b <= kb else 1].append((a, b))
    return halves


def cut_one(raw, raw_sq, run, lo, hi):
    """First performance of a run, its final hold trimmed, padded within [lo, hi). Returns (a, b, checks dict)."""
    import tid_youtube as ty
    from isharati.pose.keypoints import LH, RH, interpolate_missing
    a, b = run
    b = ty.first_repetition(raw_sq, a, b)
    a, b = ty.trim_hold(raw_sq, a, b)
    a, b = max(lo, a - ty.PAD), min(hi, b + ty.PAD)
    seg = raw[a:b]
    lh = ~np.isnan(seg[:, LH, 0]).all(axis=1)
    rh = ~np.isnan(seg[:, RH, 0]).all(axis=1)
    _, missing = interpolate_missing(seg)
    c = {"seconds": round((b - a) / ty.FPS, 2), "dominant_hand": round(float(max(lh.mean(), rh.mean())), 3),
         "missing": round(missing, 3)}
    fails = []
    if not ty.MIN_S <= c["seconds"] <= ty.MAX_S:
        fails.append(f"duration {c['seconds']} s")
    if c["dominant_hand"] < ty.MIN_HAND:
        fails.append(f"dominant hand {c['dominant_hand']:.0%}")
    if missing > ty.MAX_MISSING:
        fails.append(f"missing hand {missing:.2f}")
    c["fails"] = fails
    return a, b, c


def recut(p):
    """Both signs of one pair video: {word: {a, b, start, end, checks, clip}} plus the caption change."""
    import tid_youtube as ty
    from isharati.pose.keypoints import interpolate_missing, normalize
    tr, raw, meta = load_track(p["id"])
    sq = ty.square(raw, meta)
    t = tr["time"]
    tc, strength, second = caption_change(VIDEOS / Path(meta["video"]).name)
    kb = int(np.searchsorted(t, tc))
    runs = signing_runs(tr, meta)
    halves = word_spans(runs, kb)
    info = {"caption_change": round(tc, 2), "caption_jump": round(strength, 1), "next_jump": round(second, 1),
            "runs": [[list(map(int, r)) for r in h] for h in halves], "words": {}}
    if strength < 5 * max(second, 1.0):  # no clear caption change: run order alone, when there are exactly two runs
        allr = runs
        if len(allr) == 2:
            halves, kb = ([allr[0]], [allr[1]]), allr[1][0]
            info["note"] = "no clear caption change: the two signing runs in order"
        else:
            info["unfixed"] = f"no clear caption change and {len(allr)} signing runs"
            return tr, raw, meta, info
    # where the old shared clip sits in the video (scripts/face/attach.py's locate), and which word(s) it shows
    from attach import locate
    old = np.load(DICT / "signs" / f"{p['id']}.npy")
    if len(old) <= len(raw):
        s0, d = locate(old, raw)
        e0 = s0 + len(old)
        cover = [max(0, min(e0, kb) - s0) / len(old), max(0, e0 - max(s0, kb)) / len(old)]
        info["old_clip"] = {"start": round(float(t[s0]), 2), "end": round(float(t[e0 - 1]), 2), "match": round(d, 3),
                            "shows": "both" if min(cover) > 0.25 else p["words"][int(np.argmax(cover))]}
    for k, (w, h) in enumerate(zip(p["words"], halves)):
        lo, hi = (0, kb) if k == 0 else (kb, len(raw))
        if not h:
            info["words"][w] = {"fails": ["no signing run under its caption"]}
            continue
        a, b, c = cut_one(raw, sq, h[0], lo, hi)
        filled, _ = interpolate_missing(raw[a:b])
        c.update(a=int(a), b=int(b), start=round(float(t[a]), 3), end=round(float(t[b - 1]), 3),
                 runs_in_half=len(h))
        c["clip"] = normalize(filled).astype(np.float32)
        info["words"][w] = c
    return tr, raw, meta, info


# ---- comparison with AUTSL
def _fixed(clip):
    from isharati.pose import proportions
    from isharati.pose.keypoints import pin_collapsed_hands
    return proportions.fix(pin_collapsed_hands(np.asarray(clip, np.float32)))


def motion(clip):
    """Per frame, both wrists (neck-centred, shoulder widths, proportion-fixed): the sign's movement and place. On 49
    AUTSL signs the dictionary also has (a single clip), the dictionary clip of the same sign was closer than a random
    other sign's for 81 % of the AUTSL recordings by this measure; adding hand direction (76 %) or per-finger openness
    (65 %) made it worse: the two datasets' hands differ more in tracking than between signs."""
    from isharati.pose.keypoints import L_WR, R_WR
    c = _fixed(clip)
    return np.concatenate([c[:, R_WR, :2], c[:, L_WR, :2]], 1)[:, :, None].astype(np.float32)


def openness(clip):
    """Handshape: per-finger openness (tip-to-knuckle over palm length) of the signing hand (the one seen more) over
    its stillest 6 frames, as scripts/lexicon/arsl/fingerspelled.py measures it."""
    from isharati.pose.keypoints import LH, RH
    c = _fixed(clip)
    spread = [np.ptp(c[:, h, 0], axis=1).mean() for h in (RH, LH)]  # a never-seen hand is pinned to one point
    h = c[:, RH if spread[0] >= spread[1] else LH, :2]
    sp = np.r_[0, np.linalg.norm(np.diff(h, axis=0), axis=-1).mean(-1)]
    lo, hi = int(0.1 * len(c)), max(int(0.9 * len(c)) - 6, int(0.1 * len(c)) + 1)
    st = min(range(lo, hi), key=lambda i: sp[i:i + 6].sum()) if len(c) > 8 else 0
    f = h[st:st + 6].mean(0)
    palm = np.linalg.norm(f[9] - f[0]) + 1e-6
    return np.array([np.linalg.norm(f[tp] - f[kn]) / palm for kn, tp in ((2, 4), (5, 8), (9, 12), (13, 16), (17, 20))])


def dist(a, b):
    from isharati.pose.dtw import dtw_distance
    return dtw_distance(motion(a), motion(b))


def autsl_clips():
    """gloss -> [(sign_id, keypoints_path, clip, row)] of AUTSL's recordings (medoid and variant rows alike)."""
    out = defaultdict(list)
    if not AUTSL.exists():
        return out
    for r in load(AUTSL):
        p = LEX / r["keypoints_path"]
        if p.exists():
            out[r["gloss"]].append((r["sign_id"], r["keypoints_path"], np.load(p), r))
    return out


def autsl_check(words, clips, A):
    """Do AUTSL's recordings agree with the assignment run 1 -> words[0], run 2 -> words[1]? Mean DTW distance (motion)
    from each recut sign to AUTSL's recordings of each word. With both words in AUTSL the assignment agrees when
    d(r1, A1) + d(r2, A2) < d(r1, A2) + d(r2, A1); with one word w, when its recut is closer to A_w than its opposite's
    recut is. Per word, "word": the recut of w is closer to AUTSL's w than [to AUTSL's opposite, or than the opposite's
    recut is]. "handshape": the same with the openness vectors instead of motion."""
    w1, w2 = words
    have = [w for w in words if A.get(w)]
    if not have or not all(w in clips for w in words):
        return {}
    d = {(r, a): float(np.mean([dist(clips[r], c) for _, _, c, _ in A[a]])) for r in words for a in have}
    o = {(r, a): float(np.mean([np.linalg.norm(openness(clips[r]) - openness(c)) for _, _, c, _ in A[a]]))
         for r in words for a in have}
    out = {"autsl_words": have, "dtw": {f"{r}~autsl:{a}": round(v, 3) for (r, a), v in d.items()}}
    for name, m in (("motion", d), ("handshape", o)):
        if len(have) == 2:
            agree = m[w1, w1] + m[w2, w2] < m[w1, w2] + m[w2, w1]
            per = {w: m[w, w] < m[w, words[1 - k]] for k, w in enumerate(words)}
        else:
            w = have[0]
            opp = words[1 - words.index(w)]
            agree = m[w, w] < m[opp, w]
            per = {w: agree}
        out[name] = {"assignment_agrees": bool(agree), "words": {k: bool(v) for k, v in per.items()}}
    return out


def cut_all():
    import tid_youtube as ty
    from attach import save_face
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    FACE.mkdir(parents=True, exist_ok=True)
    A = autsl_clips()
    report, cand = [], defaultdict(list)  # word -> [(pair index, sign_id, info)]
    for n, p in enumerate(pairs()):
        if not (TRACKS / f"{p['id']}.npz").exists():
            report.append({"id": p["id"], "words": p["words"], "unfixed": "no track"})
            continue
        tr, raw, meta, info = recut(p)
        rec = {"id": p["id"], "title": p["title"], "words": p["words"], "old": p["old"],
               **{k: v for k, v in info.items() if k != "words"}, "signs": {}}
        clips = {w: c["clip"] for w, c in info["words"].items() if "clip" in c and not c["fails"]}
        rec["autsl"] = autsl_check(p["words"], clips, A)
        for k, w in enumerate(p["words"]):
            c = info["words"].get(w, {"fails": ["not cut"]})
            s = {kk: v for kk, v in c.items() if kk != "clip"}
            sid = f"{PREFIX}_{p['id']}_{k + 1}"
            s["sign_id"] = sid
            if not c["fails"] and "clip" in c:
                np.save(OUT / "signs" / f"{sid}.npy", c["clip"])
                row = {"sign_id": sid, "gloss": w, "dataset": DATASET, "video": Path(meta["video"]).name,
                       "start": c["start"], "end": c["end"], "frames": c["b"] - c["a"], "match": 0.0, "found": True}
                fresh = not (FACE / f"{sid}.npz").exists()
                save_face(FACE, sid, tr, raw, meta, c["a"], c["b"] - c["a"], 0.0, row, c["clip"])
                s["face_detected"] = row["face_detected"]
                if fresh:
                    ty.face_manifest_append(row)
                cand[w].append((n, sid, s, p, rec))
            rec["signs"][w] = s
        report.append(rec)
        print(p["id"], " / ".join(f"{w}: {'ok' if not rec['signs'][w]['fails'] else rec['signs'][w]['fails']}"
                                    for w in p["words"]), flush=True)
    entries = build_entries(report, cand, A)
    write_entries(entries)
    summary(report, entries)


def build_entries(report, cand, A):
    """One entry per word: the recut from the first video of its pair (the clip the lexicon used); a second video's
    recut becomes a variant row; a word with no passing recut takes AUTSL's medoid clip when AUTSL records it."""
    words = []
    for rec in report:
        for k, w in enumerate(rec["words"]):
            if w not in words:
                words.append(w)
    pair_of, old_of = {}, {}
    for rec in report:
        for k, w in enumerate(rec["words"]):
            pair_of.setdefault(w, rec["words"])
            old_of.setdefault(w, rec["old"])  # the first clip of the gloss in the dictionary file: the one in use
    entries = []
    for w in words:
        base = {"gloss": w, "dataset": DATASET, "review_status": "pending", "is_religious": False, "is_letter": False,
                "pair": pair_of[w]}
        cs = cand.get(w, [])
        if cs:
            primary = None
            for n, sid, s, p, rec in cs:
                row = {**base, "sign_id": sid, "keypoints_path": f"pairs_fixed/signs/{sid}.npy", "fixed_from": p["old"],
                       "pair": p["words"], "source": f"TİD Sözlüğü youtu.be/{p['id']} «{p['title']}», "
                                                     f"{s['start']:.2f}-{s['end']:.2f} s",
                       "checks": {k: s[k] for k in ("seconds", "dominant_hand", "missing")}}
                if rec.get("autsl"):
                    row["autsl_check"] = {k: rec["autsl"][k] for k in ("autsl_words", "motion", "handshape")}
                if primary is None:
                    primary = sid
                else:
                    row["variant_of"] = primary
                entries.append(row)
            continue
        if A.get(w):
            med = next((x for x in A[w] if not x[3].get("variant_of", "").startswith("autsl_")), A[w][0])
            sid = f"{PREFIX}_autsl_{med[0]}"
            np.save(OUT / "signs" / f"{sid}.npy", med[2].astype(np.float32))
            f = FACE / f"{med[0]}.npz"
            if f.exists() and not (FACE / f"{sid}.npz").exists():
                shutil.copy2(f, FACE / f"{sid}.npz")
            entries.append({**base, "sign_id": sid, "keypoints_path": f"pairs_fixed/signs/{sid}.npy",
                            "fixed_from": old_of[w], "source": f"AUTSL ({med[3].get('source', med[0])}): the recut "
                                                              f"failed its checks", "autsl_fallback": med[0]})
    return entries


def write_entries(entries):
    tmp = OUT / "entries.jsonl.tmp"
    tmp.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8")
    os.replace(tmp, OUT / "entries.jsonl")


def summary(report, entries):
    fixed_words = {e["gloss"] for e in entries if not e.get("variant_of")}
    all_words = {w for r in report for w in r["words"]}
    pairs_seen = {tuple(r["words"]) for r in report}
    pairs_fixed = {pr for pr in pairs_seen if all(w in fixed_words for w in pr)}
    pairs_part = {pr for pr in pairs_seen if any(w in fixed_words for w in pr)} - pairs_fixed
    agree = {}
    for kind in ("motion", "handshape"):
        pa = [(r["id"], r["autsl"][kind]["assignment_agrees"]) for r in report if r.get("autsl")]
        pw = [(r["id"], w, v) for r in report if r.get("autsl") for w, v in r["autsl"][kind]["words"].items()]
        agree[kind] = {"pair_videos": {"agree": sum(a for _, a in pa), "of": len(pa),
                                       "disagree": [i for i, a in pa if not a]},
                       "words": {"agree": sum(a for _, _, a in pw), "of": len(pw),
                                 "disagree": [f"{i}:{w}" for i, w, a in pw if not a]}}
    unfixed = {}
    for w in sorted(all_words - fixed_words):
        why = []
        for r in report:
            if w in r["words"]:
                s = r.get("signs", {}).get(w, {})
                why.append(f"{r['id']}: " + (", ".join(s.get("fails", [])) or r.get("unfixed", "?")))
                shows = r.get("old_clip", {}).get("shows")
        unfixed[w] = {"why": why, "old_clip_shows": shows}
    out = {"videos": len(report), "pairs": len(pairs_seen), "pairs_fixed": len(pairs_fixed),
           "pairs_partly_fixed": sorted("/".join(pr) for pr in pairs_part),
           "pairs_unfixed": sorted("/".join(pr) for pr in pairs_seen - pairs_fixed - pairs_part),
           "words": len(all_words), "words_fixed": len(fixed_words), "words_unfixed": unfixed,
           "autsl_fallbacks": [e["gloss"] for e in entries if e.get("autsl_fallback")],
           "variants": len([e for e in entries if e.get("variant_of")]),
           "autsl_agreement": agree, "videos_detail": report}
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "videos_detail"}, ensure_ascii=False, indent=1))


# ---- QA
def qa():
    """Per pair video, a sheet: row 1 the first word's clip, row 2 the second's (6 frames each from the video, with
    the tracked hands drawn), labelled with the word, times and checks."""
    import cv2
    QA.mkdir(parents=True, exist_ok=True)
    rep = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    for r in rep["videos_detail"]:
        if "signs" not in r:
            continue
        tr, raw, meta = load_track(r["id"])
        cap = cv2.VideoCapture(str(VIDEOS / f"{r['id']}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        want = {int(round(float(tr["time"][i]) * fps)) for w in r["words"] if "a" in r["signs"][w]
                for i in np.linspace(r["signs"][w]["a"], r["signs"][w]["b"] - 1, 6).round().astype(int)}
        frames, i, last = {}, 0, None
        while len(frames) < len(want):  # only the frames the sheet shows (memory)
            ok, img = cap.read()
            if not ok:
                break
            last = img
            if i in want:
                frames[i] = img
            i += 1
        cap.release()
        rows = []
        for k, w in enumerate(r["words"]):
            s = r["signs"][w]
            W, H = 240, 135
            if "a" not in s:
                tile = np.zeros((H + 24, W * 6, 3), np.uint8)
                cv2.putText(tile, f"{k + 1}. {w.translate(ASCII)}: NOT CUT {s.get('fails')}", (6, 20), 0, 0.6, (0, 0, 255), 2)
                rows.append(tile)
                continue
            picks = np.linspace(s["a"], s["b"] - 1, 6).round().astype(int)
            tiles = []
            for i in picks:
                img = frames.get(int(round(float(tr["time"][i]) * fps)), last).copy()
                h, wd = img.shape[:2]
                for hand, col in (("left_hand", (0, 200, 0)), ("right_hand", (0, 128, 255))):
                    for x, y, _ in tr[hand][i]:
                        if np.isfinite(x):
                            cv2.circle(img, (int(x * wd), int(y * h)), 3, col, -1)
                img = cv2.resize(img, (W, H))
                cv2.putText(img, f"{float(tr['time'][i]):.2f}s", (4, 16), 0, 0.45, (0, 0, 255), 1)
                tiles.append(img)
            band = np.full((24, W * 6, 3), 255, np.uint8)
            label = (f"{k + 1}. {w.translate(ASCII).upper()}  {s['start']:.2f}-{s['end']:.2f}s  hand {s['dominant_hand']:.0%}  "
                     f"{'OK' if not s['fails'] else 'FAIL ' + ', '.join(s['fails'])}")
            ag = r.get("autsl", {}).get("motion", {}).get("words", {}).get(w)
            if ag is not None:
                label += f"  AUTSL {'agrees' if ag else 'DISAGREES'}"
            cv2.putText(band, label, (6, 17), 0, 0.5, (0, 0, 0), 1)
            rows.append(np.vstack([band, np.hstack(tiles)]))
        title = np.full((26, W * 6, 3), 230, np.uint8)
        old = r.get("old_clip", {})
        cv2.putText(title, f"{r['id']}  caption change {r.get('caption_change')} s   old shared clip "
                           f"{old.get('start')}-{old.get('end')} s = {str(old.get('shows', '?')).translate(ASCII)}",
                    (6, 18), 0, 0.5, (0, 0, 0), 1)
        name = f"{r['words'][0]}_{r['words'][1]}_{r['id']}.jpg"
        ok, buf = cv2.imencode(".jpg", np.vstack([title, *rows]))
        (QA / name).write_bytes(buf.tobytes())  # imwrite cannot write non-ASCII paths on Windows
    print("QA sheets ->", QA)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "track":
        track_all(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "cut":
        cut_all()
    elif mode == "qa":
        qa()
