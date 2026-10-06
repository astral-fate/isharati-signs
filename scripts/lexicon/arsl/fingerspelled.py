"""Words that a native signer says are fingerspelled, built as signs from the KArSL letter signs.

The lexicon never fingerspells on its own: a word without a sign is reported, not spelled. When a Deaf reviewer
decides that a word IS spelled (fingerspelled.json: word -> letters, reviewer, date), this builds its sign (letters
separated by spaces, words by "|", which become a pause; "then": an existing sign (or a list of signs) signed after the
letters, such as the honorific in «علي رضي الله عنه»; entries not yet seen by the reviewer carry "review_status": "pending"):
  - each letter's hold, the stillest stretch of its KArSL clip (not the raise and lower around it), HOLD frames;
  - all letters on one hand: KArSL signs 30 of its 31 letters with the signer's left hand and «د» with the right,
    so a letter on the other hand is mirrored (x -> -x, left and right joints swapped) onto the word's majority hand;
  - letters joined by smoothstep transitions (isharati.pose.stitch._transition), TRANSITION frames each.
The face track is built from the letters' face files the same way (data/face/ar/<letter sign_id>.npz; a mirrored
letter swaps its Left/Right blendshapes), so the avatar keeps the signer's expression.

Output: data/lexicon_v2/reviewed/{signs/fs_*.npy, entries.jsonl} (dataset "reviewed_fingerspelling", review_status
"approved" for the reviewer's decision) and data/face/ar/fs_*.npz. scripts/lexicon/arsl/merge.py reads reviewed/
first, so a reviewed word replaces any sign the other sources had for it.
  PYTHONPATH=src python scripts/lexicon/arsl/fingerspelled.py && python scripts/lexicon/arsl/merge.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.lexicon import arabic as ar_lexicon  # noqa: E402
from isharati.pose.keypoints import L_EL, L_SH, L_WR, LH, R_EL, R_SH, R_WR, RH  # noqa: E402
from isharati.pose.stitch import _transition  # noqa: E402

OUT = DATA / "lexicon_v2" / "reviewed"
FACES = DATA / "face" / "ar"
HOLD, TRANSITION = 8, 5  # frames at 25 fps: about half a second per letter
SWAP = [(R_SH, L_SH), (R_EL, L_EL), (R_WR, L_WR)]


def signing_hand(clip):
    """'right' or 'left': the hand whose wrist rises highest (image y points down)."""
    return "right" if clip[:, R_WR, 1].min() < clip[:, L_WR, 1].min() else "left"


def mirror(clip):
    out = clip.copy()
    out[..., 0] *= -1                                 # neck-centred: x -> -x is the mirror image
    for a, b in SWAP:
        out[:, [a, b]] = out[:, [b, a]]
    out[:, LH], out[:, RH] = clip[:, RH].copy(), clip[:, LH].copy()
    out[:, LH, 0] *= -1
    out[:, RH, 0] *= -1
    return out


def hold(clip, hand):
    """Start of the HOLD-frame window where the signing hand moves least, inside the middle 80% of the clip."""
    pts = clip[:, RH if hand == "right" else LH, :2]
    speed = np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=-1).mean(-1)]
    lo, hi = int(0.1 * len(clip)), max(int(0.9 * len(clip)) - HOLD, int(0.1 * len(clip)))
    starts = range(lo, max(lo, hi) + 1)
    return min(starts, key=lambda s: speed[s:s + HOLD].sum()) if len(clip) > HOLD else 0


def mirror_face(f, names):
    blend, face = f["blend"].copy(), f["face"].astype(np.float32).copy()
    idx = {n: k for k, n in enumerate(names)}
    for n, k in idx.items():
        if n.endswith("Left") and n[:-4] + "Right" in idx:
            j = idx[n[:-4] + "Right"]
            blend[:, [k, j]] = blend[:, [j, k]]
    face[..., 0] *= -1
    return blend, face


KARSL_KP = DATA / "karsl_kp"
FINGERS = ((2, 4), (5, 8), (9, 12), (13, 16), (17, 20))  # thumb, index, middle, ring, little: knuckle, tip


def openness(clip):
    """Per-finger openness (tip-to-knuckle distance / palm length) of the signing hand over its stillest frames."""
    base = LH.start if signing_hand(clip) == "left" else RH.start
    h = clip[:, base:base + 21]
    sp = np.r_[0, np.linalg.norm(np.diff(h[:, :, :2], axis=0), axis=-1).mean(-1)]
    lo, hi = int(0.1 * len(clip)), max(int(0.9 * len(clip)) - 6, int(0.1 * len(clip)) + 1)
    s = min(range(lo, hi), key=lambda i: sp[i:i + 6].sum()) if len(clip) > 8 else 0
    f = h[s:s + 6].mean(0)
    palm = np.linalg.norm(f[9] - f[0]) + 1e-6
    return np.array([np.linalg.norm(f[t] - f[m]) / palm for m, t in FINGERS])


def letter_clip(entry, lex):
    """The KArSL recording of a letter whose handshape is closest to the median over all its recordings. The lexicon
    keeps the recording most typical in its whole-body motion, and a letter is its handshape: for alef that recording
    had the index finger open (openness 0.94) where the signer's fist (all recordings: 0.39) has it closed."""
    import csv
    from isharati.pose import proportions
    from isharati.pose.keypoints import interpolate_missing, normalize, pin_collapsed_hands
    manifest = KARSL_KP / "manifest.csv"
    clips = []
    if manifest.exists():
        for r in csv.DictReader(open(manifest, encoding="utf-8")):
            if r["sign_id"] != entry.sign_id or r["ok"] != "True" or not (KARSL_KP / r["npy_path"]).exists():
                continue
            raw = np.load(KARSL_KP / r["npy_path"])
            if len(raw) < 8:
                continue
            filled, missing = interpolate_missing(raw)
            if missing <= 0.3:
                clips.append(proportions.fix(pin_collapsed_hands(normalize(filled))))
    if len(clips) < 3:  # too few recordings for a consensus: the lexicon's own clip
        return lex.keypoints(entry)
    shapes = np.array([openness(c) for c in clips])
    return clips[int(np.argmin(np.linalg.norm(shapes - np.median(shapes, 0), axis=1)))]


def build(word, letters, lex, gap_before=(), then=None):
    entries = [lex._letters[ch] for ch in letters]
    clips = [letter_clip(e, lex) for e in entries]
    hands = [signing_hand(c) for c in clips]
    main = max(set(hands), key=hands.count)
    parts, faces, names = [], [], None
    gaps = set(gap_before or ())
    for k, (e, c, h) in enumerate(zip(entries, clips, hands)):
        flip = h != main
        c2 = mirror(c) if flip else c
        s = hold(c2, main)
        seg = c2[s:s + HOLD].astype(np.float32)
        if parts:  # between two words of a name, a pause: the transition takes twice as long
            parts.append(_transition(parts[-1][-1], seg[0], TRANSITION * (2 if k in gaps else 1)).astype(np.float32))
        parts.append(seg)
        fp = FACES / f"{e.sign_id}.npz"
        if fp.exists():
            z = np.load(fp)
            names = [str(n) for n in z["blend_names"]]
            blend, face = mirror_face(z, names) if flip else (z["blend"].astype(np.float32), z["face"].astype(np.float32))
            faces.append((blend[s:s + HOLD], face[s:s + HOLD], float(z["ratio"]) if "ratio" in z.files else np.nan))
        else:
            faces.append(None)
    then_parts = []  # existing signs after the letters, e.g. the honorific: «علي» + «رضي الله عنه»
    for t in then or ():
        t_clip = lex.keypoints(t).astype(np.float32)
        parts.append(_transition(parts[-1][-1], t_clip[0], TRANSITION * 2).astype(np.float32))
        parts.append(t_clip)
        fp = FACES / f"{t.sign_id}.npz"
        z = np.load(fp) if fp.exists() else None
        then_parts.append((len(t_clip), (z["blend"].astype(np.float32), z["face"].astype(np.float32)) if z is not None else None))
    clip = np.concatenate(parts)
    face_out = None
    if all(f is not None for f in faces):  # the face over the same frames: holds, and linear blends in between
        b_parts, f_parts = [], []
        for k, (b, f, _) in enumerate(faces):
            if b_parts:
                w = np.linspace(0, 1, TRANSITION * (2 if k in gaps else 1) + 2, dtype=np.float32)[1:-1]
                b_parts.append((1 - w[:, None]) * b_parts[-1][-1] + w[:, None] * b[0])
                f_parts.append((1 - w[:, None, None]) * f_parts[-1][-1] + w[:, None, None] * f[0])
            b_parts.append(b.astype(np.float32))
            f_parts.append(f.astype(np.float32))
        for n_t, tface in then_parts:  # each appended sign's own face if it has one; else NaN (expression kept)
            tb, tf = tface if tface is not None and len(tface[0]) == n_t else (
                np.full((n_t, b_parts[0].shape[1]), np.nan, np.float32), np.full((n_t, *f_parts[0].shape[1:]), np.nan, np.float32))
            last_b = next((r for r in reversed(np.concatenate(b_parts)) if np.isfinite(r).all()), b_parts[0][0])
            last_f = next((r for r in reversed(np.concatenate(f_parts)) if np.isfinite(r).all()), f_parts[0][0])
            w = np.linspace(0, 1, TRANSITION * 2 + 2, dtype=np.float32)[1:-1]
            nb = tb[0] if np.isfinite(tb[0]).all() else last_b
            nf = tf[0] if np.isfinite(tf[0]).all() else last_f
            b_parts.append((1 - w[:, None]) * last_b + w[:, None] * nb)
            f_parts.append((1 - w[:, None, None]) * last_f + w[:, None, None] * nf)
            b_parts.append(tb)
            f_parts.append(tf)
        face_out = (np.concatenate(b_parts), np.concatenate(f_parts), np.nanmean([f[2] for f in faces]), names)
    return clip, face_out, main, hands


def main():
    words = json.loads((Path(__file__).with_name("fingerspelled.json")).read_text(encoding="utf-8"))["words"]
    lex = ar_lexicon.qa_lexicon(DATA / "lexicon_v2" / "lexicon_qa.jsonl")
    (OUT / "signs").mkdir(parents=True, exist_ok=True)
    FACES.mkdir(parents=True, exist_ok=True)
    rows = []
    for word, info in words.items():
        words_letters = [w.split() for w in info["letters"].split("|")]  # "|" separates words, spaces letters
        letters = [ch for w in words_letters for ch in w]
        gap_before, k = [], 0
        for w in words_letters[:-1]:
            k += len(w)
            gap_before.append(k)
        missing = [ch for ch in letters if ch not in lex._letters]
        if missing:
            print(f"{word}: no letter sign for {missing}, skipped")
            continue
        sid = "fs_" + hashlib.sha1(word.encode()).hexdigest()[:8]
        wanted = info.get("then") or []
        wanted = [wanted] if isinstance(wanted, str) else wanted  # one gloss or a list: «رضي», «الله», «هي»
        then = [lex.lookup(g) for g in wanted]
        if None in then:
            print(f"{word}: no sign for {[g for g, t in zip(wanted, then) if t is None]} to follow the letters, skipped")
            continue
        clip, face, hand, hands = build(word, letters, lex, gap_before, then)
        np.save(OUT / "signs" / f"{sid}.npy", clip)
        if face is not None:
            blend, pts, ratio, names = face
            np.savez_compressed(FACES / f"{sid}.npz", blend=blend.astype(np.float16), face=pts.astype(np.float16),
                                blend_names=np.array(names), aspect=1.0, ratio=ratio, match=0.0, start=0.0,
                                end=len(clip) / 25, video="fingerspelled from KArSL letters")
        rows.append({"gloss": word, "sign_id": sid, "dataset": "reviewed_fingerspelling",
                     "keypoints_path": f"reviewed/signs/{sid}.npy", "review_status": info.get("review_status", "approved"),
                     "is_religious": info.get("is_religious", False), "is_letter": False,
                     "fingerspelled": " ".join(letters), "reviewed_by": info.get("reviewed_by", ""),
                     "reviewed_on": info.get("reviewed_on", ""), "note": info.get("note", "")})
        mirrored = [ch for ch, h in zip(letters, hands) if h != hand]
        print(f"{word}: {' '.join(letters)} -> {sid}, {len(clip)} frames on the {hand} hand"
              f"{', mirrored: ' + ' '.join(mirrored) if mirrored else ''}; face {'yes' if face else 'no'}")
    (OUT / "entries.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} fingerspelled words -> {OUT / 'entries.jsonl'}")


if __name__ == "__main__":
    main()
