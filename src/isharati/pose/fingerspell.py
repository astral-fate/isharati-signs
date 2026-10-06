"""A fingerspelled word as one clip, built from a lexicon's letter signs (any language).

The steps of scripts/lexicon/arsl/fingerspelled.py (Arabic, KArSL letters), made language-agnostic: given the same
letter clips, build(..., motion=(), hand=None) returns exactly what that script's build() returns. Used for the ASL and
TİD alphabets by scripts/lexicon/alphabet/fingerspell_test.py. For each letter:
  - a static letter contributes its hold, the stillest HOLD-frame stretch of its clip (not the raise and lower around it);
  - a motion letter (ASL J and Z, the TİD letters with a movement) contributes its whole clip, which the alphabet
    builder cut to exactly that movement: a hold would freeze it into a different handshape;
  - all letters end up on one hand. With mirror=True (KArSL, whose signer uses either hand) a letter signed with the
    other hand is mirrored (x -> -x, left and right joints swapped) onto the word's majority hand; with a fixed hand
    (one signer's alphabet) nothing is mirrored and the hold is found on that hand;
  - letters are joined by smoothstep transitions (isharati.pose.stitch._transition), TRANSITION frames each.
The face track is built from the letters' face files the same way (<faces>/<letter sign_id>.npz; a mirrored letter
swaps its Left/Right blendshapes), so the avatar keeps the signer's expression.
"""
from pathlib import Path

import numpy as np

from isharati.pose.keypoints import L_EL, L_SH, L_WR, LH, R_EL, R_SH, R_WR, RH
from isharati.pose.stitch import _transition

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


def hold(clip, hand, frames=HOLD):
    """Start of the frames-long window where the signing hand moves least, inside the middle 80% of the clip."""
    pts = clip[:, RH if hand == "right" else LH, :2]
    speed = np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=-1).mean(-1)]
    lo, hi = int(0.1 * len(clip)), max(int(0.9 * len(clip)) - frames, int(0.1 * len(clip)))
    starts = range(lo, max(lo, hi) + 1)
    return min(starts, key=lambda s: speed[s:s + frames].sum()) if len(clip) > frames else 0


def mirror_face(f, names):
    blend, face = f["blend"].copy(), f["face"].astype(np.float32).copy()
    idx = {n: k for k, n in enumerate(names)}
    for n, k in idx.items():
        if n.endswith("Left") and n[:-4] + "Right" in idx:
            j = idx[n[:-4] + "Right"]
            blend[:, [k, j]] = blend[:, [j, k]]
    face[..., 0] *= -1
    return blend, face


def build(entries, keypoints, faces: Path, motion=(), hand=None, frames=HOLD, transition=TRANSITION):
    """entries: the letters' SignEntry rows in order; keypoints: entry -> [T,50,3] (lexicon.keypoints).
    motion: glosses of motion letters (whole clip kept). hand: None = majority hand with mirroring (KArSL), or
    'right' / 'left' = one signer's hand, never mirrored.
    Returns (clip, face or None, hand used, per-letter hands, per-letter frame counts)."""
    clips = [keypoints(e) for e in entries]
    hands = [signing_hand(c) for c in clips]
    main = hand or max(set(hands), key=hands.count)
    parts, faces_, names, lengths = [], [], None, []
    for e, c, h in zip(entries, clips, hands):
        flip = hand is None and h != main
        c2 = mirror(c) if flip else c
        if e.gloss in motion:
            s, n = 0, len(c2)
        else:
            s, n = hold(c2, main, frames), frames
        seg = c2[s:s + n].astype(np.float32)
        lengths.append(len(seg))
        if parts:
            parts.append(_transition(parts[-1][-1], seg[0], transition).astype(np.float32))
        parts.append(seg)
        fp = Path(faces) / f"{e.sign_id}.npz"
        if fp.exists():
            z = np.load(fp)
            names = [str(x) for x in z["blend_names"]]
            blend, face = mirror_face(z, names) if flip else (z["blend"].astype(np.float32), z["face"].astype(np.float32))
            faces_.append((blend[s:s + n], face[s:s + n], float(z["ratio"]) if "ratio" in z.files else np.nan))
        else:
            faces_.append(None)
    clip = np.concatenate(parts)
    face_out = None
    if all(f is not None for f in faces_):  # the face over the same frames: holds, and linear blends in between
        b_parts, f_parts = [], []
        for b, f, _ in faces_:
            if b_parts:
                w = np.linspace(0, 1, transition + 2, dtype=np.float32)[1:-1]
                b_parts.append((1 - w[:, None]) * b_parts[-1][-1] + w[:, None] * b[0])
                f_parts.append((1 - w[:, None, None]) * f_parts[-1][-1] + w[:, None, None] * f[0])
            b_parts.append(b.astype(np.float32))
            f_parts.append(f.astype(np.float32))
        face_out = (np.concatenate(b_parts), np.concatenate(f_parts), np.nanmean([f[2] for f in faces_]), names)
    return clip, face_out, main, hands, lengths


def save_face(path: Path, face, n_frames, video):
    """The built face track in the data/face/<lang>/<sign_id>.npz format (scripts/face/attach.py save_face)."""
    blend, pts, ratio, names = face
    np.savez_compressed(path, blend=blend.astype(np.float16), face=pts.astype(np.float16),
                        blend_names=np.array(names), aspect=1.0, ratio=ratio, match=0.0, start=0.0,
                        end=n_frames / 25, video=video)
