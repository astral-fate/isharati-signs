"""Gloss -> pose for ASL without fingerspelling Islamic terms.

Words with no sign in the lexicon are not fingerspelled; each becomes a short marked hold (kind "missing") so the
video keeps its timing and the report lists exactly which signs still have to be collected. Everything else is
isharati.gloss2pose.StitchPoser's stitching with smoothstep transitions.
"""
import numpy as np

from isharati.pose.stitch import _transition
from isharati.pose.keypoints import L_EL, L_SH, L_WR, LH, R_EL, R_SH, R_WR, RH  # noqa: F401
from isharati.types import GlossResult, Segment

MISSING_FRAMES = 12  # about half a second at 25 fps
REST_HOLD = 15       # frames held with the hands down at the end, so the video does not stop mid-sign


def relaxed_hand(palm: float, medial: float) -> np.ndarray:
    """21 hand landmarks of a hand hanging at rest, relative to its wrist (image axes: y down, z toward the camera
    negative): fingers down and gently curled toward the palm, palm facing the thigh (medial: +1 or -1 in x, toward the
    body's midline), thumb and index in front. palm = wrist -> middle-knuckle length."""
    h = np.zeros((21, 3), np.float32)
    knuckles = {5: (0.90, -0.30), 9: (0.95, -0.10), 13: (0.92, 0.10), 17: (0.84, 0.28)}  # (down, depth) x palm
    lengths = {5: 0.85, 9: 0.95, 13: 0.88, 17: 0.70}                                      # finger length x palm
    curl = np.radians([12, 35, 55])                       # each segment's angle from straight down, toward the palm
    for k, (down, depth) in knuckles.items():
        p = np.array([0.0, down * palm, depth * palm], np.float32)
        h[k] = p
        for j, (share, a) in enumerate(zip((0.45, 0.30, 0.25), curl)):
            p = p + lengths[k] * palm * share * np.array([medial * np.sin(a), np.cos(a), 0.0], np.float32)
            h[k + 1 + j] = p
    for j, (m, d, z) in enumerate(((0.10, 0.25, -0.25), (0.15, 0.45, -0.42), (0.18, 0.60, -0.50), (0.20, 0.75, -0.55))):
        h[1 + j] = np.array([medial * m, d, z], np.float32) * palm
    return h


def rest_pose(frame: np.ndarray) -> np.ndarray:
    """The same body with the arms hanging: elbows and wrists straight below the shoulders, and each hand relaxed at the
    side (relaxed_hand), whatever the sign's handshape. Lexicon clips are trimmed to the hands-up span, so without this a
    video starts and ends mid-air. (Keeping the sign's handshape turned 180° left a fist or a spread hand, twisted, at
    the avatar's side before and after every sentence.)"""
    f = frame.copy()
    sw = float(np.linalg.norm(f[R_SH] - f[L_SH])) or 1.0
    for sh, el, wr, hand in ((R_SH, R_EL, R_WR, RH), (L_SH, L_EL, L_WR, LH)):
        side = np.sign(f[sh, 0]) or 1.0
        out = 0.1 * side  # neck-centred: a little away from the body, on the shoulder's side
        f[el] = f[sh] + np.array([out, 1.1, 0], np.float32)
        f[wr] = f[el] + np.array([0, 1.0, 0], np.float32)
        palm = float(np.linalg.norm(frame[hand][9] - frame[hand][0]))
        palm = float(np.clip(palm, 0.25 * sw, 0.6 * sw)) if palm > 1e-6 else 0.4 * sw
        f[hand] = f[wr] + relaxed_hand(palm, medial=-side)
    return f


def relax_collapsed_hands(kp: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """An undetected hand is stored collapsed onto its wrist (pin_collapsed_hands), which the avatar renders as a
    crumpled fist at the hip. Give such frames a relaxed hanging hand at the wrist instead."""
    out = kp.copy()
    for t in range(len(out)):
        f = out[t]
        sw = float(np.linalg.norm(f[R_SH] - f[L_SH])) or 1.0
        for sh, wr, hand in ((R_SH, R_WR, RH), (L_SH, L_WR, LH)):
            if np.ptp(f[hand, :2], axis=0).max() < eps * sw:
                f[hand] = f[wr] + relaxed_hand(0.4 * sw, medial=-(np.sign(f[sh, 0]) or 1.0))
    return out


class MissingAwarePoser:
    def __init__(self, lexicon, n_transition: int = 8):
        self.lexicon, self.n = lexicon, n_transition

    def pose(self, result: GlossResult) -> tuple[np.ndarray, list[Segment]]:
        clips = []
        for item in result.items:
            e = None if item.oov else self.lexicon.lookup(item.text)
            if e is not None:
                kp = self.lexicon.keypoints(e)
                if kp is not None:
                    kp = relax_collapsed_hands(np.nan_to_num(kp, nan=0.0).astype(np.float32))
                clips.append((e.gloss, "sign", kp))
            else:
                clips.append((item.text, "missing", None))
        if not any(kind == "sign" for _, kind, _ in clips):
            raise ValueError("nothing in the lexicon to sign")
        rest = next(kp[0] for _, kind, kp in clips if kind == "sign")
        parts, segs, t = [], [], 0
        for label, kind, kp in clips:
            if kp is None:  # hold the current pose (or the first sign's start) for a missing sign
                kp = np.repeat((parts[-1][-1] if parts else rest)[None], MISSING_FRAMES, axis=0)
            if parts and self.n > 0:
                tr = _transition(parts[-1][-1], kp[0], self.n)
                parts.append(tr)
                segs.append(Segment("", t, t + len(tr), "transition"))
                t += len(tr)
            parts.append(kp.astype(np.float32))
            segs.append(Segment(label, t, t + len(kp), kind))
            t += len(kp)
        # start from and return to hands-down, then hold, so the video has a clear beginning and end
        start, end = rest_pose(parts[0][0]), rest_pose(parts[-1][-1])
        n = max(self.n, 8)
        lead = np.concatenate([np.repeat(start[None], 5, axis=0), _transition(start, parts[0][0], n)])
        tail = np.concatenate([_transition(parts[-1][-1], end, n), np.repeat(end[None], REST_HOLD, axis=0)])
        segs = [Segment("", 0, len(lead), "transition")] + [Segment(s.label, s.start + len(lead), s.end + len(lead), s.kind)
                                                           for s in segs]
        segs.append(Segment("", t + len(lead), t + len(lead) + len(tail), "transition"))
        return np.concatenate([lead] + parts + [tail]).astype(np.float32), segs
