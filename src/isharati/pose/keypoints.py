"""Pose contract [T,50,3]: 8 upper-body joints + 21 left-hand + 21 right-hand, neck-centred, shoulder-scaled."""
import numpy as np

from isharati.types import N_JOINTS

NOSE, NECK, R_SH, R_EL, R_WR, L_SH, L_EL, L_WR = range(8)
LH = slice(8, 29)
RH = slice(29, 50)


def interpolate_missing(raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Linearly interpolate NaNs over time per joint/coord; never-seen joints become 0.
    Returns (filled float32, ratio of frames in which the dominant hand was missing).
    The dominant hand is the one detected more often: in one-handed signs (letters, numbers) the resting
    hand is usually out of frame, and counting it would reject every one-handed sign."""
    x = np.asarray(raw, dtype=np.float32)
    T = x.shape[0]
    lh_missing = np.isnan(x[:, LH, :]).all(axis=(1, 2))
    rh_missing = np.isnan(x[:, RH, :]).all(axis=(1, 2))
    dominant = rh_missing if rh_missing.sum() <= lh_missing.sum() else lh_missing
    ratio = float(np.mean(dominant)) if T else 1.0
    out = x.copy()
    t = np.arange(T, dtype=np.float32)
    for j in range(N_JOINTS):
        for c in range(3):
            v = out[:, j, c]
            ok = ~np.isnan(v)
            if not ok.any():
                out[:, j, c] = 0.0
            elif not ok.all():
                out[:, j, c] = np.interp(t, t[ok], v[ok])
    return out, ratio


ARM_BONES = ((R_SH, R_EL), (R_EL, R_WR), (L_SH, L_EL), (L_EL, L_WR))


# the 2D length assumed for each arm bone at least, in shoulder widths (2D 90th percentiles over the dictionary
# sources: upper arm 0.83-1.32, forearm 0.68-0.94), for a clip in which a bone never lies flat to the camera
MIN_BONE = {(R_SH, R_EL): 0.8, (L_SH, L_EL): 0.8, (R_EL, R_WR): 0.7, (L_EL, L_WR): 0.7}


def _hands_to_arms(k, x, ref, med, near=0.5):
    """The body model loses a raised arm while the hand model still tracks the hand (shahada clip: the right hand 1-2.3
    shoulder widths from the right body wrist), and the avatar then joined a hand on one side to an arm on the other,
    across the face. In place: hands swapped between the arms are swapped back (k); where a detected hand is far from
    its own body wrist, the wrist moves to the hand's wrist and the elbow is rebuilt with both bones at their lengths
    (x: the 8 body joints, image axes, y down), bending the way it bent in the previous frame."""
    T = len(k)
    has = lambda base: np.abs(k[:, base:base + 21]).sum((1, 2)) > 1e-6  # noqa: E731
    # swapped: each hand sits on the other arm's wrist
    dLL = np.linalg.norm(k[:, 8, :2] - x[:, L_WR, :2], axis=-1); dLR = np.linalg.norm(k[:, 8, :2] - x[:, R_WR, :2], axis=-1)
    dRR = np.linalg.norm(k[:, 29, :2] - x[:, R_WR, :2], axis=-1); dRL = np.linalg.norm(k[:, 29, :2] - x[:, L_WR, :2], axis=-1)
    swap = has(8) & has(29) & (dLR < near * med) & (dRL < near * med) & (dLL > near * med) & (dRR > near * med)
    if swap.any():
        k[swap, 8:29], k[swap, 29:50] = k[swap, 29:50].copy(), k[swap, 8:29].copy()
    for sh, el, wr, base in ((R_SH, R_EL, R_WR, 29), (L_SH, L_EL, L_WR, 8)):
        r1, r2 = ref[(sh, el)], ref[(el, wr)]
        far = has(base) & (np.linalg.norm(k[:, base, :2] - x[:, wr, :2], axis=-1) > near * med)
        # frames before the first good one are rebuilt backwards from it, so a clip that starts with a lost arm
        # bends its elbow the way the following frames do instead of flicking at the start
        f0 = int(np.argmin(far)) if (~far).any() else T
        order = list(range(f0 - 1, -1, -1)) + list(range(f0, T))
        prev = x[f0, el, :2].copy() if f0 < T else None
        for t in order:
            if t == f0:
                prev = x[t, el, :2].copy()
            if not far[t]:
                prev = x[t, el, :2].copy()
                continue
            S, W = x[t, sh, :2], k[t, base, :2].astype(np.float32)
            x[t, wr, :2] = W
            v = W - S
            dist = float(np.linalg.norm(v))
            u = v / max(dist, 1e-6)
            E0 = x[t, el, :2]
            side0 = np.sign(S[0]) or 1.0
            if (E0[0] * side0 > 0.1 * med and np.linalg.norm(E0 - S) <= 1.35 * r1
                    and np.linalg.norm(W - E0) <= 1.35 * r2):            # the tracked elbow still fits: keep it
                E = E0
            elif dist >= r1 + r2 or dist <= abs(r1 - r2):            # out of reach / too close: along the line
                E = S + u * min(r1, dist) if dist >= r1 + r2 else x[t, el, :2]
            else:                                                    # two-circle intersection: the elbow's two choices
                a = (r1 ** 2 - r2 ** 2 + dist ** 2) / (2 * dist)
                h = np.sqrt(max(r1 ** 2 - a ** 2, 0.0))
                n = np.array([-u[1], u[0]], np.float32)
                c1, c2 = S + a * u + h * n, S + a * u - h * n
                # never the bend that crosses the body's midline (TİD «selam»: the wrong one put the elbow at the
                # chest's centre); of the rest, the one closest to the previous frame's elbow, else the outward one
                side = np.sign(S[0]) or 1.0
                ok = [c for c in (c1, c2) if c[0] * side > 0.1 * med] or [max((c1, c2), key=lambda c: c[0] * side)]
                ref_pt = prev if prev is not None else S + np.array([side * 0.5 * med, 0.8 * med], np.float32)
                E = min(ok, key=lambda c: np.linalg.norm(c - ref_pt))
            x[t, el, :2] = E
            prev = E.copy()


def repair_body(pose: np.ndarray) -> np.ndarray:
    """Body joints a tracker got wrong, repaired (frame count unchanged).

    Continuous recordings lose the body now and then: in the shahada clip of new_arabic_sources single frames flip a
    shoulder's depth (shoulder width 1.0 -> 1.84) and the body model loses the raised arm while the hand model still
    tracks the hand, which the avatar played as an arm thrown across the face. A bone looks shorter when it points at
    the camera but never longer, so its length is about the longest it appears in 2D:
      1. 2D outliers (shoulder width far from the clip's median; an elbow or wrist much farther from its parent than
         that length, or out and straight back in one frame) are dropped and re-interpolated over time;
      2. arms rejoin their hands (_hands_to_arms), and hands swapped between the arms are swapped back;
      3. depth spikes (a joint's depth away from its 5-frame median and back) take the median, and the shoulders'
         depth difference is capped (a twist, not a flip). MediaPipe's depth is exaggerated throughout, but the avatar
         is tuned to that scale (it halves and clamps it), so depth is otherwise left alone: rescaling it to rigid
         bones put a fist held out in front (KArSL «الزكاة») inside the avatar's chest.
    """
    k = np.asarray(pose, dtype=np.float32).copy()
    T = len(k)
    if T < 3:
        return k
    x = k[:, :8].copy()
    seen = np.abs(x).sum(-1) > 1e-6                                 # a joint stored as 0 was never detected
    sw2 = np.linalg.norm(x[:, R_SH, :2] - x[:, L_SH, :2], axis=-1)
    if (sw2 > 1e-6).sum() < 3:
        return k
    med = float(np.median(sw2[sw2 > 1e-6]))
    bad = np.zeros((T, 8), bool)
    bad[:, [R_SH, L_SH]] |= ((sw2 < 0.75 * med) | (sw2 > 1.3 * med))[:, None]
    ref = {}
    for a, b in ARM_BONES:
        L2 = np.linalg.norm(x[:, b, :2] - x[:, a, :2], axis=-1)
        ok = seen[:, a] & seen[:, b] & ~bad[:, a]
        ref[(a, b)] = max(float(np.percentile(L2[ok], 90)) if ok.sum() >= 3 else 0.0, MIN_BONE[(a, b)] * med)
        bad[:, b] |= ok & (L2 > 1.25 * ref[(a, b)])
    # an elbow past the body's midline while its wrist stays out on its own side: the tracker lost the elbow
    # (TİD «selam», first frames: the right elbow at x 0.03 with the shoulder at -0.49), which swung the arm across
    # the chest as the sign began. The other edge check (out and back in one frame) cannot see it at a clip's start.
    for sh, el, wr in ((R_SH, R_EL, R_WR), (L_SH, L_EL, L_WR)):
        side = np.sign(np.median(x[:, sh, 0][seen[:, sh]])) if seen[:, sh].any() else 0.0
        if side:
            bad[:, el] |= seen[:, el] & (x[:, el, 0] * side < 0.1 * med) & (x[:, wr, 0] * side > 0.4 * med)
    for j in (R_EL, R_WR, L_EL, L_WR):                              # one-frame excursions: out and straight back
        p = x[:, j, :2]
        mid = 0.5 * (p[:-2] + p[2:])
        away = np.linalg.norm(p[1:-1] - mid, axis=-1)
        span = np.linalg.norm(p[2:] - p[:-2], axis=-1)
        bad[1:-1, j] |= (away > 0.5 * med) & (span < 0.5 * away)
    if bad.any():
        y = x.copy()
        y[bad | ~seen] = np.nan
        filled, _ = interpolate_missing(np.concatenate([y, np.full((T, N_JOINTS - 8, 3), np.nan, np.float32)], 1))
        x = np.where(bad[..., None], filled[:, :8], x)
    _hands_to_arms(k, x, ref, med)
    # 3. depth spikes -> the 5-frame median; shoulders twisted by at most what a turned torso shows
    if T >= 5:
        z = np.pad(x[:, :, 2], ((2, 2), (0, 0)), mode="edge")
        zmed = np.median(np.stack([z[i:i + T] for i in range(5)]), axis=0)
        spike = np.abs(x[:, :, 2] - zmed) > 0.4 * med
        x[:, :, 2] = np.where(spike, zmed, x[:, :, 2])
    mz = 0.5 * (x[:, R_SH, 2] + x[:, L_SH, 2])
    half = np.clip(x[:, R_SH, 2] - x[:, L_SH, 2], -0.6 * med, 0.6 * med) / 2
    x[:, R_SH, 2], x[:, L_SH, 2] = mz + half, mz - half
    k[:, :8] = np.where(seen[..., None], x, k[:, :8])  # (_hands_to_arms may also have swapped k's hands)
    return k


def pin_collapsed_hands(pose: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """A never-detected hand is stored as 21 joints at one point (the zero fill, shifted by normalisation), which
    renders as a stray limb. Pin such a hand to its wrist per frame: it reads as a resting hand, not a glitch."""
    out = np.array(pose, dtype=np.float32, copy=True)
    for hand, wrist in ((LH, L_WR), (RH, R_WR)):
        h = out[:, hand, :]
        collapsed = (h.max(axis=1) - h.min(axis=1)).max(axis=-1) < eps
        out[collapsed, hand, :] = out[collapsed, wrist:wrist + 1, :]
    return out


def normalize(filled: np.ndarray) -> np.ndarray:
    """Centre every frame on the neck; divide by the clip's median shoulder width."""
    x = np.asarray(filled, dtype=np.float32)
    widths = np.linalg.norm(x[:, R_SH] - x[:, L_SH], axis=-1)
    widths = widths[np.isfinite(widths) & (widths > 1e-6)]
    if widths.size == 0:
        raise ValueError("no shoulders detected in clip")
    scale = float(np.median(widths))
    return ((x - x[:, NECK:NECK + 1, :]) / scale).astype(np.float32)


# ---- extraction (MediaPipe Holistic) ---------------------------------------
from dataclasses import dataclass  # noqa: E402

from isharati.types import FPS  # noqa: E402

_BODY_IDS = [(NOSE, 0), (R_SH, 12), (R_EL, 14), (R_WR, 16), (L_SH, 11), (L_EL, 13), (L_WR, 15)]


@dataclass
class ExtractResult:
    pose: np.ndarray
    missing_ratio: float
    ok: bool


def frame_from_holistic(results) -> np.ndarray:
    """One MediaPipe Holistic result -> [50,3] with NaN for anything not detected."""
    f = np.full((N_JOINTS, 3), np.nan, np.float32)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        for j, mp_id in _BODY_IDS:
            f[j] = (lm[mp_id].x, lm[mp_id].y, lm[mp_id].z)
        f[NECK] = (f[R_SH] + f[L_SH]) / 2
    for hand, sl in ((results.left_hand_landmarks, LH), (results.right_hand_landmarks, RH)):
        if hand:
            f[sl] = [(p.x, p.y, p.z) for p in hand.landmark]
    return f


def extract(video_path, fps: int = FPS) -> ExtractResult:
    """Video -> normalised pose at `fps`. ok=False (pose all zeros) when no body is ever detected."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    return _pose_from_frames(frames, src_fps, fps)


def frame_paths(frame_dir) -> list:
    """Image files in natural order by their last number (1, 2, 10 — not 1, 10, 2)."""
    import re
    from pathlib import Path

    def key(p):
        nums = re.findall(r"\d+", p.stem)
        return (int(nums[-1]) if nums else -1, p.name)

    return sorted((p for p in Path(frame_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")), key=key)


def extract_image_dir(frame_dir, src_fps: float = 30.0, fps: int = FPS) -> ExtractResult:
    """A directory of sequential frame images (e.g. KArSL on Kaggle) -> normalised pose at `fps`.
    Unreadable images are skipped; a directory with none readable gives ok=False."""
    import cv2

    frames = []
    for p in frame_paths(frame_dir):
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return _pose_from_frames(frames, src_fps, fps)


def _pose_from_frames(frames, src_fps, fps) -> ExtractResult:
    import mediapipe as mp

    if not frames:
        return ExtractResult(np.zeros((1, N_JOINTS, 3), np.float32), 1.0, False)
    n_out = max(1, int(round(len(frames) * fps / src_fps)))
    picks = np.linspace(0, len(frames) - 1, n_out).round().astype(int)
    raw = []
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holo:
        for i in picks:
            raw.append(frame_from_holistic(holo.process(frames[i])))
    raw = np.stack(raw)
    body_seen = ~np.isnan(raw[:, R_SH, 0])
    filled, ratio = interpolate_missing(raw)
    if not body_seen.any():
        return ExtractResult(np.zeros_like(filled), 1.0, False)
    return ExtractResult(normalize(filled), ratio, True)
