"""One body shape across sign sources.

Pose clips keep MediaPipe's normalised image coordinates, so each video's aspect ratio stretches the body: the
nose-to-neck height over shoulder width is 0.53 in KArSL, 0.84 in Tawasol, 0.74 in ASL Citizen, 1.10 in Obscure ASL
and 0.28 in Isharah's pixel data, where real bodies vary by perhaps 15 %. Stitched together, the head jumps at every
cut from one source to another. Each clip's height is rescaled about the neck so its median ratio is REF; timing,
handshapes and relative motion are untouched.
"""
import numpy as np

from isharati.pose.keypoints import L_SH, NECK, NOSE, R_SH

# A real body's nose-to-neck height over shoulder width, in square units: 0.47-0.55 (median ~0.50) on five signers'
# datasets after their x was scaled by the frame aspect (scripts/hub/publish_quran_sign.square_units). The old value,
# 0.74, was ASL Citizen's median in raw 4:3 webcam coordinates, so every lexicon clip was normalised to a body ~1.5x
# too tall: hands sat too low and too far from the shoulder, out of the realistic avatars' reach.
REF = 0.50


def ratio(clip: np.ndarray) -> float:
    head = np.abs(clip[:, NOSE, 1] - clip[:, NECK, 1])
    width = np.linalg.norm(clip[:, R_SH, :2] - clip[:, L_SH, :2], axis=-1)
    ok = np.isfinite(head) & np.isfinite(width) & (width > 1e-6)
    return float(np.median(head[ok] / width[ok])) if ok.any() else REF


def fix(clip: np.ndarray, ref: float = REF) -> np.ndarray:
    """The clip with its height scaled about the neck so that ratio(clip) == ref."""
    r = ratio(clip)
    if not np.isfinite(r) or r <= 0:
        return clip
    out = np.array(clip, np.float32, copy=True)
    neck_y = out[:, NECK:NECK + 1, 1]
    out[..., 1] = neck_y + (out[..., 1] - neck_y) * (ref / r)
    return out


def apply(lexicon):
    """Make lexicon.keypoints return proportion-fixed clips (cached), for every source alike."""
    original, cache = lexicon.keypoints, {}

    def keypoints(entry):
        if entry.sign_id not in cache:
            cache[entry.sign_id] = fix(original(entry))
        return cache[entry.sign_id]
    lexicon.keypoints = keypoints
    return lexicon
