import numpy as np

from isharati.pose.keypoints import L_EL, L_SH, L_WR, R_EL, R_SH, R_WR, repair_body


def _clip(T=20):
    """A still signer: arms hanging, hands at their wrists (image axes, y down, shoulder width 1)."""
    k = np.zeros((T, 50, 3), np.float32)
    k[:, R_SH] = [-0.5, 0, 0]; k[:, L_SH] = [0.5, 0, 0]
    k[:, R_EL] = [-0.6, 0.8, 0]; k[:, L_EL] = [0.6, 0.8, 0]
    k[:, R_WR] = [-0.6, 1.5, 0]; k[:, L_WR] = [0.6, 1.5, 0]
    k[:, 29:50] = k[:, [R_WR]] + np.linspace(0, 0.2, 21)[None, :, None] * [0, 1, 0]
    k[:, 8:29] = k[:, [L_WR]] + np.linspace(0, 0.2, 21)[None, :, None] * [0, 1, 0]
    return k


def test_clean_clip_is_unchanged():
    k = _clip()
    assert np.allclose(repair_body(k), k)


def test_shoulder_depth_flip_is_removed():
    k = _clip()
    k[10, R_SH, 2] = -0.9                       # one frame flips a shoulder toward the camera
    out = repair_body(k)
    assert abs(out[10, R_SH, 2]) < 0.1


def test_lost_arm_rejoins_its_hand():
    k = _clip()
    k[5:15, 29:50] += [0.0, -1.4, 0]            # the hand raised by the face; the body model kept the arm down
    out = repair_body(k)
    gap = np.linalg.norm(out[5:15, 29, :2] - out[5:15, R_WR, :2], axis=-1)
    assert gap.max() < 1e-3                       # the wrist follows the hand
    upper = np.linalg.norm(out[5:15, R_EL, :2] - out[5:15, R_SH, :2], axis=-1)
    assert np.all(upper < 1.05)                   # the rebuilt elbow keeps the arm's length


def test_swapped_hands_are_swapped_back():
    k = _clip()
    k[8, 8:29], k[8, 29:50] = k[8, 29:50].copy(), k[8, 8:29].copy()
    out = repair_body(k)
    assert np.allclose(out[8, 29:50], _clip()[8, 29:50])
