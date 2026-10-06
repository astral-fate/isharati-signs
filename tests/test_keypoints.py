import numpy as np
import pytest
from isharati.pose.keypoints import interpolate_missing, normalize, NECK, R_SH, L_SH


def test_interpolate_fills_gaps_linearly():
    x = np.zeros((5, 50, 3), np.float32)
    x[:, :, 0] = np.arange(5, dtype=np.float32)[:, None]
    x[1:4, 10, :] = np.nan
    y, ratio = interpolate_missing(x)
    assert not np.isnan(y).any()
    assert y[2, 10, 0] == pytest.approx(2.0)


def test_all_hands_missing_gives_zero_hands_and_ratio_one():
    x = np.random.default_rng(0).random((4, 50, 3)).astype(np.float32)
    x[:, 8:, :] = np.nan
    y, ratio = interpolate_missing(x)
    assert ratio == 1.0
    assert not np.isnan(y).any()
    assert np.all(y[:, 8:, :] == 0)


def test_ratio_counts_frames_with_the_dominant_hand_missing():
    x = np.zeros((4, 50, 3), np.float32)
    x[:, 8:29, :] = np.nan    # resting (left) hand never in frame
    x[0, 29:50, :] = np.nan   # frame 0: the signing (right) hand is missing too
    _, ratio = interpolate_missing(x)
    assert ratio == pytest.approx(0.25)


def test_one_handed_sign_with_resting_hand_never_seen_is_not_missing():
    x = np.zeros((4, 50, 3), np.float32)
    x[:, 8:29, :] = np.nan    # left hand out of frame for the whole clip (one-handed letter/number sign)
    _, ratio = interpolate_missing(x)
    assert ratio == 0.0


def test_normalize_centers_on_neck_and_scales_by_shoulders():
    x = np.zeros((3, 50, 3), np.float32)
    x[:, R_SH] = [1, 0, 0]
    x[:, L_SH] = [3, 0, 0]
    x[:, NECK] = [2, 0, 0]
    y = normalize(x)
    assert np.allclose(y[:, NECK], 0)
    assert np.allclose(np.linalg.norm(y[:, R_SH] - y[:, L_SH], axis=-1), 1)
    assert y.dtype == np.float32


def test_normalize_rejects_missing_shoulders():
    with pytest.raises(ValueError):
        normalize(np.zeros((3, 50, 3), np.float32))


def test_collapsed_hand_is_pinned_to_its_wrist():
    from isharati.pose.keypoints import L_WR, LH, R_WR, RH, pin_collapsed_hands
    x = np.random.default_rng(1).normal(size=(3, 50, 3)).astype(np.float32)
    x[:, LH, :] = -0.7                      # never-seen left hand: 21 joints at one point, away from the body
    y = pin_collapsed_hands(x)
    assert np.allclose(y[:, LH, :], y[:, L_WR:L_WR + 1, :])
    assert np.array_equal(y[:, RH, :], x[:, RH, :])   # a real hand is untouched
    assert np.array_equal(y[:, :8], x[:, :8]) and x[0, 8, 0] == np.float32(-0.7)   # body kept, input not mutated
