import numpy as np
import pytest
from isharati.pose.dtw import dtw_distance, resample


def _curve(T, phase=0.0):
    t = np.linspace(0, 1, T, dtype=np.float32)
    return np.sin(2 * np.pi * (t[:, None, None] + phase)) * np.ones((1, 50, 3), np.float32)


def test_resample_length():
    assert resample(_curve(10), 32).shape == (32, 50, 3)


def test_dtw_identical_is_zero():
    a = _curve(20)
    assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_dtw_time_warp_is_closer_than_a_different_motion():
    a = _curve(20)
    slow = np.repeat(a, 2, axis=0)
    other = _curve(20, phase=0.4)
    assert dtw_distance(a, slow) < dtw_distance(a, other)
