"""Dynamic time warping over pose sequences (used for the Lexicon medoid and the baseline recogniser)."""
import numpy as np


def resample(x: np.ndarray, n: int) -> np.ndarray:
    """Linear resampling along time to exactly n frames."""
    T = x.shape[0]
    if T == n:
        return x.astype(np.float32)
    src = np.linspace(0, T - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None].astype(np.float32)
    return (x[lo] * (1 - w) + x[hi] * w).astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray, max_len: int = 32) -> float:
    """Path-length-normalised DTW with Euclidean frame cost; sequences longer than max_len are resampled."""
    if len(a) > max_len:
        a = resample(a, max_len)
    if len(b) > max_len:
        b = resample(b, max_len)
    A = a.reshape(len(a), -1)
    B = b.reshape(len(b), -1)
    cost = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))
