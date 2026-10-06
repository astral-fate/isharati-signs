"""Retrieval-stitching Gloss -> Pose (the Hybrid backbone). Transitions use smoothstep interpolation."""
import numpy as np

from isharati.types import GlossResult, Segment


def _transition(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    s = np.arange(1, n + 1, dtype=np.float32) / (n + 1)
    w = (s * s * (3 - 2 * s))[:, None, None]
    return (a[None] * (1 - w) + b[None] * w).astype(np.float32)


class StitchPoser:
    def __init__(self, lexicon, n_transition: int = 8):
        self.lexicon, self.n = lexicon, n_transition

    def _clips(self, result: GlossResult):
        clips = []
        for item in result.items:
            if item.oov:
                for e in self.lexicon.fingerspell(item.text):
                    clips.append((e.gloss, "fingerspell", self.lexicon.keypoints(e)))
            else:
                e = self.lexicon.lookup(item.text)
                if e is None:
                    raise KeyError(f"gloss not in lexicon: {item.text}")
                clips.append((e.gloss, "sign", self.lexicon.keypoints(e)))
        return clips

    def pose(self, result: GlossResult) -> tuple[np.ndarray, list[Segment]]:
        clips = self._clips(result)
        if not clips:
            raise ValueError("nothing to sign")
        parts, segs, t = [], [], 0
        for i, (label, kind, kp) in enumerate(clips):
            if i > 0 and self.n > 0:
                tr = _transition(parts[-1][-1], kp[0], self.n)
                parts.append(tr)
                segs.append(Segment("", t, t + len(tr), "transition"))
                t += len(tr)
            parts.append(kp)
            segs.append(Segment(label, t, t + len(kp), kind))
            t += len(kp)
        return np.concatenate(parts).astype(np.float32), segs
