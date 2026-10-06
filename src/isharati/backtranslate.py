"""Back-translation: recognise each generated sign segment and compare with the source Gloss sequence.
The M2 baseline is DTW 1-nearest-neighbour; Plan 2 replaces it with a fine-tuned recogniser."""
import random
from pathlib import Path

import numpy as np

from isharati.pose.dtw import dtw_distance
from isharati.lexicon.arabic import Lexicon
from isharati.types import BTResult


def wer(ref: list[str], hyp: list[str]) -> float:
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return d[m] / n


class DTWRecognizer:
    def __init__(self, templates: list[tuple[str, np.ndarray]], name: str):
        if not templates:
            raise ValueError("recognizer needs at least one template")
        self.templates, self.name = templates, name

    @classmethod
    def from_lexicon(cls, lexicon: Lexicon) -> "DTWRecognizer":
        return cls([(e.gloss, lexicon.keypoints(e)) for e in lexicon.sign_entries()], "dtw-1nn-lexicon (circular)")

    @classmethod
    def from_templates(cls, path: Path) -> "DTWRecognizer":
        lex = Lexicon.load(path)
        return cls([(e.gloss, lex.keypoints(e)) for e in lex.sign_entries()], "dtw-1nn-heldout-signer")

    def predict(self, seg_pose: np.ndarray) -> str:
        return min(self.templates, key=lambda t: dtw_distance(seg_pose, t[1]))[0]


def score(pose, segments, source_glosses: list[str], recognizer) -> BTResult:
    rec = [recognizer.predict(pose[s.start:s.end]) for s in segments if s.kind == "sign"]
    hits = sum(r == s for r, s in zip(rec, source_glosses))
    acc = hits / len(source_glosses) if source_glosses else 0.0
    return BTResult(recognized=rec, gloss_acc=acc, wer=wer(source_glosses, rec))


class SampledRecognizer:
    """For lexicons without held-out templates and too large for 1-NN over every sign (Urdu ISL has 8,000): the
    answer's own signs plus a fixed sample of distractors. Circular, like from_lexicon, and labelled so."""

    def __init__(self, lexicon, n_distractors: int = 200, seed: int = 0):
        entries = sorted(lexicon.sign_entries(), key=lambda e: e.sign_id)
        self.lexicon = lexicon
        self.distractors = random.Random(seed).sample(entries, min(n_distractors, len(entries)))

    def for_glosses(self, glosses) -> "DTWRecognizer":
        chosen = {e.sign_id: e for e in self.distractors}
        for g in glosses:
            e = self.lexicon.lookup(g)
            if e:
                chosen[e.sign_id] = e
        return DTWRecognizer([(e.gloss, self.lexicon.keypoints(e)) for e in chosen.values()],
                             "dtw-1nn-lexicon-sample (circular)")
