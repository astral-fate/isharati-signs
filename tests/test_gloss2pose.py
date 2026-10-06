import numpy as np
import pytest
from isharati.pose.stitch import StitchPoser
from isharati.lexicon.arabic import MissingLetterSign
from isharati.types import GlossItem, GlossResult


def _gr(*items):
    return GlossResult(list(items), "test")


def test_stitch_concatenates_signs_with_transition(lexicon):
    pose, segs = StitchPoser(lexicon, n_transition=8).pose(_gr(GlossItem("وجه"), GlossItem("يد")))
    assert pose.shape == (20 + 8 + 20, 50, 3) and pose.dtype == np.float32
    assert [s.kind for s in segs] == ["sign", "transition", "sign"]
    assert [s.label for s in segs if s.kind == "sign"] == ["وجه", "يد"]
    assert segs[0].start == 0 and segs[-1].end == len(pose)
    assert all(a.end == b.start for a, b in zip(segs, segs[1:]))


def test_transition_is_smooth(lexicon):
    n = 8
    pose, segs = StitchPoser(lexicon, n_transition=n).pose(_gr(GlossItem("وجه"), GlossItem("يد")))
    A = lexicon.keypoints(lexicon.lookup("وجه"))[-1]
    B = lexicon.keypoints(lexicon.lookup("يد"))[0]
    t = segs[1]
    window = pose[t.start - 1: t.end + 1]
    steps = np.abs(np.diff(window, axis=0))
    bound = 1.5 * np.abs(B - A) / (n + 1) + 1e-5
    assert np.all(steps <= bound[None])


def test_oov_word_is_fingerspelled(lexicon):
    pose, segs = StitchPoser(lexicon).pose(_gr(GlossItem("مرفق", oov=True)))
    assert [s.label for s in segs if s.kind == "fingerspell"] == ["م", "ر", "ف", "ق"]
    assert not np.isnan(pose).any()


def test_missing_letter_sign_is_raised_not_skipped(lexicon):
    with pytest.raises(MissingLetterSign):
        StitchPoser(lexicon).pose(_gr(GlossItem("وجه"), GlossItem("abc", oov=True)))


def test_empty_result_raises(lexicon):
    with pytest.raises(ValueError, match="nothing to sign"):
        StitchPoser(lexicon).pose(_gr())
