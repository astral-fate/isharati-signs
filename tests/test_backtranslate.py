import pytest
from isharati.backtranslate import DTWRecognizer, score, wer
from isharati.pose.stitch import StitchPoser
from isharati.types import GlossItem, GlossResult


def test_wer_basic_cases():
    assert wer(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(1 / 3)
    assert wer([], []) == 0.0
    assert wer(["a"], []) == 1.0
    assert wer(["a", "b"], ["b"]) == pytest.approx(0.5)


def test_recognizer_identifies_its_own_template(lexicon):
    r = DTWRecognizer.from_lexicon(lexicon)
    assert r.predict(lexicon.keypoints(lexicon.lookup("وجه"))) == "وجه"
    assert "circular" in r.name


def test_score_on_stitched_pose_ignores_transitions_and_fingerspelling(lexicon):
    gr = GlossResult([GlossItem("وجه"), GlossItem("مرفق", oov=True), GlossItem("يد")], "t")
    pose, segs = StitchPoser(lexicon).pose(gr)
    bt = score(pose, segs, gr.glosses, DTWRecognizer.from_lexicon(lexicon))
    assert bt.recognized == ["وجه", "يد"]
    assert bt.gloss_acc == 1.0 and bt.wer == 0.0


def test_score_with_no_sign_segments_is_zero_accuracy(lexicon):
    gr = GlossResult([GlossItem("مرفق", oov=True)], "t")
    pose, segs = StitchPoser(lexicon).pose(gr)
    bt = score(pose, segs, gr.glosses, DTWRecognizer.from_lexicon(lexicon))
    assert bt.recognized == [] and bt.gloss_acc == 0.0 and bt.wer == 0.0


def test_from_templates_loads_heldout_file(lexicon_dir):
    import shutil
    shutil.copy(lexicon_dir / "lexicon.jsonl", lexicon_dir / "templates.jsonl")
    r = DTWRecognizer.from_templates(lexicon_dir / "templates.jsonl")
    assert r.name == "dtw-1nn-heldout-signer"
    assert len(r.templates) == 8
