import numpy as np
import pytest
from isharati.lexicon.arabic import MissingLetterSign


def test_lookup_is_normalised(lexicon):
    assert lexicon.lookup("صَلاة").sign_id == "r5"
    assert lexicon.lookup("اسلام").sign_id == "r6"   # hamza folded
    assert lexicon.lookup("كعب") is None


def test_match_token_uses_light_stemming(lexicon):
    assert lexicon.match_token("والصلاة").gloss == "صلاة"
    assert lexicon.match_token("وجهك").gloss == "وجه"
    assert lexicon.match_token("امسح") is None


def test_fingerspell_letters_in_order(lexicon):
    assert [e.gloss for e in lexicon.fingerspell("مِرفق")] == ["م", "ر", "ف", "ق"]


@pytest.mark.parametrize("word", ["abc", "3", "ﷺ"])
def test_fingerspell_raises_on_characters_without_letter_sign(lexicon, word):
    with pytest.raises(MissingLetterSign):
        lexicon.fingerspell(word)


def test_keypoints_load(lexicon):
    kp = lexicon.keypoints(lexicon.lookup("وجه"))
    assert kp.shape == (20, 50, 3) and kp.dtype == np.float32


def test_sign_entries_and_keys_exclude_letters(lexicon):
    assert all(not e.is_letter for e in lexicon.sign_entries())
    assert "وجه" in lexicon.gloss_keys()
    assert "م" not in lexicon.gloss_keys()


def test_keypoints_pin_collapsed_hands(lexicon_dir):
    import numpy as np
    from isharati.pose.keypoints import LH, L_WR
    from isharati.lexicon.arabic import Lexicon
    clip = np.load(lexicon_dir / "signs" / "r1.npy")
    clip[:, LH, :] = 3.0
    np.save(lexicon_dir / "signs" / "r1.npy", clip)
    kp = Lexicon.load(lexicon_dir / "lexicon.jsonl").keypoints(Lexicon.load(lexicon_dir / "lexicon.jsonl").lookup("وضوء"))
    assert np.allclose(kp[:, LH, :], kp[:, L_WR:L_WR + 1, :])
