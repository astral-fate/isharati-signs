from isharati.text.arabic import normalize_ar, strip_diacritics, candidates
from isharati.types import GlossItem, GlossResult


def test_normalize_removes_diacritics_and_folds_letters():
    assert normalize_ar("الصَّلاةُ") == "الصلاه"
    assert normalize_ar("إسلام") == "اسلام"


def test_normalize_handles_tatweel_digits_and_latin():
    assert normalize_ar("وضـــوء ٣ times!") == "وضوء 3"


def test_normalize_empty_and_none():
    assert normalize_ar("") == ""
    assert normalize_ar(None) == ""
    assert normalize_ar("   ") == ""


def test_strip_diacritics_keeps_letter_shapes():
    assert strip_diacritics("مِرْفَق") == "مرفق"
    assert strip_diacritics("صلاة") == "صلاة"


def test_candidates_strip_prefixes_and_suffixes():
    assert candidates("والصلاه")[0] == "والصلاه"
    assert "صلاه" in candidates("والصلاه")
    assert "وجه" in candidates("وجهك")
    assert "يد" in candidates("يديك")
    assert "غسل" in candidates("اغسل")


def test_gloss_result_properties():
    r = GlossResult([GlossItem("وجه"), GlossItem("امسح", oov=True)], "rule")
    assert r.glosses == ["وجه"]
    assert r.oov == ["امسح"]


def test_candidates_strip_at_most_one_prefix():
    c = candidates("الوالدين")
    assert "دين" not in c          # الوالدين must never reach "religion"
    assert "والد" in c
