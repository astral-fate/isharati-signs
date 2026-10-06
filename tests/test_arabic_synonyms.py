"""Reviewed Arabic synonyms (src/isharati/lexicon/synonyms_ar.json): a word without its own sign finds the sign of the
same meaning, with or without clitics, and never displaces a word's own sign or the matcher's guards."""
import pytest


@pytest.fixture(scope="module")
def lex():
    from isharati.pipeline import load_lexicon
    try:
        return load_lexicon("ar")
    except FileNotFoundError:
        pytest.skip("Arabic lexicon data not present")


@pytest.mark.parametrize("word, gloss", [("نكاح", "زواج"), ("النكاح", "زواج"), ("ونكاح", "زواج"),
                                         ("الزوجة", "زوج"), ("زوجته", "زوج"), ("بالزوجة", "زوج")])
def test_synonym_finds_the_sign_of_the_same_meaning(lex, word, gloss):
    assert lex.match_token(word).gloss == gloss


def test_synonyms_never_override_own_signs_or_guards(lex):
    assert lex.match_token("زوج").gloss == "زوج" and lex.match_token("زواج").gloss == "زواج"
    assert lex.match_token("الجنة").gloss != "جن"           # a final ة is still never stripped by the matcher
    assert lex.match_token("إسلام").gloss != "سلام"        # nor an initial alef
    assert all(target in lex._signs for target in lex._aliases.values())


@pytest.mark.parametrize("word, gloss", [("تصوم", "صوم"), ("وتصوم", "صوم"), ("يصومون", "صوم"), ("تشهد", "شهادة"),
                                         ("تحج", "حج"), ("وتؤتي", "أعطى")])
def test_every_conjugation_of_a_reviewed_verb_finds_its_sign(lex, word, gloss):
    assert lex.match_token(word).gloss == gloss
