import pytest
from isharati.text.levels import classify_level


@pytest.mark.parametrize("text,level", [
    ("ما حكم صلاة المسافر؟", "D"),
    ("هل يجوز لي أن أجمع الصلاة؟", "D"),
    ("أفتوني في مسألتي", "D"),
    ("اختلف العلماء في هذه المسألة", "C"),
    ("ما القول الراجح في هذا؟", "C"),
    ("الوضوء قبل الصلاة", "B"),
])
def test_classify_level(text, level):
    assert classify_level(text) == level


def test_level_patterns_match_whole_words_only():
    assert classify_level("الحكمة والموعظة الحسنة") == "B"
    assert classify_level("المخالفة للنظام") == "B"


@pytest.mark.parametrize("text,level", [
    ("هل الموسيقى حرام؟", "D"),
    ("وهل يجوز لي ذلك", "D"),
    ("هل تجوز الصلاة هنا", "D"),
    ("هل يحرم هذا", "D"),
    ("بحكم الشرع", "D"),
    ("في هذه المسألة الخلاف", "C"),
    ("اختلاف العلماء في ذلك", "C"),
])
def test_inflected_and_prefixed_fatwa_phrasings_are_referred(text, level):
    assert classify_level(text) == level
