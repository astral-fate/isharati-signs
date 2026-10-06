"""Content-level scope guard (spec §5). C (disputed) and D (fatwa / personal case) are referred, never signed.

Matching is per token through the light stemmer, so clitics and the article (و، ف، ب، ال) do not hide a trigger.
Over-referral is the safe failure direction for this guard."""
from isharati.text.arabic import candidates, normalize_ar


class ReferralRequired(Exception):
    """The request is outside scope and must be referred to a qualified specialist."""


def _norm(words: str) -> tuple[str, ...]:
    return tuple(normalize_ar(w) for w in words.split())


# single tokens that on their own signal a ruling request / a disputed matter
_D_WORDS = {normalize_ar(w) for w in "حرام حلال يحرم يجوز تجوز يحل يصح افتوني افتني فتوى يفتي".split()}
_C_WORDS = {normalize_ar(w) for w in "خلاف اختلاف راجح".split()}
# multi-token phrases (each word matched through the stemmer, in order, adjacent)
_D_PHRASES = [_norm(p) for p in ("ما حكم", "هل علي", "يجب علي", "حكم شرع")]
_C_PHRASES = [_norm(p) for p in ("اختلف علماء", "قول راجح")]


def _stems(text: str) -> list[set[str]]:
    return [set(candidates(tok)) for tok in normalize_ar(text).split()]


def _has_word(stems, words) -> bool:
    return any(s & words for s in stems)


def _has_phrase(stems, phrase) -> bool:
    n = len(phrase)
    return any(all(phrase[k] in stems[i + k] for k in range(n)) for i in range(len(stems) - n + 1))


def classify_level(text: str) -> str:
    stems = _stems(text)
    if _has_word(stems, _D_WORDS) or any(_has_phrase(stems, p) for p in _D_PHRASES):
        return "D"
    if _has_word(stems, _C_WORDS) or any(_has_phrase(stems, p) for p in _C_PHRASES):
        return "C"
    return "B"
