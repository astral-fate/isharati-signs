"""Arabic text normalisation shared by lexicon lookup and glossing."""
import re

_TASHKEEL = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ALEF = re.compile("[آأإٱ]")
_NON = re.compile(r"[^ء-ي0-9\s]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ب", "ف", "ل")
_SUFFIXES = ("كم", "هم", "ها", "نا", "ين", "ون", "ات", "ان", "ك", "ه", "ي")
_VERB_PREFIXES = ("ا", "ي", "ت", "ن")


def strip_diacritics(text: str | None) -> str:
    """Remove tashkeel and tatweel only; letter shapes (ة, أ, ى ...) are kept for fingerspelling."""
    t = _TASHKEEL.sub("", text or "").replace(_TATWEEL, "")
    return re.sub(r"\s+", " ", t).strip()


def normalize_ar(text: str | None) -> str:
    """Lookup key: no diacritics, folded alef/yaa/taa-marbuta/hamza seats, Latin digits, no punctuation or Latin."""
    t = strip_diacritics(text)
    t = _ALEF.sub("ا", t).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = t.translate(_AR_DIGITS)
    t = _NON.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def candidates(token: str) -> list[str]:
    """The normalised token, then light-stemmed variants, longest (least stripped) first.
    At most ONE prefix (clitic or verb prefix) is removed, so «الوالدين» can never be reduced to «دين»."""
    base = normalize_ar(token)
    if not base:
        return []
    seen: dict[str, bool] = {base: False}       # stem -> a prefix was already removed
    frontier = [(base, False)]
    for _ in range(2):
        nxt = []
        for w, prefixed in frontier:
            variants = []
            if not prefixed:
                variants += [(w[len(p):], True) for p in _PREFIXES if w.startswith(p) and len(w) - len(p) >= 2]
                if len(w) >= 4 and w[0] in _VERB_PREFIXES:
                    variants.append((w[1:], True))
            variants += [(w[: -len(s)], prefixed) for s in _SUFFIXES if w.endswith(s) and len(w) - len(s) >= 2]
            for v, pf in variants:
                if v not in seen:
                    seen[v] = pf
                    nxt.append((v, pf))
        frontier = nxt
    rest = sorted((w for w in seen if w != base), key=len, reverse=True)
    return [base] + rest
