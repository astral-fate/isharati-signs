"""English normalisation and light lemmatisation for matching words to ASL glosses."""
import re
import unicodedata

# ASL drops articles, the copula and most function words; the sign order carries them
STOPWORDS = set("""a an the is are am was were be been being to of and or but so that this these those it its
there their they them he him his she her we us our you your i me my do does did done has have had having will would
shall should can could may might must as at by for from in into on onto with about than then also very just
which who whom whose what when where why how if""".split())

IRREGULAR = {"is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "am": "be", "has": "have",
             "had": "have", "did": "do", "does": "do", "done": "do", "went": "go", "gone": "go", "made": "make",
             "said": "say", "gave": "give", "given": "give", "came": "come", "knew": "know", "known": "know",
             "taught": "teach", "believed": "believe", "children": "child", "people": "person", "men": "man",
             "women": "woman", "better": "good", "best": "good", "prayed": "pray", "fasted": "fast"}


def normalize_en(text: str | None) -> str:
    """Lookup key: lower case, letters and digits only, single spaces. Transliterated names keep their letters:
    accents are folded and the ‘ ’ ʿ ʾ marks inside a word are dropped («Qur’ān» -> quran, «‘Ā’ishah» -> aishah)."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"(?<=\w)[‘’ʿʾ`](?=\w)|(?<![\w])[‘ʿʾ`](?=\w)", "", t)  # ayn/hamza marks, not quotes
    t = t.lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"(?<=[a-z])'(?=[a-z])", "", t)  # «Qur'an» -> quran, «don't» -> dont
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def candidates(token: str) -> list[str]:
    """The normalised token, then lemma guesses (irregular forms, plural and verb endings), longest first."""
    w = normalize_en(token)
    if not w:
        return []
    out = [w]
    if w in IRREGULAR:
        out.append(IRREGULAR[w])
    rules = [("ies", "y"), ("ied", "y"), ("ves", "f"), ("sses", "ss"), ("es", ""), ("s", ""), ("ing", ""),
             ("ing", "e"), ("ed", ""), ("ed", "e"), ("ly", "")]
    for suf, rep in rules:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            out.append(w[: -len(suf)] + rep)
    # doubled consonant before -ing / -ed: "praying" is fine, "getting" -> "get"
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) > len(suf) + 3 and w[-len(suf) - 1] == w[-len(suf) - 2]:
            out.append(w[: -len(suf) - 1])
    return list(dict.fromkeys(out))


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9'’]+", text or "") if t]
