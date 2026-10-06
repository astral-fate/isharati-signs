"""Gloss -> sign lookup over lexicon.jsonl, with letter signs for fingerspelling."""
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from isharati.text.arabic import candidates, normalize_ar, strip_diacritics
from isharati.pose.keypoints import pin_collapsed_hands, repair_body
from isharati.types import SignEntry


class MissingLetterSign(KeyError):
    """A character to fingerspell has no letter sign in the Lexicon."""


class Lexicon:
    def __init__(self, entries: list[SignEntry], root: Path):
        self.root = Path(root)
        self._signs = {normalize_ar(e.gloss): e for e in entries if not e.is_letter}
        self._letters = {e.gloss: e for e in entries if e.is_letter}
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        path = Path(path)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls([SignEntry(**r) for r in rows], path.parent)

    def lookup(self, gloss: str) -> SignEntry | None:
        return self._signs.get(normalize_ar(gloss))

    def match_token(self, token: str) -> SignEntry | None:
        for c in candidates(token):
            if c in self._signs:
                return self._signs[c]
        return None

    def fingerspell(self, word: str) -> list[SignEntry]:
        out = []
        for ch in strip_diacritics(word):
            if ch.isspace():
                continue
            if ch not in self._letters:
                raise MissingLetterSign(ch)
            out.append(self._letters[ch])
        return out

    def keypoints(self, entry: SignEntry) -> np.ndarray:
        if entry.keypoints_path not in self._cache:
            self._cache[entry.keypoints_path] = repair_body(pin_collapsed_hands(np.load(self.root / entry.keypoints_path)))
        return self._cache[entry.keypoints_path]

    def sign_entries(self) -> list[SignEntry]:
        return list(self._signs.values())

    def gloss_keys(self) -> list[str]:
        return sorted(self._signs, key=lambda k: (-len(k.split()), -len(k)))

    def to_rows(self) -> list[dict]:
        return [asdict(e) for e in list(self._signs.values()) + list(self._letters.values())]


# ---- the matcher used by the Arabic Q&A mode
from isharati.text.arabic import strip_diacritics as strip  # noqa: E402
from isharati.text import arabic_morph as morph  # noqa: E402
from isharati.text.arabic_morph import NUMBERS, is_verb, lemmas, singulars  # noqa: E402


def qa_lexicon(path):
    """The Arabic lexicon with two matching fixes for this mode (the Arabic app's matcher is left unchanged):
    a word also finds the «ال» form of its stem («وحج» -> «الحج»), a word starting with alef is never matched
    by stripping that alef («إسلام» must not become «سلام», peace), and a final «ة» is never stripped."""

    class ArabicQALexicon(Lexicon):
        def _match(self, token):
            bare = normalize_ar(token)
            for prefix in ("وبال", "وال", "بال", "ال", "وب", "و", "ب"):  # «الخمس», «بخمس», «والخمس» -> 5
                if bare.startswith(prefix) and bare[len(prefix):] in NUMBERS:
                    bare = bare[len(prefix):]
                    break
            number = NUMBERS.get(bare)
            if number is not None and self.lookup(number) is not None:  # «خمس», «خمسة» -> the sign «5»
                return self.lookup(number)
            forms = candidates(token)
            base = normalize_ar(token)
            if len(base) >= 4 and base.endswith("ا"):  # tanween alef: «محمدًا» -> «محمدا» -> «محمد»
                forms += [c for c in candidates(base[:-1]) if c not in forms]
            for c in list(forms):  # regular endings: plurals and feminine forms back to the singular
                forms += [s for s in singulars(c) if s not in forms]
            forms += [c for c in lemmas(token) if c not in forms]  # CAMeL lemma, cached; only if nothing matched yet
            if is_verb(token):  # «تكفر» (expiates) must not be stripped to «كفر» (disbelief): exact form only
                base = normalize_ar(token)
                forms = [base, base[1:]] if base.startswith("و") and len(base) > 3 else [base]
                for c in forms:
                    e = self.lookup(c)
                    if e is not None:
                        return e
                return None
            for c in forms:
                for form in (c, "ال" + c):
                    e = self.lookup(form)
                    if e is not None and form != c and not strip(e.gloss).startswith("ال"):
                        continue  # «وان» + «ال» is «ألوان» once hamza is folded: only a real article «ال» counts
                    if e is not None and self._keeps_alef(token, e) and self._long_enough(token, e):
                        return e
            e = super().match_token(token)
            return e if e is not None and self._keeps_alef(token, e) and self._long_enough(token, e) else None

        def _gloss_words(self):
            if not hasattr(self, "_gw"):
                self._gw = {w.removeprefix("ال") for e in self.sign_entries() for w in normalize_ar(e.gloss).split()}
            return self._gw

        def _long_enough(self, token, entry):
            """A two-letter sign is matched only when that really is the word: «أبي» -> أب, but «شرك» (polytheism)
            is not «شر» (evil)."""
            g = normalize_ar(entry.gloss).removeprefix("ال")
            if len(g) >= 3:
                return True
            w = normalize_ar(token)
            for prefix in ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل"):  # prefixes keep the word
                if w.startswith(prefix) and len(w) - len(prefix) >= 2:
                    w = w[len(prefix):]
                    break
            if w == g:  # only prefixes were removed
                return True
            # a suffix was removed: right for «أبي» (my father) -> أب, wrong for «شرك» (polytheism) -> شر. A word
            # that the lexicon itself uses (inside «الشرك بالله») stands as itself; otherwise the analyser decides
            if w in self._gloss_words():
                return False
            return g in lemmas(token) or g in lemmas(w)

        @staticmethod
        def _keeps_alef(token, entry):
            def bare(w):
                w = normalize_ar(w)
                for p in ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل"):
                    if w.startswith(p) and len(w) - len(p) >= 3:
                        return w[len(p):]
                return w
            t, g = bare(token), bare(entry.gloss)
            if t.startswith("ا") and not g.startswith("ا"):
                return False
            # a final taa marbuta is part of the word, not the pronoun «ـه»: «مدرسة» (school) is not «مدرس»
            # (teacher), «عائلة» (family) is not «عائل»
            return not (strip(token).endswith("ة") and not normalize_ar(entry.gloss).endswith("ه"))

        def lookup(self, gloss):
            e = super().lookup(gloss)
            if e is None and getattr(self, "_aliases", None):
                target = self._aliases.get(normalize_ar(gloss))
                e = self._signs.get(target) if target else None
            return e

        def _lex_index(self):
            """(verb|nom, CAMeL lemma) -> gloss key, from the most probable reading of each one-word gloss. A gloss
            enters only when its own best reading is unambiguous: «قال» is the verb قال; «كفر» (best reading a proper
            noun, then the verb كَفَّر *expiated* and the noun كُفْر) enters nowhere, so «وتكفر» cannot reach it."""
            if not hasattr(self, "_lexidx"):
                # glosses written with vowel marks («شِعر», poetry) are left out: undiacritised, «شعر» reads as the
                # verb *to feel*, which is not what the sign means
                keys = [k for k, e in self._signs.items() if " " not in k and strip(e.gloss) == e.gloss.strip()]
                morph.lemmatize(keys, analysed=True)
                idx = {}
                for k in keys:
                    top = morph.top_analyses(k)
                    kinds = {(self._kind(a["pos"]), a["lex"]) for a in top}
                    if len(kinds) != 1 or None in next(iter(kinds)):
                        continue
                    # and only a gloss spelled as that lemma: CAMeL reads «فجل» (radish) as ف+جل, «يسلم» as the verb
                    # سلّم and «قطة» via قط, which would send «وجل», «وسلم», «قط» to the wrong signs
                    if k != self._lemma_key(next(iter(kinds))[1]):  # («إلهام» is not ال+هام)
                        continue
                    idx.setdefault(next(iter(kinds)), []).append(k)
                # the gloss spelled like the lemma first («قال» before «يقول», «لون» before «ألوان»), then the shortest
                self._lexidx = {kl: sorted(ks, key=lambda g: (len(g), g))[0]
                                for kl, ks in idx.items()}
            return self._lexidx

        MARGIN = 0.5  # log-probability lead the best reading needs over a reading with another lemma

        @staticmethod
        def _lemma_key(lex):
            return normalize_ar(re.sub(r"[_\d].*$", "", lex))

        @staticmethod
        def _kind(pos):
            return "verb" if pos == "verb" else "nom" if pos in morph.NOMINAL else None

        def _lemma_match(self, token):
            """Inflected forms through CAMeL's most probable reading: «قالت، فقال، يقول» -> the verb sign قال, «ربك،
            بيده» -> رب، يد (clitics, person, number and tense come from the analysis, not from string stripping). A
            verb reading reaches only a gloss that is itself that verb (same diacritised lemma); a noun reading only a
            gloss that is that noun. Several best readings pointing to different signs (or to none): no match."""
            top = morph.top_analyses(token)
            if not top:
                return None
            idx, found = self._lex_index(), set()
            for a in top:
                kind = self._kind(a["pos"])
                g = idx.get((kind, a["lex"])) if kind else None
                if g is None and kind == "verb":  # a reviewed verb signed with its act's noun: «تصوم» -> صوم
                    g = getattr(self, "_verb_lemmas", {}).get(self._lemma_key(a["lex"]))
                if g is None:
                    return None
                found.add(g)
            if len(found) != 1:
                return None
            # a close second reading with another lemma: too ambiguous out of context («فقدت»: ف+قاد or فقد,
            # «وسعت»: و+سعى or وسع)
            keys = {self._lemma_key(a["lex"]) for a in top}  # «قال» as قال-u or قال-i is one word
            if any(self._lemma_key(x["lex"]) not in keys and x["lp"] > -99 and top[0]["lp"] - x["lp"] < self.MARGIN
                   for x in morph.analyses(token)):
                return None
            e = self._signs[found.pop()]
            bare = normalize_ar(token)
            if (bare.endswith("ا") and not normalize_ar(e.gloss).endswith("ا")
                    and all(a["pos"] != "verb" and a["enc0"] == "0" for a in top)):
                return None  # an adverbial accusative is another word: «جدا» (very) is not جد (grandfather)
            if bare.endswith("ه") and any(a["enc0"] != "0" for a in top):
                # a final «ه» read as the pronoun: not if the same letters also read as a «ة» noun («جنه»: جنة)
                if any(x["enc0"] == "0" and x["lp"] > -99 and normalize_ar(x["lex"]).endswith("ه")
                       for x in morph.analyses(token)):
                    return None
            return e if self._keeps_alef(token, e) else None

        def _verb_reading(self, token):
            """The sign of a verb reading about as probable as the best one (within MARGIN), if exactly one sign."""
            readings = morph.analyses(token)
            if not readings:
                return None
            best = max(a["lp"] for a in readings)
            idx, found = self._lex_index(), set()
            for a in readings:
                if a["pos"] == "verb" and best - a["lp"] < self.MARGIN:
                    g = idx.get(("verb", a["lex"])) or getattr(self, "_verb_lemmas", {}).get(self._lemma_key(a["lex"]))
                    if g:
                        found.add(g)
            return self._signs[found.pop()] if len(found) == 1 else None

        def match_token(self, token):
            """The matcher above, the reviewed synonyms, then lemma matching through the analyser. A match the string
            matcher reached only by taking a letter off the front («فتحت» -> ف + تحت, under) gives way to an equally
            probable verb reading that has a sign («فَتَحَت», she opened -> يفتح)."""
            e = self._match(token)
            if e is not None and normalize_ar(e.gloss).removeprefix("ال") != normalize_ar(token).removeprefix("ال"):
                bare = normalize_ar(token)
                if bare[:1] in ("ف", "و", "ب", "ل") and not normalize_ar(e.gloss).startswith(bare[:1]):
                    v = self._verb_reading(token)
                    if v is not None:
                        return v
            if e is None:
                e = self._alias(token)
            return e if e is not None else self._lemma_match(token)

        def _alias(self, token):
            e = None
            if e is None and getattr(self, "_aliases", None):
                bare = normalize_ar(token)
                for w in [bare] + [bare[len(p):] for p in ("وال", "فال", "بال", "ال", "و", "ف", "ب", "ل")
                                   if bare.startswith(p) and len(bare) - len(p) >= 2]:
                    if w in self._aliases:  # «ونكاح», «بالزوجة»: a clitic before a listed word
                        return self._signs.get(self._aliases[w])
            return e

    base = Lexicon.load(path)
    lex = ArabicQALexicon.__new__(ArabicQALexicon)
    lex.__dict__.update(base.__dict__)
    # reviewed synonyms (lexicon/synonyms_ar.json): words with no sign of their own, signed with a sign of the same
    # meaning («نكاح» -> زواج, «الزوجة» -> زوج); only targets that exist, and never over a word's own sign
    reviewed = json.loads((Path(__file__).with_name("synonyms_ar.json")).read_text(encoding="utf-8"))
    syn = reviewed["synonyms"]
    lex._aliases = {normalize_ar(w): normalize_ar(g) for w, g in syn.items()
                    if normalize_ar(g) in lex._signs and normalize_ar(w) not in lex._signs}
    # reviewed verbs signed with the noun of the act (صام -> صوم): every conjugation the analyser reads as that verb
    lex._verb_lemmas = {normalize_ar(v): normalize_ar(g) for v, g in reviewed.get("verb_lemmas", {}).items()
                        if normalize_ar(g) in lex._signs}
    return lex
