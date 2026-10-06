"""Turkish Sign Language (TİD) lexicon: the TİD dictionary signs, glossed by Turkish words. Keys use Turkish casing.

A word is matched (TIDLexicon.match) by the first of these routes that finds a sign:
  exact      the word is a gloss (Turkish casing; circumflex and ASCII spellings: «zekât», «oruc» for «oruç»)
  compound   the word is a multi-word gloss written as one word
  lemma      a zeyrek lemma of the word is a gloss (kitabı -> kitap, geldi -> gelmek, ağzı -> ağız). Readings that change
             the meaning (inanmadı, imansız, öldürdü) never match, and among the readings a gloss lemma is preferred
  synonym    the word or its lemma is listed in synonyms_tr.json (reviewed: nebi -> peygamber, salat -> namaz)
  prefix     only for a word the analyser does not know at all: the one stem-eligible gloss of PREFIX_MIN+ letters that
             the word starts with. The old rule (any word sharing a gloss's first five letters) was wrong for about a
             third of the words it matched (scripts/eval/tr_matcher_eval.py)
match_tokens also matches multi-word glosses over a word sequence (namazı kıldı -> «namaz kılmak»), longest first."""
import json
import re
from pathlib import Path

from isharati.config import DATA
from isharati.lexicon.asl import ASLLexicon
from isharati.text import turkish_morph as morph
from isharati.text.turkish import tr_norm
from isharati.types import SignEntry

TID_ROOT = DATA / "lexicon_tr"
TID_FILE = Path("dictionaries") / "entries_tid_dictionary.jsonl"
ALPHABET = Path("alphabet") / "entries.jsonl"  # one signer's manual alphabet (scripts/lexicon/alphabet/build.py)
NAMES = Path("names") / "entries.jsonl"        # names spelled from it (scripts/lexicon/alphabet/names.py)
# a second TİD source for the corpus words the dictionary lacks: the rest of the TİD Sözlüğü channel and TiDiSLaM's
# Islamic terms (scripts/lexicon/tid/tid_youtube.py)
TID_YOUTUBE = Path("tid_youtube") / "entries.jsonl"
# AUTSL (Ankara University TSL dataset, research use only): the medoid recording across signers of each of its signs
# the dictionary lacks (scripts/lexicon/tid_datasets/autsl_build.py); its "variant_of" rows (more signers, and AUTSL's
# recordings of dictionary signs) are skipped by add_words
AUTSL = Path("autsl") / "entries.jsonl"
# the dictionary's opposite pairs re-cut into one sign per word (scripts/lexicon/tid/pairs_fix.py): 66 dictionary clips
# were shared by two opposite glosses (iyi / kötü, var / yok, ...); these entries replace them (override_words)
PAIRS_FIXED = Path("pairs_fixed") / "entries.jsonl"
REVIEWED = Path("reviewed") / "recordings.jsonl"
# gap fillers from two YouTube sources (scripts/lexicon/tid/extra_sources.py): «Our hands are speaking» (youth-exchange
# participants, probably learners) and The Sign Polyglot, all pending review; loaded last, so they never replace a sign,
# and their recordings of words the lexicon already signs are "variant_of" rows (skipped by add_words)
EXTRA = Path("extra") / "entries.jsonl"
# Cüneyt Küçükoğlu's «Hareketli Sözlük» channel (scripts/lexicon/tid/cuneyt.py), pending review: after the dictionary,
# tid_youtube and AUTSL, so it only fills gaps; its recordings of words those sign are "variant_of" rows
CUNEYT = Path("cuneyt") / "entries.jsonl"
# words whose dictionary clip still shows the opposite word (the re-cut failed QA, scripts/lexicon/tid/pairs_fix.py):
# withdrawn, so a later source (tid_youtube, AUTSL) can sign them, else they are fingerspelled
WRONG_SIGN = ("eksik", "yaşlı", "üst")
SYNONYMS = Path(__file__).with_name("synonyms_tr.json")
PREFIX_MIN = 6  # the stem fallback: only a gloss at least this long, for a word the analyser does not know
_FOLD = str.maketrans("çğışöüâîû", "cgisouaiu")


class TIDLexicon(ASLLexicon):
    _key = staticmethod(tr_norm)
    _letter_key = staticmethod(tr_norm)  # the 29 letters of the Turkish alphabet: «I» -> ı, «İ» -> i, «Ç» -> ç

    @staticmethod
    def _spell_chars(word):
        return tr_norm(word).replace(" ", "")

    def __init__(self, entries, root):
        super().__init__(entries, root)
        self._prefix: dict[str, SignEntry] = {}
        for k, e in self._signs.items():
            if len(k) > 5:
                self._prefix.setdefault(k[:5], e)
        self._stem_ok: set[str] = set(self._signs)  # glosses the stem fallback may use: the dictionary, stems=True
        self._aliases: dict[str, str] = {}
        self._idx, self._version = None, 0  # the lookup tables (_index), rebuilt when _version changes

    prefix_fallback = True  # the stem fallback for words the analyser does not know (off: lemma and synonyms only)

    def match_token(self, token):
        return self.match(token)[0]

    def lookup(self, gloss):
        """The sign of a gloss, or of the gloss a reviewed synonym points to."""
        e = self._signs.get(self._key(gloss))
        return e if e is not None else self._alias(gloss)

    def match(self, token):
        """(sign, route) for a word or a multi-word gloss, (None, None) if nothing matches; the routes are listed in
        the module docstring."""
        n = tr_norm(token)
        if not n:
            return None, None
        if n in self._signs:
            return self._signs[n], "exact"
        idx = self._index()
        if " " in n:  # a multi-word text (the LLM glosser's «namaz kılmak»): the phrase, by its words' lemmas
            ws = n.split()
            e = self._phrase_at(ws, 0, len(ws))
            if e is not None:
                return e, "lemma"
            e = self._alias(n)
            return (e, "synonym") if e is not None else (None, None)
        f = morph.fold(n)
        if f in idx["word"]:  # a circumflex spelling of a gloss («zekât» / «zekat»)
            return idx["word"][f], "exact"
        # a gloss written without Turkish letters (the LLM glosser sometimes writes «oruc» for «oruç»): the sign whose
        # key folds to it, when exactly one does
        if n.isascii() and n in self._ascii_keys():
            return self._ascii_keys()[n], "exact"
        if f in idx["compound"]:
            return idx["compound"][f], "compound"
        lems = morph.lemmas(n)
        for lem in lems:  # best reading first; the first lemma that is a gloss
            if lem in idx["lemma"]:
                return idx["lemma"][lem], "lemma"
        for w in [f] + lems[:1]:  # the best reading only: «dile» (to the tongue) must not reach dilemek's synonym
            e = self._alias(w)
            if e is not None:
                return e, "synonym"
        if self.prefix_fallback and not morph.readings(n):
            e = self._prefix_match(f)
            if e is not None:
                return e, "prefix"
        return None, None

    def match_tokens(self, tokens):
        """Signs for a word sequence: [(start, end, sign, route)]. A multi-word gloss takes precedence over its words
        (longest first); unmatched words are left out."""
        ws = [tr_norm(t) for t in tokens]
        idx, out, i = self._index(), [], 0
        while i < len(ws):
            hit = None
            for size in range(min(idx["max_phrase"], len(ws) - i), 1, -1):
                key = " ".join(ws[i:i + size])
                if key in self._signs:
                    hit = (size, self._signs[key], "exact")
                    break
                e = self._phrase_at(ws, i, size)
                if e is not None:
                    hit = (size, e, "lemma")
                    break
            if hit is None:
                e, route = self.match(ws[i])
                hit = (1, e, route)
            if hit[1] is not None:
                out.append((i, i + hit[0], hit[1], hit[2]))
            i += hit[0]
        return out

    def _phrase_at(self, ws, i, size):
        """The multi-word gloss of `size` words at ws[i], each word compared as written or by its lemma."""
        if i + size > len(ws) or not ws[i]:
            return None
        by_first = self._index()["phrase"].get(size)
        if not by_first:
            return None
        firsts = [morph.fold(ws[i])] + morph.lemmas(ws[i])
        for first in firsts:
            for gws, e in by_first.get(first, ()):
                if all(g in ({morph.fold(w)} | set(morph.lemmas(w)) if w else set())
                       for g, w in zip(gws[1:], ws[i + 1:i + size])):
                    return e
        return None

    def _alias(self, word):
        """A reviewed synonym's sign: only for a word with no sign of its own, and only if the target is signed."""
        if not self._aliases or tr_norm(word) in self._signs:
            return None
        t = self._aliases.get(morph.fold(word))
        return self._signs.get(t) if t else None

    def load_synonyms(self, path: Path = SYNONYMS) -> int:
        """Reviewed synonyms (synonyms_tr.json): word -> the gloss of a sign of the same meaning. Returns how many of
        their targets this lexicon signs (the others wait until it does)."""
        path = Path(path)
        syn = json.loads(path.read_text(encoding="utf-8"))["synonyms"] if path.exists() else {}
        self._aliases = {morph.fold(w): self._key(g) for w, g in syn.items() if " " not in w.strip()}
        # a multi-word entry is a fixed formula signed by one sign («sallallahu aleyhi ve sellem» -> ALEYHİSSELAM):
        # it becomes a phrase gloss of that sign, so the matcher covers the whole formula and the glosser signs it once
        added = 0
        self._formulas = {}
        for w, g in syn.items():
            if " " in w.strip() and self._key(g) in self._signs:
                self._formulas[tuple(morph.fold(x) for x in w.split())] = self._signs[self._key(g)].gloss
                if self._key(w) not in self._signs:
                    self._signs[self._key(w)] = self._signs[self._key(g)]
                    added += 1
        if added:
            self._version = getattr(self, "_version", 0) + 1
        return sum(1 for g in self._aliases.values() if g in self._signs) + added

    def rewrite_formulas(self, text: str) -> str:
        """The text with each fixed formula (synonyms_tr.json multi-word entries) replaced by the gloss of its sign:
        «Peygamber sallallahu aleyhi ve sellem dedi» -> «Peygamber aleyhi dedi». The LLM glosser sees only glosses, so
        it split the formula into words and «sallallahu» came out as a missing sign. Matching is on folded words
        (sallallâhu = sallallahu); punctuation and the rest of the text are kept."""
        formulas = getattr(self, "_formulas", None)
        if not formulas:
            return text
        words = list(re.finditer(r"[^\W\d_]+", text))
        folded = [morph.fold(m.group(0).lower()) for m in words]
        longest = max(len(k) for k in formulas)
        out, last, i = [], 0, 0
        while i < len(words):
            for n in range(min(longest, len(words) - i), 1, -1):
                g = formulas.get(tuple(folded[i:i + n]))
                if g:
                    out.append(text[last:words[i].start()] + g)
                    last, i = words[i + n - 1].end(), i + n
                    break
            else:
                i += 1
        return "".join(out) + text[last:]

    def _prefix_match(self, f):
        """The stem fallback: the stem-eligible gloss of PREFIX_MIN+ letters that the (unknown) word starts with and is
        longer than; none if glosses with different signs compete."""
        if len(f) <= PREFIX_MIN:
            return None
        hits = {id(e): e for g, e in self._index()["stem"].items() if len(f) > len(g) and f.startswith(g)}
        return next(iter(hits.values())) if len(hits) == 1 else None

    def _index(self):
        """Lookup tables over the current glosses, rebuilt whenever the signs change."""
        state = (len(self._signs), self._version)
        if self._idx is not None and self._idx_state == state:
            return self._idx
        word, compound, lemma, phrase, stem = {}, {}, {}, {}, {}
        for k, e in self._signs.items():
            f = morph.fold(k)
            if " " in k:
                ws = f.split()
                compound.setdefault("".join(ws), e)
                phrase.setdefault(len(ws), {}).setdefault(ws[0], []).append((tuple(ws), e))
            else:
                word.setdefault(f, e)
                if k in self._stem_ok and len(f) >= PREFIX_MIN:
                    stem[f] = e
        lemma.update(word)
        # a gloss written as an inflected form also signs its lemma («gel» -> gelmek), when it has one lemma and no
        # gloss of that lemma exists; zeyrek's verb lemma is the infinitive, as the dictionary glosses verbs
        for k, e in self._signs.items():
            if " " not in k:
                ls = morph.lemmas(k)
                if len(ls) == 1 and ls[0] not in lemma:
                    lemma[ls[0]] = e
        self._idx = {"word": word, "compound": compound, "lemma": lemma, "phrase": phrase, "stem": stem,
                     "max_phrase": max(phrase, default=1)}
        self._idx_state = state
        return self._idx

    def _ascii_keys(self) -> dict[str, SignEntry]:
        if getattr(self, "_ascii_n", None) != len(self._signs):  # rebuilt after add_words
            by = {}
            for k, e in self._signs.items():
                f = k.translate(_FOLD)
                if f != k:
                    by.setdefault(f, []).append(e)
            self._ascii = {f: es[0] for f, es in by.items() if len(es) == 1 and f not in self._signs}
            self._ascii_n = len(self._signs)
        return self._ascii

    def withdraw(self, glosses) -> int:
        """Drops these glosses' signs (and the stem matches that pointed to them). Returns the number dropped."""
        dropped = 0
        for g in glosses:
            old = self._signs.pop(self._key(g), None)
            if old is not None:
                dropped += 1
                for p in [p for p, pe in self._prefix.items() if pe is old]:
                    del self._prefix[p]
        self._ascii_n = None
        self._version += 1
        return dropped

    def override_words(self, path: Path) -> int:
        """Word signs that REPLACE the lexicon's sign of the same gloss (add_words never displaces one): corrections,
        such as the dictionary's opposite pairs re-cut into one sign per word (PAIRS_FIXED). "variant_of" rows are
        skipped. A stem match that pointed to a replaced sign follows it. Returns the number of glosses replaced."""
        path = Path(path)
        if not path.exists():
            return 0
        replaced = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("variant_of"):
                continue
            e = SignEntry(**{k: v for k, v in row.items() if k in SignEntry.__dataclass_fields__})
            if e.is_letter:
                continue
            k = self._key(e.gloss)
            old = self._signs.get(k)
            self._signs[k] = e
            replaced += old is not None
            for p, pe in list(self._prefix.items()):
                if pe is old and old is not None:
                    self._prefix[p] = e
        self._ascii_n = None  # rebuild the ASCII-folded keys
        self._version += 1
        return replaced

    def add_words(self, path: Path, stems: bool = False) -> int:
        """ASLLexicon.add_words (a gloss the lexicon already signs keeps its sign); stems=True also lets the new
        glosses be used by the stem fallback (match: words the analyser does not know), as the dictionary's are."""
        before = set(self._signs)
        added = super().add_words(path)
        if stems:
            for k in [k for k in self._signs if k not in before]:  # file order: the first of a prefix wins
                self._stem_ok.add(k)
                if len(k) > 5:
                    self._prefix.setdefault(k[:5], self._signs[k])
        self._version += 1
        return added


def tid_lexicon(root: Path | None = None, extra: bool = True, cuneyt: bool = True) -> TIDLexicon:
    root = Path(root or TID_ROOT)
    rows = [json.loads(l) for l in (root / TID_FILE).read_text(encoding="utf-8").splitlines() if l.strip()]
    lex = TIDLexicon([SignEntry(**r) for r in rows], root)  # keypoints_path is relative to lexicon_tr/
    lex.override_words(root / PAIRS_FIXED)  # the opposite pairs' own signs, over the clip the two words shared
    lex.override_words(root / REVIEWED)  # recordings chosen over a missing or weak sign (reviewed_recordings.py)
    lex.withdraw(WRONG_SIGN)  # before the other sources, which may have a right sign for them
    lex.add_letters(root / ALPHABET)  # letters for fingerspelling
    lex.add_words(root / NAMES)       # fingerspelled names (Sahaba), where no sign exists
    lex.add_words(root / TID_YOUTUBE, stems=True)  # the second TİD source, never over a sign the lexicon has
    lex.add_words(root / AUTSL, stems=True)  # dataset signs: a dictionary sign of the same gloss wins
    if cuneyt:  # cuneyt.py builds against the lexicon without it
        lex.add_words(root / CUNEYT, stems=True)  # the Hareketli Sözlük channel: only words the sources above lack
    if extra:  # extra_sources.py builds against the lexicon without them
        lex.add_words(root / EXTRA, stems=True)  # learner and tutorial recordings last: only words nothing else signs
    lex.load_synonyms()  # reviewed synonyms (synonyms_tr.json): never over a word's own sign, only to a signed gloss
    return lex
