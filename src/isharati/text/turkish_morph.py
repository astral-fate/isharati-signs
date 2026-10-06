"""Turkish morphology for matching words to TİD signs: zeyrek (a pure-Python port of Zemberek) readings of a word,
cached on disk, the way isharati.text.arabic_morph caches CAMeL Tools lemmas.

A reading is (lemma, part of speech, morphemes). zeyrek gives the dictionary lemma, so consonant alternation and vowel
drop are undone (kitabı -> kitap, rengi -> renk, ağzı -> ağız, oğlu -> oğul) and a verb's lemma is its infinitive
(geldi -> gelmek), which is how the TİD dictionary glosses verbs. Readings that change the meaning of the lemma
(negation «inanmadı», causative «öldürdü», «-sız» «imansız», «-lık» «kulluk», ...) are never used for matching, and a
reading that needs a zero derivation («kıldı» read as «it was hair») is used only when no plainer reading exists.

The analyser takes ~10 s and ~300 MB to load, so words are looked up in the cache first; without zeyrek installed the
cache alone is used and an uncached word has no readings ([]).
"""
import json
import logging
import os
from pathlib import Path

from isharati.config import DATA
from isharati.text.turkish import tr_norm

LEMMA_CACHE = DATA / "lexicon_tr" / "lemma_cache.json"
_CIRC = str.maketrans("âîû", "aiu")

# derivations whose result does not mean the lemma (or means its opposite): a reading containing one is not used
BLOCK = frozenset((
    "Neg", "Unable", "WithoutHavingDoneSo", "WithoutBeingAbleToHaveDoneSo", "NotState",  # not / without doing
    "Caus",                                     # öl-dür (die -> kill), gör-ün-dür
    "Without", "With", "Ness", "Agt", "Dim", "Related", "JustLike", "Ly", "FitFor", "Rel", "Become", "Acquire",
    "Ord", "Dist", "Recip", "Reflex", "InTermsOf", "Hastily", "EverSince", "Almost", "Stay", "Start", "Repeat",
    "Since", "ActOf", "Inf3",                   # gel-iş (a coming, also «gelişmek» to develop)
))
# readings are ranked: a plainer reading first; among equals a common noun, then a verb, then the rest
_POS_RANK = {"Noun": 0, "Verb": 1, "Adj": 2, "Adv": 3}
# function words and numbers: when a word has such a reading, only those readings count (see lemmas)
CLOSED = frozenset(("Conj", "Pron", "Postp", "Det", "Ques", "Num"))

_cache: dict | None = None
_analyzer = None


def fold(word: str) -> str:
    """Turkish casing and circumflex folded away: «Rasûl» -> rasul, «zekât» -> zekat (keys and lemmas compare this way)."""
    return tr_norm(word).translate(_CIRC)


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(LEMMA_CACHE.read_text(encoding="utf-8")) if LEMMA_CACHE.exists() else {}
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _get_analyzer():
    """zeyrek's analyser, loaded once; None when zeyrek is not installed."""
    global _analyzer
    if _analyzer is None:
        try:
            import zeyrek
        except ImportError:
            _analyzer = False
            return None
        logging.getLogger("zeyrek").setLevel(logging.ERROR)
        for name in list(logging.root.manager.loggerDict):
            if name.startswith("zeyrek"):
                logging.getLogger(name).setLevel(logging.ERROR)
        _patch_zeyrek()
        _analyzer = zeyrek.MorphAnalyzer()
    return _analyzer or None


def _patch_zeyrek():
    """zeyrek 0.1.3 shares mutable attribute sets: calculate_phonetic_attributes is lru-cached and returns one set that
    callers then .add() to, and a search path starts with its stem's own set and mutates it. The lexicon degrades as
    words are analysed (after a few thousand words «eder», «alıp», «olarak» get no reading at all). Every caller gets
    its own copy here, as Zemberek (Java) does with EnumSet.clone(). Must run before MorphAnalyzer() is built."""
    import inspect
    import textwrap
    from zeyrek import attributes, morphotactics, rulebasedanalyzer
    if getattr(attributes, "_isharati_patched", False):
        return
    cached = attributes.calculate_phonetic_attributes

    def fresh(word, predecessor_attrs=None):
        return set(cached(word, predecessor_attrs))
    for mod in (attributes, morphotactics, rulebasedanalyzer):
        if hasattr(mod, "calculate_phonetic_attributes"):
            mod.calculate_phonetic_attributes = fresh
    initial = morphotactics.SearchPath.initial.__func__

    def initial_copy(cls, stem_transition, tail):
        path = initial(cls, stem_transition, tail)
        path.phonetic_attributes = set(path.phonetic_attributes)
        return path
    morphotactics.SearchPath.initial = classmethod(initial_copy)
    src = textwrap.dedent(inspect.getsource(rulebasedanalyzer.RuleBasedAnalyzer.advance))
    old = "attributes = path.phonetic_attributes if tail_equals_surface"
    if old in src:
        ns = {}
        exec(src.replace(old, "attributes = set(path.phonetic_attributes) if tail_equals_surface"),
             vars(rulebasedanalyzer), ns)
        rulebasedanalyzer.RuleBasedAnalyzer.advance = ns["advance"]
    attributes._isharati_patched = True


def _parse(an, word):
    out = []
    try:
        res = an._parse(word)  # one word; MorphAnalyzer.analyze would run NLTK's sentence tokenizer first
    except Exception:  # noqa: BLE001 - an analyser crash on one odd word must not stop the batch
        res = []
    for a in res:
        if a is None:
            continue
        item = a.dict_item
        r = [fold(item.lemma), item.primary_pos.value, "+".join(m[0].id_ for m in a.morphemes),
             1 if getattr(item.secondary_pos, "value", "") == "Prop" else 0]
        if r not in out:
            out.append(r)
    return out


def analyze(words, save: bool = True) -> int:
    """Readings of many words, cached on disk (keys are tr_norm forms). Returns the number of words analysed."""
    cache = _load()
    todo = sorted({tr_norm(w) for w in words if w and tr_norm(w) and tr_norm(w) not in cache})
    todo = [w for w in todo if " " not in w]
    if not todo:
        return 0
    an = _get_analyzer()
    if an is None:
        return 0
    for i, w in enumerate(todo):
        cache[w] = _parse(an, w)
        if save and i and i % 5000 == 0:
            _save()
    if save:
        _save()
    return len(todo)


def _save():
    tmp = LEMMA_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, LEMMA_CACHE)


def readings(word):
    """All readings of a word: [(lemma, pos, morphemes, is_proper)], [] if unknown to the analyser."""
    w = tr_norm(word)
    cache = _load()
    if w not in cache:
        analyze([w])
    return [tuple(r) for r in cache.get(w, [])]


def known(word) -> bool:
    """True if the analyser has at least one reading of the word (then no stem guess is made for it)."""
    return bool(readings(word))


def _rank(r):
    lemma, pos, morphs, prop = r
    ms = morphs.split("+")
    return (ms.count("Zero"), prop, _POS_RANK.get(pos, 4), len(ms))


def lemmas(word):
    """The usable lemmas of a word, best reading first: readings with a meaning-changing derivation (BLOCK) are
    dropped, and only the plainest readings are kept (fewest zero derivations), so «kıldı» gives kılmak, not kıl.
    A function-word reading wins outright: «eğer» is «if», not «eğmek» (to bend); «bunun» is «bu», not «bun»."""
    rs = [r for r in readings(word) if not BLOCK.intersection(r[2].split("+"))]
    if not rs:
        return []
    best = min(_rank(r)[0] for r in rs)
    rs = [r for r in rs if _rank(r)[0] == best]
    closed = [r for r in rs if r[1] in CLOSED]
    if closed:
        rs = closed
    out = []
    for r in sorted(rs, key=_rank):
        if r[0] not in out:
            out.append(r[0])
    return out


def is_blocked_only(word) -> bool:
    """True if the analyser knows the word but every reading changes its lemma's meaning (inanmadı, imansız)."""
    rs = readings(word)
    return bool(rs) and all(BLOCK.intersection(r[2].split("+")) for r in rs)
