"""Arabic morphology for matching words to signs: stems that stay words, regular plurals to the singular,
CAMeL Tools lemmas (cached on disk), verb detection, number words."""
import json
from pathlib import Path

from isharati.config import DATA, ROOT
from isharati.text.arabic import candidates, normalize_ar

# function words: not counted as content words and not signed. Question words (لماذا، كيف، متى، أين، ماذا، هل) and
# the negations that have a sign (لم، لن، ليس) are NOT here: they carry meaning and are signed. «من» and «ما» stay (as
# prepositions/relatives they are function words; the glosser signs them only in a question, see QUESTION_WORDS).
STOPWORDS = {normalize_ar(w) for w in (
    "من في على إلى الى عن ثم أن إن ما هذا هذه ذلك الذي التي الذين هو هي هم قد مع كما أو أي "
    "و ف ب ل ك يا إذا اذا كل بعض عند بين حتى لقد وقد وهو وهي وما ومن وفي "
    # added: prepositions with a pronoun suffix, particles, conjunction combos, suffix-only remnants (CAMeL POS
    # prep/conj/part/pron over the corpus, reviewed by hand)
    "عنه عنها عنهم عنهما عنك عنكم منه منها منهم منكم منا مني به بها بهم بكم بي بك إليه إليها إليهم إليك إليكم "
    "عليه عليها عليهم عليكم علينا عليك علي له لها لهم لكم لنا لي لك فيه فيها فيهم فيكم معه معهم عنده عندهم "
    "إلا الا أنه انه إنه إنها أنها أنهم إنهم أنكم إنكم أني إني إنما انما أنما لعل لكن ولكن لكنه أما إذ اذ إذن "
    "ثم وثم فإن فان وإن وان وأن فإذا وإذا فلما ولما لما أو وأما أي أيضا هذا وهذا وهذه وذلك فذلك ذلكم تلك "
    "كذلك هكذا الذي والذي التي والتي الذين والذين اللذين اللاتي عن وعن ومن ممن مما فيما بما كما لأن لان "
    "ف و نا هما هن أنتم نحن أنا أنت إياك إياه إياهم أولئك هؤلاء هنا هناك ذا ذات أيها يا ويا قد فقد وقد "
    "على وعلى إلى وإلى في وفي بل أم حين حيث إذا كي لكي ليكون").split()}

# signed function words: the glosser keeps these when the model returns them (question words, negation, pronouns)
QUESTION_WORDS = {normalize_ar(w) for w in "لماذا كيف متى أين ماذا هل من ما".split()}
NEGATIONS = {normalize_ar(w) for w in "لا لم لن ليس".split()}
PRONOUN_SIGNS = {normalize_ar(w) for w in "أنا أنت أنتم أنتما نحن هو هي هم هما".split()}


def strict_forms(word):
    """A word's stems for comparing words with each other: the stemmer can cut «آلة», «الكمان» and «إليه» all down
    to «ال», so stems shorter than three letters are left out (the whole word is always kept)."""
    w = normalize_ar(word)
    forms = {w} | {c for c in candidates(w) if len(c) >= 3}
    if len(w) >= 4 and w.endswith("ا"):  # tanween alef: «محمدا»
        forms.add(w[:-1])
    for c in list(forms):
        forms |= {s for s in singulars(c) if len(s) >= 3}
    return forms


def singulars(w):
    """Regular Arabic endings back to the singular (normalised forms: ة is written ه)."""
    out = []
    if len(w) >= 5 and w.endswith("وات"):          # صلوات -> صلاه, زكوات -> زكاه
        out.append(w[:-3] + "اه")
    if len(w) >= 5 and w.endswith("ات"):           # مكتوبات -> مكتوبه, سيارات -> سياره
        out += [w[:-2] + "ه", w[:-2]]
    if len(w) >= 5 and (w.endswith("ون") or w.endswith("ين")):  # مسلمون, مسلمين -> مسلم
        out.append(w[:-2])
    if len(w) >= 4 and w.endswith("ه"):            # مفروضه -> مفروض (a feminine adjective)
        out.append(w[:-1])
    return out


LEMMA_CACHE = DATA / "lexicon_v2" / "lemma_cache.json"
# the full analyses (lemma, POS, clitics, person) live in their own file: processes running older code rewrite
# LEMMA_CACHE with only its "l"/"v" fields
ANALYSIS_CACHE = DATA / "lexicon_v2" / "analysis_cache_ar.json"
NLP_PYTHON = Path(r"D:\islam\models\.venv-nlp\Scripts\python.exe")
_lemmas = None
_analyses = None


def lemmas(word):
    """CAMeL Tools lemmas of a word (broken plurals «الخطايا» -> خطيئة, verbs «تمحو» -> محا), cached on disk;
    [] when the NLP environment is not installed."""
    global _lemmas
    if _lemmas is None:
        _lemmas = json.loads(LEMMA_CACHE.read_text(encoding="utf-8")) if LEMMA_CACHE.exists() else {}
    if word not in _lemmas:
        lemmatize([word])
    v = _lemmas.get(word, [])
    return [normalize_ar(x) for x in (v.get("l", []) if isinstance(v, dict) else v)]


def is_verb(word):
    """True if the analyser can read the word as a verb (then only its exact form may match a sign)."""
    lemmas(word)
    v = _lemmas.get(word)
    return isinstance(v, dict) and v.get("v", False)


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:
        return {}


def lemmatize(words, analysed=False):
    """Look up many words in one call to the analyser (it takes a few seconds to load). analysed: also make sure
    each word has its full analyses (ANALYSIS_CACHE)."""
    global _lemmas, _analyses
    if _lemmas is None:
        _lemmas = _load(LEMMA_CACHE)
    if analysed and _analyses is None:
        _analyses = _load(ANALYSIS_CACHE)
    todo = sorted({w for w in words if w not in _lemmas or (analysed and w not in _analyses)})
    if not todo or not NLP_PYTHON.exists():
        return
    import subprocess
    r = subprocess.run([str(NLP_PYTHON), str(ROOT / "scripts" / "lexicon" / "arsl" / "lemmatize_ar.py")], input=json.dumps(todo, ensure_ascii=False),
                       capture_output=True, text=True, encoding="utf-8", timeout=3600)
    try:
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        got = {w: {"l": [], "v": False, "a": []} for w in todo}
    new_lemmas = {w: {"l": v.get("l", []), "v": v.get("v", False)} for w, v in got.items() if w not in _lemmas}
    _lemmas.update(new_lemmas)
    if new_lemmas:
        _save(LEMMA_CACHE, _lemmas)
    if _analyses is not None:
        _analyses.update({w: v.get("a", []) for w, v in got.items()})
        _save(ANALYSIS_CACHE, _analyses)


def _save(path, cache):
    """Merge with what others wrote to the file meanwhile, then replace it atomically (temp file + rename)."""
    import os
    import time
    for w, v in _load(path).items():
        cache.setdefault(w, v)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    for _ in range(40):  # Windows refuses the rename while another process is reading the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    tmp.unlink(missing_ok=True)  # only a speed-up: the analyses stay in memory this time


NOMINAL = ("noun", "adj", "noun_quant", "adj_comp", "noun_num")


def analyses(word):
    """CAMeL analyses of a normalised word, most probable first: dicts with pos, lex (diacritised lemma), lp
    (pos_lex_logprob), enc0, per, num, gen. [] when the NLP environment is not installed."""
    w = normalize_ar(word)
    if _analyses is None or w not in _analyses:
        lemmatize([w], analysed=True)
    rows = (_analyses or {}).get(w, [])
    return [dict(zip(("pos", "lex", "lp", "enc0", "per", "num", "gen"), r)) for r in rows]


def top_analyses(word):
    """The analyses that share the best pos_lex_logprob (the word's most probable reading, out of context); [] when
    the analyser has no frequency for any reading (-99) or no analysis."""
    a = analyses(word)
    if not a or a[0]["lp"] <= -99:
        return []
    return [x for x in a if x["lp"] == a[0]["lp"]]


NUMBERS = {normalize_ar(w): str(n) for n, ws in {
    1: "واحد واحدة", 2: "اثنان اثنين اثنتان اثنتين", 3: "ثلاث ثلاثة", 4: "أربع أربعة", 5: "خمس خمسة", 6: "ست ستة",
    7: "سبع سبعة", 8: "ثمان ثماني ثمانية", 9: "تسع تسعة", 10: "عشر عشرة"}.items() for w in ws.split()}
