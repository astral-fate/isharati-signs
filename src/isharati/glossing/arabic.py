"""LLM glossing of Arabic answers into Arabic Sign Language, restricted to the lexicon: ArSL word order by
example (odd-numbered ArabSign sentences only), pronoun signs, phrase-sign guard, a re-check of skipped words."""
import json

from isharati.text.arabic import candidates, normalize_ar
from isharati.text.arabic_morph import PRONOUN_SIGNS, QUESTION_WORDS, STOPWORDS, lemmatize, strict_forms

# ArSL sentence order, shown by examples. They come only from the odd-numbered ArabSign sentences (Luqman, FG 2023):
# the even-numbered ones are the held-out test set of scripts/eval/eval_arabsign_gloss.py.
ORDER_EXAMPLES = [
    ("اقدم لكم اليوم برنامج اخر", "اليوم اقدم أنتم برنامج اخر"),
    ("لاتشرك بالله", "لا شرك الله"),
    ("خلق الله الحياة", "الحياة خلق الله"),
    ("خلق الله الناس للعبادة", "الله خلق الناس عبادة"),
    ("انا مسافر", "انا سفر"),
    ("احتمال ان يكون غدا امتحان", "غدا احتمال امتحان"),
    ("نسأل الله ان نلقاكم", "الله لقاء أنتم"),
    ("اول كلمة", "كلمة اول"),
    ("نلقاكم غدا باذن الله", "لقاء غدا اسم الله"),
    ("نرحب بكم دائما", "ترحيب أنتم دائما"),
]

GLOSS_SYSTEM = (
    "You convert Modern Standard Arabic into an Arabic Sign Language (ArSL) gloss sequence. "
    "Use ONLY glosses from the allowed list, in ArSL order, and keep the meaning. ArSL order: time words first "
    "(اليوم، غدا), the topic before what is said about it, a modifier after its noun (كلمة اول). "
    "Map every content word to the allowed gloss with the same meaning: plurals and inflections to the listed form "
    "(صلوات -> الصلاة، يصومون -> الصوم), verbs and participles to their base or verbal-noun sign (مسافر -> سفر، "
    "نرحب -> ترحيب، نلقاكم -> لقاء أنتم), number words to the listed digit (خمس -> 5). Attached pronouns become pronoun "
    "signs: لكم، بكم، عليكم، نلقاكم -> أنتم; لنا، علينا -> نحن; لي، بي -> أنا; له، به -> هو; لها -> هي; لهم -> هم. "
    "Use a listed phrase sign only when the text contains all of its words (محمد رسول الله); never let a phrase "
    "stand for a different word (الله أكبر is not الله تعالى). "
    "Drop function words that are not signed (أن، في، من، على، و، ثم), but keep question words (لماذا، كيف، متى، "
    "أين، ماذا، هل، من؟ who) and negation (لا، لم، لن، ليس; a negating ما as لا or لم): they are signed. A content word with no listed gloss "
    "is output as {\"text\": <the word without prefixes>, \"oov\": true}. Every content word of the text must "
    "appear once, as a gloss or as oov (رمضان with no listed gloss -> {\"text\": \"رمضان\", \"oov\": true}); "
    "never add a sign whose meaning is not in the text. Examples (Arabic -> ArSL): "
    + "; ".join(f"{a} -> {b}" for a, b in ORDER_EXAMPLES) + ". "
    "Return JSON only: {\"glosses\": [{\"text\": ..., \"oov\": false}]}"
)

# a preposition with an attached pronoun is signed as the pronoun
PRONOUNS = {"أنتم": "لكم بكم عليكم منكم اليكم إليكم معكم عندكم فيكم", "أنت": "لك بك عليك منك معك عندك",
            "نحن": "لنا بنا علينا منا معنا عندنا فينا", "أنا": "لي بي مني معي عندي",
            "هو": "له به عليه منه معه عنده", "هي": "لها بها عليها منها معها", "هم": "لهم بهم عليهم منهم معهم"}
PRONOUN_OF = {normalize_ar(w): sign for sign, ws in PRONOUNS.items() for w in ws.split()}


class ArabicQAGlosser:
    """LLM-only Arabic glossing for this mode, constrained to the lexicon; the Arabic app's LLMGlosser is unchanged."""
    name = "llm"

    def __init__(self, lexicon, client):
        self.lexicon, self.client = lexicon, client

    def gloss(self, text):
        from isharati.text.arabic import strip_diacritics
        from isharati.llm import GlossError, require_text as _require_text
        from isharati.types import AlignRow, GlossItem, GlossResult
        text = _require_text(text)
        allowed = "، ".join(e.gloss for e in self.lexicon.sign_entries())
        phrases = self.phrases_in(text)
        hint = ("\n\nThese phrase signs match words in the text; use each for its words: " + "، ".join(phrases)) if phrases else ""
        msgs = [{"role": "system", "content": GLOSS_SYSTEM},
                {"role": "user", "content": f"Allowed glosses: {allowed}\n\nText: {strip_diacritics(text)}{hint}"}]
        glosses = self._ask(msgs)
        skipped = self._skipped(text, glosses)
        if skipped:  # the model sometimes drops words that have a sign («في اليوم والليلة»): one re-check
            msgs = msgs + [{"role": "assistant", "content": json.dumps({"glosses": glosses}, ensure_ascii=False)},
                           {"role": "user", "content": "These words of the text have a listed gloss but are not in "
                            "your answer: " + "، ".join(skipped) + ". Return the full corrected JSON with each one "
                            "in its place, unless a phrase gloss you used already covers it (such as الشهادتين for "
                            "لا إله إلا الله)."}]
            try:
                glosses = self._ask(msgs) or glosses
            except Exception:
                pass
        lemmatize([strip_diacritics(str(g.get("text", ""))).strip() for g in glosses] + normalize_ar(text).split())
        items, trace = [], []
        for g in glosses:
            t = strip_diacritics(str(g.get("text", ""))).strip()
            if not t or self._dropped(t, text):
                continue
            pron = PRONOUN_OF.get(normalize_ar(t))
            entry = (self.lexicon.lookup(pron) if pron else None) or self.lexicon.lookup(t) or self.lexicon.match_token(t)
            if entry is not None and len(normalize_ar(entry.gloss).split()) > 1 and entry.gloss not in phrases:
                entry = None  # a phrase sign whose words are not in the text (الله تعالى for الله أكبر): not used
            if entry:
                items.append(GlossItem(entry.gloss))
                trace.append(AlignRow(t, normalize_ar(t), "llm", entry.gloss, entry.sign_id, entry.review_status))
            elif len(t.split()) > 1:  # a phrase with no sign of its own: sign its words («حج البيت» -> الحج بيت)
                text_forms = set().union(*[strict_forms(x) for x in normalize_ar(text).split()])
                for w in t.split():
                    if normalize_ar(w) in STOPWORDS or not (strict_forms(w) & text_forms):
                        continue  # a word the text does not have (تعالى from a misused phrase) is not signed
                    e = self.lexicon.match_token(w)
                    items.append(GlossItem(e.gloss) if e else GlossItem(w, oov=True))
                    trace.append(AlignRow(w, normalize_ar(w), "llm-split", e.gloss, e.sign_id, e.review_status)
                                 if e else AlignRow(w, normalize_ar(w), "missing"))
            else:
                items.append(GlossItem(t, oov=True))
                trace.append(AlignRow(t, normalize_ar(t), "missing"))
        items, trace = self._merge_phrases(items, trace, text, phrases)
        return GlossResult(items, self.name, trace)

    @staticmethod
    def _dropped(t, text):
        """A function word the model returned that is not signed. Pronoun signs (هو، أنتم ...) and prepositions with a
        pronoun (له -> هو) are kept; question words and negation are never stopwords; «من»/«ما» are signed only when
        the text is a question (otherwise they are the preposition / relative)."""
        n = normalize_ar(t)
        if n in PRONOUN_OF or n in PRONOUN_SIGNS or n not in STOPWORDS:
            return False
        return not (n in QUESTION_WORDS and ("؟" in text or "?" in text))

    def _merge_phrases(self, items, trace, text, phrases):
        """A recorded phrase of 3+ words that the text contains word for word («عائشة رضي الله عنها», «علي رضي الله
        عنه») is signed as that phrase, whatever the model did: a run of consecutive glosses made only of the phrase's
        words («عائشة», «رضي», «الله») becomes the phrase sign, and such leftovers right next to the phrase are
        dropped. The same words elsewhere in the sentence are left alone."""
        from isharati.types import AlignRow, GlossItem
        tokens = normalize_ar(text).split()
        words_of = lambda g: normalize_ar(g).split()  # noqa: E731
        for phrase in sorted(phrases, key=lambda g: -len(words_of(g))):
            pw = words_of(phrase)
            if len(pw) < 3 or not any(tokens[i:i + len(pw)] == pw for i in range(len(tokens) - len(pw) + 1)):
                continue
            entry = self.lexicon.lookup(phrase)
            if entry is None:
                continue
            inside = lambda it: set(words_of(it.text)) <= set(pw)  # noqa: E731  (a no-sign «عنها» is the phrase too)
            n = len(items)
            at = next((k for k in range(n) if normalize_ar(items[k].text) == normalize_ar(entry.gloss)), None)
            if at is None:  # the model split the phrase: its longest run of 2+ glosses inside it becomes the phrase
                best = None
                k = 0
                while k < n:
                    j = k
                    while j < n and inside(items[j]):
                        j += 1
                    if j - k >= 2 and (best is None or j - k > best[1] - best[0]):
                        best = (k, j)
                    k = max(j, k + 1)
                if best is None:
                    continue
                k, j = best
                items = items[:k] + [GlossItem(entry.gloss)] + items[j:]
                trace = trace[:k] + [AlignRow(phrase, normalize_ar(phrase), "phrase", entry.gloss, entry.sign_id,
                                              entry.review_status)] + trace[j:]
                at = k
            # leftovers on either side of the phrase that only repeat its words
            lo, hi = at, at + 1
            while lo > 0 and inside(items[lo - 1]) and normalize_ar(items[lo - 1].text) != normalize_ar(entry.gloss):
                lo -= 1
            while hi < len(items) and inside(items[hi]) and normalize_ar(items[hi].text) != normalize_ar(entry.gloss):
                hi += 1
            items = items[:lo] + [items[at]] + items[hi:]
            trace = trace[:lo] + [trace[at]] + trace[hi:]
        return items, trace

    def phrases_in(self, text, window=8):
        """Multi-word glosses whose words all occur in the text within a few words of each other («وأن محمدًا عبده
        ورسوله ... الله» -> «محمد رسول الله»), longest first, so the model uses the one phrase sign."""
        toks = [strict_forms(w) for w in normalize_ar(text).split()]
        found = []
        for e in self.lexicon.sign_entries():
            words = [w for w in normalize_ar(e.gloss).split() if w not in STOPWORDS]
            if len(words) < 2:
                continue
            want = [strict_forms(w) for w in words]
            for i in range(len(toks)):
                span = toks[max(0, i - window): i + window]
                if all(any(wf & tf for tf in span) for wf in want):
                    found.append(e.gloss)
                    break
        return sorted(set(found), key=len, reverse=True)[:6]

    def _ask(self, msgs):
        from isharati.glossing.asl import parse_glosses
        from isharati.llm import GlossError
        raw = ""
        for _ in range(3):  # reasoning models sometimes return an empty or cut-off reply
            raw = self.client.chat(msgs)
            glosses = parse_glosses(raw)
            if glosses:
                return glosses
        raise GlossError(f"malformed LLM output after 3 attempts: {raw[:120]!r}")

    def _skipped(self, text, glosses):
        """Words of the text whose sign is in the lexicon but not in the model's glosses."""
        used = {normalize_ar(str(g.get("text", ""))) for g in glosses}
        used |= {normalize_ar(e.gloss) for g in glosses for e in [self.lexicon.match_token(str(g.get("text", "")))] if e}
        out = []
        for tok in normalize_ar(text).split():
            if tok in STOPWORDS or any(c in STOPWORDS for c in candidates(tok)):
                continue
            e = self.lexicon.match_token(tok)
            if e is not None and not e.is_letter and normalize_ar(e.gloss) not in used and e.gloss not in out:
                out.append(e.gloss)
        return out
