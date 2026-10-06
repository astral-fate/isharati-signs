"""English text -> ASL gloss sequence. Every non-OOV gloss is guaranteed to exist in the ASL lexicon;
anything else is fingerspelled. Same GlossResult contract as isharati.llm."""
from isharati.text.english import STOPWORDS, normalize_en, tokens
from isharati.llm import require_text as _require_text
from isharati.types import AlignRow, GlossItem, GlossResult


class ASLRuleGlosser:
    """Word order kept, function words dropped, each word matched by form or lemma."""
    name = "rule"

    def __init__(self, lexicon):
        self.lexicon = lexicon

    def gloss(self, text) -> GlossResult:
        text = _require_text(text)
        items, trace = [], []
        for tok in tokens(text):
            norm = normalize_en(tok)
            if norm in STOPWORDS:
                trace.append(AlignRow(tok, norm, "stopword"))
                continue
            e = self.lexicon.match_token(tok)
            if e is not None:
                method = "exact" if self.lexicon.lookup(tok) is e else "stem"
                items.append(GlossItem(e.gloss))
                trace.append(AlignRow(tok, norm, method, e.gloss, e.sign_id, e.review_status))
            else:
                items.append(GlossItem(tok, oov=True))
                trace.append(AlignRow(tok, norm, "fingerspell"))
        return GlossResult(items, self.name, trace)


from isharati.glossing.llm import ASL_SPEC, LLMGlosser, parse_glosses  # noqa: F401  (re-exported)

SYSTEM = ASL_SPEC.system


class ASLLLMGlosser(LLMGlosser):
    """English -> ASL (the glosser the paper's runs used); the shared code is isharati.glossing.llm."""
    def __init__(self, lexicon, client):
        super().__init__(lexicon, client, ASL_SPEC)
