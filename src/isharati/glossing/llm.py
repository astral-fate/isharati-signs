"""Text -> sign glosses with an LLM, restricted to one lexicon's vocabulary. The language comes from a GlossSpec:
English -> ASL, Turkish -> TİD (Turkish glosses), Urdu -> ISL (English glosses). The model proposes; the lexicon
decides what is signed (anything it cannot match is marked missing)."""
import json
import re
from dataclasses import dataclass
from typing import Callable

from isharati.llm import GlossError, require_text
from isharati.text.english import normalize_en
from isharati.text.turkish import tr_norm
from isharati.retrieval.turkish_urdu import TR_STOP as _TR_STOP, UR_STOP as _UR_STOP  # function words
from isharati.types import AlignRow, GlossItem, GlossResult

_JSON = 'Return JSON only: {"glosses": [{"text": ..., "oov": false}]}'


@dataclass(frozen=True)
class GlossSpec:
    system: str
    normalize: Callable[[str], str]
    name: str = "llm"
    stop: frozenset = frozenset()  # function words: when the model returns one without a sign, it is dropped, not
    #                                reported as a missing sign (Turkish «ve», a split-off case suffix «-den»)


ASL_SPEC = GlossSpec(
    "You convert English text into an American Sign Language (ASL) gloss sequence. "
    "Use ONLY glosses from the allowed list, in ASL order (topic first, time signs early, no articles or copula), "
    "and keep the meaning. Prefer a listed synonym over fingerspelling when the meaning is the same. "
    "Names and words with no listed gloss must be output as {\"text\": <the word>, \"oov\": true} so they are "
    "fingerspelled; keep those short. " + _JSON, normalize_en)

TID_SPEC = GlossSpec(
    "You convert Turkish text into a Turkish Sign Language (TİD) gloss sequence. The glosses are Turkish words: use "
    "ONLY glosses from the allowed list, in TİD order (topic first, time signs early, no suffixes or copula), and keep "
    "the meaning. Prefer a listed synonym when the meaning is the same. Words with no listed gloss must be output as "
    "{\"text\": <the word>, \"oov\": true}; keep those short. " + _JSON, tr_norm,
    stop=frozenset(_TR_STOP | {"den", "dan", "ten", "tan", "nin", "nın", "nun", "nün", "yi", "yı"}))

ISL_SPEC = GlossSpec(
    "You convert Urdu text into an Indian Sign Language (ISL) gloss sequence. The glosses are English words: use ONLY "
    "glosses from the allowed list, in ISL order (subject, object, verb; time signs early; no articles, postpositions "
    "or copula), and keep the meaning. Prefer a listed synonym when the meaning is the same. Urdu words with no listed "
    "gloss must be output as {\"text\": <the Urdu word>, \"oov\": true}; keep those short. " + _JSON, normalize_en,
    stop=frozenset(_UR_STOP))

_ITEM = re.compile(r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"oov"\s*:\s*(true|false)\s*\}')


def parse_glosses(raw: str) -> list[dict]:
    """The model's {"glosses": [...]}; if the JSON is malformed (a missing comma, a cut-off tail), the
    well-formed {"text": ..., "oov": ...} items are taken one by one, in order."""
    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s:
        try:
            glosses = json.loads(raw[s:e + 1]).get("glosses")
            if isinstance(glosses, list):
                return glosses
        except (json.JSONDecodeError, AttributeError):
            pass
    return [{"text": json.loads(f'"{t}"'), "oov": o == "true"} for t, o in _ITEM.findall(raw)]


def merge_phrase_items(texts: list[str], lexicon, longest: int = 4) -> list[str]:
    """Consecutive model outputs that together name one multi-word sign, joined back into it: the model returned
    «kelime-i», «şehadet» for the gloss «kelime i şehadet», and both came out as missing (red) signs. Greedy,
    longest window first; a window is merged only when the lexicon has a sign for the joined text whose gloss is itself
    several words (so two unrelated signs are never fused into a different one)."""
    out, i = [], 0
    while i < len(texts):
        for n in range(min(longest, len(texts) - i), 1, -1):
            joined = " ".join(texts[i:i + n])
            e = lexicon.lookup(joined)
            if e is not None and len(e.gloss.replace("-", " ").split()) > 1:
                out.append(e.gloss)
                i += n
                break
        else:
            out.append(texts[i])
            i += 1
    return out


class LLMGlosser:
    def __init__(self, lexicon, client, spec: GlossSpec = ASL_SPEC):
        self.lexicon, self.client, self.spec = lexicon, client, spec
        self.name = spec.name

    def gloss(self, text) -> GlossResult:
        text = require_text(text)
        rewrite = getattr(self.lexicon, "rewrite_formulas", None)  # TİD: a fixed formula -> its one sign's gloss
        if rewrite:
            text = rewrite(text)
        norm = self.spec.normalize
        allowed = ", ".join(sorted({norm(e.gloss.rstrip("0123456789")) for e in self.lexicon.sign_entries()}))
        msgs = [{"role": "system", "content": self.spec.system},
                {"role": "user", "content": f"Allowed glosses: {allowed}\n\nText: {text}"}]
        # reasoning models sometimes return an empty or cut-off reply; ask again before giving up
        raw, glosses = "", []
        for attempt in range(3):
            raw = self.client.chat(msgs)
            glosses = parse_glosses(raw)
            if glosses:
                break
            msgs = msgs[:1] + [{"role": "user", "content": msgs[1]["content"] +
                                "\n\nReply with the JSON object only, no reasoning."}]
        if not glosses:
            raise GlossError(f"malformed LLM output after 3 attempts: {raw[:120]!r}")
        items, trace = [], []
        texts = [str(g.get("text", "")).strip() for g in glosses]
        texts = merge_phrase_items([t for t in texts if t], self.lexicon)
        for t in texts:
            entry = self.lexicon.match_token(t)  # the lexicon, not the model's flag, decides
            if entry:
                items.append(GlossItem(entry.gloss))
                trace.append(AlignRow(t, norm(t), "llm", entry.gloss, entry.sign_id, entry.review_status))
            elif norm(t) in self.spec.stop or t in self.spec.stop:
                trace.append(AlignRow(t, norm(t), "stopword"))  # a function word: not signed, not a missing sign
            else:
                items.append(GlossItem(t, oov=True))
                trace.append(AlignRow(t, norm(t), "fingerspell"))
        return GlossResult(items, self.name, trace)


class FallbackGlosser:
    """Text to sign must not fail just because the free LLM quota is used up: when every LLM backend fails, gloss with
    the rule-based glosser instead (word by word, no reordering; the report shows backend "rule")."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.name = getattr(primary, "name", "llm")

    def gloss(self, text) -> GlossResult:
        try:
            return self.primary.gloss(text)
        except GlossError:
            return self.fallback.gloss(text)
