# Isharati Landing Page and Four-Language Translator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dark, animated React landing page with the translator embedded, real avatar signing clips, live lexicon
figures, and Turkish → TİD and Urdu → ISL added next to English → ASL and Arabic → ArSL.

**Architecture:** The Python pipeline gains language-aware lexicons and a language-parameterised LLM glosser, so
`Pipeline("tr" | "ur")` works like English. FastAPI keeps the API and serves a Vite-built React app from `web/dist`
(old page at `/classic`). The React app reuses the existing three.js avatar module (`src/isharati/app/static/avatar.js`)
through a Vite alias, so there is one avatar implementation. Clips are rendered offline by a headless recorder page and
encoded with ffmpeg.

**Tech Stack:** Python 3.11+ (FastAPI, numpy, pytest); Node 22, React 18, Vite 5, TypeScript 5, Framer Motion 11,
three 0.169.0, @pixiv/three-vrm 3.1.0, Vitest + Testing Library; Playwright (Python) and ffmpeg for clips and e2e.

**Spec:** `docs/superpowers/specs/2026-10-02-landing-page-design.md`

## Global Constraints

- Python tests run with `py -m pytest` from the repo root (the Python 3.14 install that has FastAPI). Four tests
  (`tests/test_extract.py` ×3, `tests/test_render.py` ×1) already fail on this machine because mediapipe and the
  ffmpeg writer are missing; they are not regressions. Every other test must pass.
- Hugging Face commands run with `env -u HF_TOKEN` (the system `HF_TOKEN` is stale). Never print or echo API keys.
- WSLP (CC BY-NC-ND) is used only behind `ISHARATI_URDU_WSLP` (default on, `0`/`false`/`no` turns it off); the Space
  stays private. Isharah poses are never used or published.
- Library versions are pinned to the classic page's: `three@0.169.0`, `@pixiv/three-vrm@3.1.0`.
- Interface strings exist in four languages (`en`, `ar`, `tr`, `ur`); `ar` and `ur` render right-to-left. Every key in
  `en.json` exists in the other three.
- Every number on the page comes from `src/isharati/app/static/stats.json` (generated from data) or the API; coverage
  is `all.content_token_coverage` from `results/eda.json` (Urdu: `isl_on_en`).
- Clips: 3–6 s, ≤ 1.5 MB per file, ≤ 15 MB for the whole `web/public/clips/` folder.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MCZ7kRB6SCynLU61xgepiB
  ```
- Never edit files under `D:\islam\isharati` (the old repo); `data/` is a junction to it and is read-only for this work.

## Review Focus

1. **RTL interface over an LTR answer:** switching the interface to Arabic or Urdu while an English or Turkish answer
   is shown must keep the answer, chips and question box in the sign language's own direction, not the UI's
   (test: Task 10, `textDir`).
2. **Poses with non-finite values or a single frame:** the keypoint canvas must still fit and draw (test: Task 8,
   `poseBounds`).
3. **A saved avatar that no longer exists** (e.g. renamed VRM): the picker falls back to the default avatar instead of a
   blank canvas (test: Task 9, `pickAvatar`).
4. **No clips yet** (fresh checkout before Task 12, or `clips.json` missing on the Space): the gallery and hero must
   render avatars without video, not fail (test: Task 7, `getClips` returns `[]`).
5. **Proxy or sign-in HTML instead of JSON, and the free-quota 503:** shown as localised messages, never a JSON parse
   error (test: Task 7, `ApiError` kinds).

---

## File structure

**Python (modify / create)**

| File | Responsibility |
|---|---|
| `src/isharati/text/turkish.py` (create) | `tr_norm` (moved from `retrieval/turkish_urdu.py`) |
| `src/isharati/retrieval/turkish_urdu.py` (modify) | imports `tr_norm` from `text/turkish.py` |
| `src/isharati/lexicon/asl.py` (modify) | `ASLLexicon` key and candidate functions become overridable |
| `src/isharati/lexicon/turkish.py` (create) | `TIDLexicon`, `tid_lexicon()` |
| `src/isharati/lexicon/isl.py` (create) | `isl_lexicon()`, `wslp_enabled()` |
| `src/isharati/glossing/llm.py` (create) | `GlossSpec`, `LLMGlosser`, `parse_glosses`, `ASL_SPEC`, `TID_SPEC`, `ISL_SPEC` |
| `src/isharati/glossing/asl.py` (modify) | `ASLLLMGlosser` becomes a thin subclass; re-exports `parse_glosses`, `SYSTEM` |
| `src/isharati/backtranslate.py` (modify) | `SampledRecognizer` |
| `src/isharati/pipeline.py` (modify) | `tr`/`ur` branches, `lexicon_paths`, `load_lexicon`, stamp |
| `scripts/eval/app_stats.py` (create) | writes `src/isharati/app/static/stats.json` |
| `src/isharati/app/server.py` (modify) | `lang` set, `/api/stats`, React dist serving, `/classic`, `/record` |
| `scripts/avatars/clip_poses.py` (create) | stitched phrase poses for clips and the hero |
| `scripts/avatars/record_clips.py` (create) | headless recording + ffmpeg → `web/public/clips/` |
| `scripts/web/e2e.py` (create) | Playwright end-to-end check and screenshots |
| `scripts/hub/deploy_space.py`, `deploy/space/Dockerfile` (modify) | stage and build `web/` |
| tests: `tests/test_languages.py` (create), `tests/test_app.py`, `tests/test_hub_deploy.py` (modify) | |

**Front end (`web/`, all new)**

| File | Responsibility |
|---|---|
| `package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `record.html` | build setup |
| `src/main.tsx`, `src/App.tsx`, `src/styles.css`, `src/test-setup.ts` | entry, assembly, theme |
| `src/types.ts`, `src/api.ts`, `src/static.d.ts` | API types and calls, typings for the shared avatar module |
| `src/i18n/i18n.tsx`, `src/i18n/{en,ar,tr,ur}.json` | interface strings, RTL |
| `src/state.tsx` | app state shared across sections (selected avatar, stats, clips) |
| `src/player.ts` | `usePlayer` clock (play, pause, seek, speed, loop) |
| `src/components/Counter.tsx`, `KeypointCanvas.tsx`, `AvatarView.tsx`, `AvatarPicker.tsx`, `GlossChips.tsx`, `ClipCard.tsx` | reusable pieces |
| `src/sections/Nav.tsx`, `Hero.tsx`, `Translator.tsx`, `HowItWorks.tsx`, `Lexicon.tsx`, `Avatars.tsx`, `Sources.tsx`, `Footer.tsx` | page sections |
| `src/record.ts` | clip recorder page script |
| `src/assets/hero-pose.json`, `public/clips/*` | generated by Task 12 |
| `src/**/*.test.ts(x)` | Vitest |

---

### Task 1: Language-aware lexicons (Turkish TİD, Urdu ISL)

**Files:**
- Create: `src/isharati/text/turkish.py`, `src/isharati/lexicon/turkish.py`, `src/isharati/lexicon/isl.py`, `tests/test_languages.py`
- Modify: `src/isharati/retrieval/turkish_urdu.py` (the `TR_LOWER`/`tr_norm` block), `src/isharati/lexicon/asl.py:21-48`

**Interfaces:**
- Produces: `isharati.text.turkish.tr_norm(text: str) -> str`;
  `isharati.lexicon.turkish.TIDLexicon(entries: list[SignEntry], root: Path)` and `tid_lexicon(root: Path | None = None) -> TIDLexicon`;
  `isharati.lexicon.isl.wslp_enabled() -> bool`, `isl_lexicon(root: Path | None = None, wslp: bool | None = None) -> ASLLexicon`;
  `ASLLexicon._key` (staticmethod, default `gloss_key`) and `ASLLexicon._candidates` (staticmethod, default `candidates`).
  All lexicons keep the interface `lookup(gloss)`, `match_token(token)`, `keypoints(entry)`, `sign_entries()`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_languages.py`:

```python
"""Turkish (TİD) and Urdu (ISL) lexicons, glossers and back-translation sampling."""
import json

import numpy as np
import pytest


def _write(root, folder, file, rows):
    d = root / folder
    (d / "signs").mkdir(parents=True, exist_ok=True)
    out = []
    for i, (gloss, sid) in enumerate(rows):
        np.save(d / "signs" / f"{sid}.npy", np.full((6, 50, 3), i, np.float32))
        out.append({"gloss": gloss, "sign_id": sid, "dataset": folder, "keypoints_path": f"{folder}/signs/{sid}.npy",
                    "review_status": "pending", "is_religious": False, "is_letter": False})
    (d / file).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")


@pytest.fixture
def tid_root(tmp_path):
    _write(tmp_path, "dictionaries", "entries_tid_dictionary.jsonl",
           [("namaz", "n1"), ("oruç", "o1"), ("iman", "i1"), ("ışık", "k1"), ("abdest", "a1")])
    return tmp_path


@pytest.fixture
def isl_root(tmp_path):
    _write(tmp_path, "cislr", "entries.jsonl", [("PRAYER", "c1"), ("BOOK", "c2")])
    _write(tmp_path, "wslp", "entries.jsonl", [("PRAYER", "w1"), ("MOSQUE", "w2")])
    return tmp_path


def test_tr_norm_turkish_casing():
    from isharati.text.turkish import tr_norm
    assert tr_norm("İMAN") == "iman" and tr_norm("IŞIK") == "ışık" and tr_norm("Namaz!") == "namaz"


def test_tid_lexicon_matches_turkish_case_and_suffixes(tid_root):
    from isharati.lexicon.turkish import tid_lexicon
    lex = tid_lexicon(tid_root)
    assert lex.match_token("İMAN").sign_id == "i1"
    assert lex.match_token("IŞIK").sign_id == "k1"
    assert lex.match_token("namazı").sign_id == "n1"      # suffix: first five letters
    assert lex.match_token("abdesti").sign_id == "a1"
    assert lex.lookup("Oruç").sign_id == "o1"
    assert lex.match_token("kitap") is None
    assert lex.keypoints(lex.lookup("namaz")).shape == (6, 50, 3)


def test_isl_lexicon_wslp_flag(isl_root, monkeypatch):
    from isharati.lexicon.isl import isl_lexicon, wslp_enabled
    on = isl_lexicon(isl_root, wslp=True)
    assert on.lookup("PRAYER").sign_id == "c1"             # CISLR wins a tie
    assert on.lookup("MOSQUE").sign_id == "w2"
    off = isl_lexicon(isl_root, wslp=False)
    assert off.lookup("MOSQUE") is None and len(off.sign_entries()) == 2
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "0")
    assert not wslp_enabled()
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "1")
    assert wslp_enabled()
    monkeypatch.delenv("ISHARATI_URDU_WSLP")
    assert wslp_enabled()                                   # on by default while the Space is private
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -m pytest tests/test_languages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'isharati.text.turkish'`.

- [ ] **Step 3: Implement.** Create `src/isharati/text/turkish.py`:

```python
"""Turkish text normalisation: Turkish casing (I -> ı, İ -> i) before lower-casing, letters only."""
import re
import unicodedata

TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def tr_norm(text):
    t = unicodedata.normalize("NFC", text or "").translate(TR_LOWER).lower()
    return re.sub(r"[^\wçğıöşüâîû]+", " ", t).strip()
```

In `src/isharati/retrieval/turkish_urdu.py`, delete the `TR_LOWER = ...` line and the `def tr_norm(...)` function
(the block under `# ---- Turkish`), and add after the existing imports:

```python
from isharati.text.turkish import tr_norm  # noqa: F401  (also used by isharati.lexicon.turkish)
```

In `src/isharati/lexicon/asl.py`, make the key and candidates overridable. Replace the body of `ASLLexicon` from
`def __init__` through `match_token` with:

```python
class ASLLexicon:
    _key = staticmethod(gloss_key)          # gloss -> lookup key (overridden for Turkish)
    _candidates = staticmethod(candidates)  # token -> keys to try, longest first

    def __init__(self, entries: list[SignEntry], root: Path):
        self.root = Path(root)
        self._signs: dict[str, SignEntry] = {}
        for e in entries:
            if not e.is_letter:
                self._signs.setdefault(self._key(e.gloss), e)  # first variant wins (the builder orders them)
        self._letters = {e.gloss.lower(): e for e in entries if e.is_letter}
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, path: Path) -> "ASLLexicon":
        path = Path(path)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls([SignEntry(**r) for r in rows], path.parent)

    def lookup(self, gloss: str) -> SignEntry | None:
        return self._signs.get(self._key(gloss))

    def match_token(self, token: str) -> SignEntry | None:
        for c in self._candidates(token):
            if c in self._signs:
                return self._signs[c]
        return None
```

(`fingerspell`, `keypoints`, `sign_entries`, `gloss_keys` stay as they are.)

Create `src/isharati/lexicon/turkish.py`:

```python
"""Turkish Sign Language (TİD) lexicon: the TİD dictionary signs, glossed by Turkish words. Keys use Turkish casing;
a word that is not a key is matched by its first five letters (namazı -> namaz), the stem rule the Turkish search uses."""
import json
from pathlib import Path

from isharati.config import DATA
from isharati.lexicon.asl import ASLLexicon
from isharati.text.turkish import tr_norm
from isharati.types import SignEntry

TID_ROOT = DATA / "lexicon_tr"
TID_FILE = Path("dictionaries") / "entries_tid_dictionary.jsonl"


class TIDLexicon(ASLLexicon):
    _key = staticmethod(tr_norm)

    def __init__(self, entries, root):
        super().__init__(entries, root)
        self._prefix: dict[str, SignEntry] = {}
        for k, e in self._signs.items():
            if len(k) > 5:
                self._prefix.setdefault(k[:5], e)

    def match_token(self, token):
        n = tr_norm(token)
        if n in self._signs:
            return self._signs[n]
        if len(n) > 5:
            return self._signs.get(n[:5]) or self._prefix.get(n[:5])
        return None


def tid_lexicon(root: Path | None = None) -> TIDLexicon:
    root = Path(root or TID_ROOT)
    rows = [json.loads(l) for l in (root / TID_FILE).read_text(encoding="utf-8").splitlines() if l.strip()]
    return TIDLexicon([SignEntry(**r) for r in rows], root)  # keypoints_path is relative to lexicon_tr/
```

Create `src/isharati/lexicon/isl.py`:

```python
"""Indian Sign Language lexicon for Urdu answers: CISLR (AFL-3.0) and, while ISHARATI_URDU_WSLP is on, WSLP
(CC BY-NC-ND 4.0; keep it off for any public deployment). Glosses are English; CISLR wins a tie, so turning WSLP off
only removes signs."""
import json
import os
from pathlib import Path

from isharati.config import DATA
from isharati.lexicon.asl import ASLLexicon
from isharati.types import SignEntry

ISL_ROOT = DATA / "lexicon_isl"
SOURCES = ("cislr", "wslp")


def wslp_enabled() -> bool:
    return os.environ.get("ISHARATI_URDU_WSLP", "1").strip().lower() not in ("0", "false", "no", "off", "")


def entries_files(root: Path | None = None, wslp: bool | None = None) -> list[Path]:
    root = Path(root or ISL_ROOT)
    use = SOURCES if (wslp_enabled() if wslp is None else wslp) else SOURCES[:1]
    return [root / s / "entries.jsonl" for s in use if (root / s / "entries.jsonl").exists()]


def isl_lexicon(root: Path | None = None, wslp: bool | None = None) -> ASLLexicon:
    root = Path(root or ISL_ROOT)
    rows = [json.loads(l) for f in entries_files(root, wslp) for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return ASLLexicon([SignEntry(**r) for r in rows], root)  # keypoints_path is relative to lexicon_isl/
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -m pytest tests/test_languages.py tests/test_asl.py -v`
Expected: all PASS (the ASL tests prove the `ASLLexicon` refactor changed nothing).

- [ ] **Step 5: Commit**

```bash
git add src/isharati/text/turkish.py src/isharati/lexicon/turkish.py src/isharati/lexicon/isl.py \
        src/isharati/lexicon/asl.py src/isharati/retrieval/turkish_urdu.py tests/test_languages.py
git commit -m "Lexicons: Turkish TİD with Turkish casing, Urdu ISL from CISLR and WSLP behind a flag" -m "<attribution lines>"
```

---

### Task 2: Language-parameterised LLM glosser

**Files:**
- Create: `src/isharati/glossing/llm.py`
- Modify: `src/isharati/glossing/asl.py:37-110` (SYSTEM, parse_glosses, ASLLLMGlosser)
- Test: `tests/test_languages.py` (append)

**Interfaces:**
- Consumes: `TIDLexicon`, `isl_lexicon` (Task 1); `isharati.types.GlossItem, GlossResult, AlignRow`.
- Produces: `GlossSpec(system: str, normalize: Callable[[str], str], name: str = "llm")`;
  `LLMGlosser(lexicon, client, spec: GlossSpec = ASL_SPEC)` with `.gloss(text) -> GlossResult`;
  `ASL_SPEC`, `TID_SPEC`, `ISL_SPEC`; `parse_glosses(raw) -> list[dict]`.
  `isharati.glossing.asl.ASLLLMGlosser(lexicon, client)` and `isharati.glossing.asl.parse_glosses` keep working.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_languages.py`:

```python
class Reply:
    """A fake LLM client: records the messages and returns a fixed reply."""
    def __init__(self, raw):
        self.raw, self.seen = raw, []

    def chat(self, msgs):
        self.seen.append(msgs)
        return self.raw


def test_turkish_glosser_uses_tid_vocabulary(tid_root):
    from isharati.glossing.llm import LLMGlosser, TID_SPEC
    from isharati.lexicon.turkish import tid_lexicon
    c = Reply('{"glosses": [{"text": "NAMAZ", "oov": false}, {"text": "İman", "oov": false},'
              ' {"text": "kitap", "oov": false}]}')
    r = LLMGlosser(tid_lexicon(tid_root), c, TID_SPEC).gloss("Namaz imanın direğidir.")
    assert r.glosses == ["namaz", "iman"] and r.oov == ["kitap"]   # the lexicon decides, not the model
    system, user = c.seen[0][0]["content"], c.seen[0][1]["content"]
    assert "Turkish Sign Language" in system and "namaz" in user and "ışık" in user


def test_urdu_glosser_uses_english_isl_glosses(isl_root):
    from isharati.glossing.llm import ISL_SPEC, LLMGlosser
    from isharati.lexicon.isl import isl_lexicon
    c = Reply('{"glosses": [{"text": "PRAYER", "oov": false}, {"text": "جماعت", "oov": true}]}')
    r = LLMGlosser(isl_lexicon(isl_root, wslp=False), c, ISL_SPEC).gloss("نماز جماعت کے ساتھ")
    assert r.glosses == ["PRAYER"] and r.oov == ["جماعت"]
    assert "Indian Sign Language" in c.seen[0][0]["content"]
    assert "mosque" not in c.seen[0][1]["content"].lower()           # WSLP off: its signs are never offered


def test_asl_glosser_unchanged_alias():
    from isharati.glossing import asl, llm
    assert asl.parse_glosses is llm.parse_glosses and asl.SYSTEM == llm.ASL_SPEC.system
    assert issubclass(asl.ASLLLMGlosser, llm.LLMGlosser)
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -m pytest tests/test_languages.py -v -k glosser`
Expected: FAIL with `ModuleNotFoundError: No module named 'isharati.glossing.llm'`.

- [ ] **Step 3: Implement.** Create `src/isharati/glossing/llm.py` (the body of `ASLLLMGlosser.gloss` moves here unchanged
except that the prompt and normaliser come from the spec):

```python
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
from isharati.types import AlignRow, GlossItem, GlossResult

_JSON = 'Return JSON only: {"glosses": [{"text": ..., "oov": false}]}'


@dataclass(frozen=True)
class GlossSpec:
    system: str
    normalize: Callable[[str], str]
    name: str = "llm"


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
    "{\"text\": <the word>, \"oov\": true}; keep those short. " + _JSON, tr_norm)

ISL_SPEC = GlossSpec(
    "You convert Urdu text into an Indian Sign Language (ISL) gloss sequence. The glosses are English words: use ONLY "
    "glosses from the allowed list, in ISL order (subject, object, verb; time signs early; no articles, postpositions "
    "or copula), and keep the meaning. Prefer a listed synonym when the meaning is the same. Urdu words with no listed "
    "gloss must be output as {\"text\": <the Urdu word>, \"oov\": true}; keep those short. " + _JSON, normalize_en)

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


class LLMGlosser:
    def __init__(self, lexicon, client, spec: GlossSpec = ASL_SPEC):
        self.lexicon, self.client, self.spec = lexicon, client, spec
        self.name = spec.name

    def gloss(self, text) -> GlossResult:
        text = require_text(text)
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
        for g in glosses:
            t = str(g.get("text", "")).strip()
            if not t:
                continue
            entry = self.lexicon.match_token(t)  # the lexicon, not the model's flag, decides
            if entry:
                items.append(GlossItem(entry.gloss))
                trace.append(AlignRow(t, norm(t), "llm", entry.gloss, entry.sign_id, entry.review_status))
            else:
                items.append(GlossItem(t, oov=True))
                trace.append(AlignRow(t, norm(t), "fingerspell"))
        return GlossResult(items, self.name, trace)
```

In `src/isharati/glossing/asl.py`, delete the `SYSTEM = (...)`, `_ITEM`, `parse_glosses` and `class ASLLLMGlosser`
definitions and add instead:

```python
from isharati.glossing.llm import ASL_SPEC, LLMGlosser, parse_glosses  # noqa: F401  (re-exported)

SYSTEM = ASL_SPEC.system


class ASLLLMGlosser(LLMGlosser):
    """English -> ASL (the glosser the paper's runs used); the shared code is isharati.glossing.llm."""
    def __init__(self, lexicon, client):
        super().__init__(lexicon, client, ASL_SPEC)
```

Remove imports in `asl.py` that are now unused (`json`, `re`, `GlossError`, `_require_text`) only if the remaining
`ASLRuleGlosser` does not use them; run `py -m pyflakes src/isharati/glossing/asl.py` if pyflakes is installed.

- [ ] **Step 4: Run the tests**

Run: `py -m pytest tests/test_languages.py tests/test_asl.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/isharati/glossing/llm.py src/isharati/glossing/asl.py tests/test_languages.py
git commit -m "Glossing: one LLM glosser for ASL, TİD and ISL, chosen by a GlossSpec" -m "<attribution lines>"
```

---

### Task 3: Turkish and Urdu in the pipeline and the API

**Files:**
- Modify: `src/isharati/backtranslate.py` (add `SampledRecognizer`), `src/isharati/pipeline.py:25-110`, `src/isharati/app/server.py:51-52`
- Test: `tests/test_languages.py`, `tests/test_app.py` (append)

**Interfaces:**
- Consumes: `tid_lexicon`, `isl_lexicon`, `wslp_enabled`, `entries_files` (Task 1); `LLMGlosser`, `TID_SPEC`, `ISL_SPEC` (Task 2); `turkish_urdu.corpus(lang)`.
- Produces: `SampledRecognizer(lexicon, n_distractors: int = 200, seed: int = 0)` with `.for_glosses(glosses) -> DTWRecognizer`;
  `pipeline.LANGS = ("en", "ar", "tr", "ur")`; `pipeline.lexicon_paths(lang) -> list[Path]`;
  `pipeline.load_lexicon(lang)` (lexicon with `proportions.apply` already applied); `Pipeline("tr" | "ur")`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_languages.py`:

```python
def test_sampled_recognizer_keeps_answer_signs(lexicon):
    from isharati.backtranslate import SampledRecognizer
    rec = SampledRecognizer(lexicon, n_distractors=2).for_glosses(["صلاة", "غير موجود"])
    labels = [g for g, _ in rec.templates]
    assert "صلاة" in labels and len(labels) <= 3


def test_stamp_and_ids_cover_new_languages(monkeypatch, tmp_path):
    from isharati import pipeline
    assert pipeline.LANGS == ("en", "ar", "tr", "ur")
    assert pipeline.question_id("Ramazan nedir?", "tr").startswith("tr")
    assert pipeline.question_id("رمضان کیا ہے؟", "ur").startswith("ur")
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "1")
    on = pipeline.stamp("ur")
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "0")
    assert pipeline.stamp("ur") != on                     # turning WSLP off invalidates saved Urdu answers
```

Append to `tests/test_app.py`:

```python
def test_turkish_and_urdu_replay(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    for lang, q in (("tr", "Ramazan nedir?"), ("ur", "رمضان کیا ہے؟")):
        run = tmp_path / server.question_id(q, lang)
        run.mkdir()
        report = {"status": "signed", "version": server.REPORT_VERSION, "stamp": server.stamp(lang), "answer": "x",
                  "glosses": ["a"]}
        (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
        monkeypatch.setattr(server, "pipeline", lambda lang: (_ for _ in ()).throw(AssertionError("no pipeline")))
        r = c.post("/api/ask", json={"question": q, "lang": lang})
        assert r.status_code == 200 and r.json()["cached"]


def test_unknown_language_rejected(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    assert c.post("/api/ask", json={"question": "hi", "lang": "fr"}).status_code == 400
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -m pytest tests/test_languages.py tests/test_app.py -v`
Expected: FAIL (`ImportError: cannot import name 'SampledRecognizer'`, `AttributeError: ... 'LANGS'`, and `400` for `tr`).

- [ ] **Step 3: Implement.** In `src/isharati/backtranslate.py` add `import random` at the top and, after `class DTWRecognizer`:

```python
class SampledRecognizer:
    """For lexicons without held-out templates and too large for 1-NN over every sign (Urdu ISL has 8,000): the
    answer's own signs plus a fixed sample of distractors. Circular, like from_lexicon, and labelled so."""

    def __init__(self, lexicon, n_distractors: int = 200, seed: int = 0):
        entries = sorted(lexicon.sign_entries(), key=lambda e: e.sign_id)
        self.lexicon = lexicon
        self.distractors = random.Random(seed).sample(entries, min(n_distractors, len(entries)))

    def for_glosses(self, glosses) -> "DTWRecognizer":
        chosen = {e.sign_id: e for e in self.distractors}
        for g in glosses:
            e = self.lexicon.lookup(g)
            if e:
                chosen[e.sign_id] = e
        return DTWRecognizer([(e.gloss, self.lexicon.keypoints(e)) for e in chosen.values()],
                             "dtw-1nn-lexicon-sample (circular)")
```

In `src/isharati/pipeline.py`:

1. Change the import line to `from isharati.backtranslate import DTWRecognizer, SampledRecognizer, score` and update
   the module docstring's usage block with:
   `python -m isharati.pipeline --lang tr "Ramazan nedir?"   # Turkish -> TİD` and
   `python -m isharati.pipeline --lang ur "رمضان کیا ہے؟"     # Urdu -> ISL`.
2. Replace `stamp` with:

```python
LANGS = ("en", "ar", "tr", "ur")


def lexicon_paths(lang: str) -> list[Path]:
    if lang == "ar":
        return [AR_LEX]
    if lang == "tr":
        from isharati.lexicon.turkish import TID_FILE, TID_ROOT
        return [TID_ROOT / TID_FILE]
    if lang == "ur":
        from isharati.lexicon.isl import entries_files
        return entries_files()
    return [LEX / "lexicon_all.jsonl"]


def stamp(lang: str = "en") -> str:
    """Changes whenever the lexicon or the answering/glossing code changes; a saved answer is replayed only while its
    stamp still matches (server.py), so fixes and new signs show up instead of an old result."""
    pkg = Path(__file__).parent
    code = [pkg / "retrieval" / "engine.py", pkg / "retrieval" / "arabic.py", pkg / "retrieval" / "turkish_urdu.py",
            pkg / "glossing" / "asl.py", pkg / "glossing" / "llm.py", pkg / "glossing" / "arabic.py",
            pkg / "text" / "arabic_morph.py", pkg / "lexicon" / "arabic.py", pkg / "lexicon" / "turkish.py",
            pkg / "lexicon" / "isl.py", pkg / "pose" / "poser.py", Path(__file__)]
    files = [*lexicon_paths(lang), *code]
    parts = [f"{f.name}:{f.stat().st_mtime_ns}:{f.stat().st_size}" for f in files if f.exists()]
    parts.append(f"files:{len(lexicon_paths(lang))}")  # Urdu: CISLR alone or CISLR + WSLP
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
```

   (With WSLP present on disk, `entries_files()` returns one file when the flag is off and two when on, so the stamp
   differs. When the data is absent — e.g. CI — both lists are empty for `ur` and the test's two stamps would be equal;
   the test runs where `data/` exists. If you need it to run without data, monkeypatch `lexicon_paths`.)

3. Add `load_lexicon` above `class Pipeline` and use it in `__init__`:

```python
def load_lexicon(lang: str):
    """The sign lexicon for a language, with one body shape across its sources (pose.proportions)."""
    from isharati.pose import proportions
    if lang == "ar":
        from isharati.lexicon import arabic as ar_lexicon
        lexicon = ar_lexicon.qa_lexicon(AR_LEX)
    elif lang == "tr":
        from isharati.lexicon.turkish import tid_lexicon
        lexicon = tid_lexicon()
    elif lang == "ur":
        from isharati.lexicon.isl import isl_lexicon
        lexicon = isl_lexicon()
    else:
        merged = LEX / "lexicon_all.jsonl"  # Islamic signs first, then dictionaries (scripts/lexicon/asl/merge.py)
        lexicon = ASLLexicon.load(merged if merged.exists() else LEX / "lexicon.jsonl")
    return proportions.apply(lexicon)
```

   Replace `Pipeline.__init__` with:

```python
    def __init__(self, lang: str = "en", client=None):
        if lang not in LANGS:
            raise ValueError(f"unsupported language {lang!r}")
        self.lang = lang
        self.client = client or default_client()
        self.lexicon = load_lexicon(lang)
        templates = None
        if lang == "ar":  # Arabic question -> Arabic sources -> Arabic Sign Language lexicon
            from isharati.glossing import arabic as ar_glossing
            from isharati.retrieval import arabic as ar_retrieval
            self.corpus, templates = ar_retrieval.corpus(), AR_TEMPLATES
            self.glosser = ar_glossing.ArabicQAGlosser(self.lexicon, self.client)  # LLM only, no rule-based glossing
        elif lang in ("tr", "ur"):  # Turkish -> TİD, Urdu -> ISL (English glosses)
            from isharati.glossing.llm import ISL_SPEC, TID_SPEC, LLMGlosser
            from isharati.retrieval import turkish_urdu
            self.corpus = turkish_urdu.corpus(lang)
            self.glosser = LLMGlosser(self.lexicon, self.client, TID_SPEC if lang == "tr" else ISL_SPEC)
        else:
            self.corpus, templates = Corpus(), LEX / "templates.jsonl"
            self.glosser = ASLLLMGlosser(self.lexicon, self.client)
        if templates is not None and templates.exists():
            self.recognizer = DTWRecognizer.from_templates(templates)
        elif lang in ("tr", "ur"):
            self.recognizer = SampledRecognizer(self.lexicon)
        else:
            self.recognizer = DTWRecognizer.from_lexicon(self.lexicon)
```

   (This removes the old `from isharati.pose import proportions as proportions; proportions.apply(self.lexicon)` lines,
   now inside `load_lexicon`.)

4. In `_sign`, replace `bt = score(pose, segments, result.glosses, recognizer)` with:

```python
    if hasattr(recognizer, "for_glosses"):  # Turkish / Urdu: this answer's signs plus fixed distractors
        recognizer = recognizer.for_glosses(result.glosses)
    bt = score(pose, segments, result.glosses, recognizer)
```

5. In `main()`, accept the new languages (the `--lang` parsing already passes any value to `Pipeline`, which now
   validates it).

In `src/isharati/app/server.py`, update the import to
`from isharati.pipeline import LANGS, OUT_DIR as OUT, REPORT_VERSION, Pipeline, question_id, stamp  # noqa: E402`,
change `class Ask`'s comment to `# "en" ASL, "ar" ArSL, "tr" TİD, "ur" ISL`, and replace the language check with:

```python
    if body.lang not in LANGS:
        raise HTTPException(400, "lang must be one of " + ", ".join(LANGS))
```

- [ ] **Step 4: Run the tests**

Run: `py -m pytest tests/test_languages.py tests/test_app.py tests/test_asl.py tests/test_arabic.py -v`
Expected: all PASS.

- [ ] **Step 5: Live check (needs `data/` and an LLM key in `D:\islam\.env`)**

Run: `PYTHONPATH=src py -m isharati.pipeline --lang tr "Ramazan nedir?"` and
`PYTHONPATH=src py -m isharati.pipeline --lang ur "رمضان کیا ہے؟"`.
Expected: each prints a report with `"status": "signed"`, a cited answer, at least one gloss, and finishes in under
two minutes. Note the run times in the commit message body. If either takes longer than two minutes, lower
`SampledRecognizer`'s default `n_distractors` to 100 and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/isharati/backtranslate.py src/isharati/pipeline.py src/isharati/app/server.py tests/
git commit -m "Pipeline and API: Turkish -> TİD and Urdu -> ISL" -m "<live run times>" -m "<attribution lines>"
```

---

### Task 3a: Text to sign — `POST /api/sign`

**Files:**
- Modify: `src/isharati/glossing/llm.py` (add `FallbackGlosser`), `src/isharati/pipeline.py` (`question_id`, `_sign`, `Pipeline.sign_text`), `src/isharati/app/server.py`
- Test: `tests/test_languages.py`, `tests/test_app.py` (append)

**Interfaces:**
- Consumes: `Pipeline`, `_sign`, `ASLRuleGlosser` (`isharati.glossing.asl`), `GlossError` (`isharati.llm`).
- Produces: `FallbackGlosser(primary, fallback)` with `.gloss(text) -> GlossResult` (the fallback's result has
  `backend == "rule"`); `question_id(text, lang="en", kind="answer" | "text") -> str` (answer ids unchanged; text ids
  `"t" + lang + hash`); `Pipeline.sign_text(text) -> dict` (report with `"mode": "text"`, `"sources": []`);
  `POST /api/sign` body `{text: str, lang: str = "en", fresh: bool = False}` → the report, `400` for empty or
  over 1,000 characters or an unknown language, `422` when no word has a sign, `503` for the quota.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_languages.py`:

```python
def test_fallback_glosser_uses_rules_when_the_llm_fails():
    from isharati.glossing.llm import FallbackGlosser
    from isharati.llm import GlossError
    from isharati.types import GlossItem, GlossResult

    class Down:
        def gloss(self, text):
            raise GlossError("all LLM backends failed")

    class Rules:
        def gloss(self, text):
            return GlossResult([GlossItem("PRAYER")], "rule")

    r = FallbackGlosser(Down(), Rules()).gloss("Prayer is light.")
    assert r.glosses == ["PRAYER"] and r.backend == "rule"


def test_text_ids_never_collide_with_answer_ids():
    from isharati.pipeline import question_id
    q = "What is Ramadan?"
    assert question_id(q, "en") != question_id(q, "en", "text")
    assert question_id(q, "en", "text").startswith("ten") and question_id(q, "en", "text").isalnum()
    assert question_id(q, "en") == question_id(q)                  # earlier answer ids still replay
```

Append to `tests/test_app.py`:

```python
class _TextOnly:
    """A pipeline stand-in: signing a text must never call the answering step."""
    def __init__(self):
        self.signed = []

    def run(self, q):
        raise AssertionError("text to sign must not answer")

    def sign_text(self, text):
        self.signed.append(text)
        return {"status": "signed", "mode": "text", "answer": text, "sources": [], "glosses": ["PRAYER"], "id": "tenabc"}


def test_sign_text_skips_answering(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    fake = _TextOnly()
    monkeypatch.setattr(server, "pipeline", lambda lang: fake)
    r = c.post("/api/sign", json={"text": "  Prayer is light.  ", "lang": "en"})
    assert r.status_code == 200 and r.json()["mode"] == "text" and fake.signed == ["Prayer is light."]


def test_sign_text_limits_and_errors(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    assert c.post("/api/sign", json={"text": "", "lang": "en"}).status_code == 400
    assert c.post("/api/sign", json={"text": "x" * 1001, "lang": "en"}).status_code == 400
    assert c.post("/api/sign", json={"text": "hi", "lang": "fr"}).status_code == 400

    class NoSigns:
        def sign_text(self, text):
            raise ValueError("nothing in the lexicon to sign")
    monkeypatch.setattr(server, "pipeline", lambda lang: NoSigns())
    r = c.post("/api/sign", json={"text": "zzz qqq", "lang": "en"})
    assert r.status_code == 422 and "sign" in r.json()["detail"]


def test_signed_text_replays(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    text = "Prayer is light."
    run = tmp_path / server.question_id(text, "en", "text")
    run.mkdir()
    report = {"status": "signed", "mode": "text", "version": server.REPORT_VERSION, "stamp": server.stamp("en"),
              "answer": text, "glosses": ["PRAYER"]}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(server, "pipeline", lambda lang: (_ for _ in ()).throw(AssertionError("no pipeline")))
    r = c.post("/api/sign", json={"text": text, "lang": "en"})
    assert r.status_code == 200 and r.json()["cached"] and r.json()["mode"] == "text"
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -m pytest tests/test_languages.py tests/test_app.py -v -k "fallback or text_ids or sign_text or signed_text"`
Expected: FAIL (`ImportError: cannot import name 'FallbackGlosser'`, `TypeError` on `question_id(..., "text")`, `404` on `/api/sign`).

- [ ] **Step 3: Implement.** Append to `src/isharati/glossing/llm.py`:

```python
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
```

In `src/isharati/pipeline.py`, replace `question_id` with:

```python
def question_id(question: str, lang: str = "en", kind: str = "answer") -> str:
    """Output folder name; English answers keep the original ids so earlier runs replay. Signed texts (Studio) get
    their own ids, so a text never replays an answer to the same words."""
    h = hashlib.sha1(question.encode()).hexdigest()[:8]
    if kind == "text":
        return "t" + lang + h
    return h if lang == "en" else lang + h
```

Change `_sign`'s signature and the two lines that use the id and the report:

```python
def _sign(question, ans, lexicon, glosser, client, recognizer, lang="en", kind="answer"):
    ...
    out = OUT_DIR / question_id(question, lang, kind)
    ...
    report = {
        "status": "signed", "mode": kind, "version": REPORT_VERSION, "stamp": stamp(lang), "lang": lang, ...
```

(only `kind`, the `question_id` call and the `"mode": kind` entry change; the rest of `_sign` stays.)

Add to `class Pipeline`, after `run`:

```python
    def sign_text(self, text: str) -> dict:
        """The user's own text, signed as written: no retrieval, no answer generation, no sources."""
        text = text.strip()
        glosser = self.glosser
        if self.lang == "en":  # the rule glosser keeps English working when the free LLM quota runs out
            from isharati.glossing.asl import ASLRuleGlosser
            from isharati.glossing.llm import FallbackGlosser
            glosser = FallbackGlosser(self.glosser, ASLRuleGlosser(self.lexicon))
        ans = {"answer": text, "sources": [], "dropped_unsupported": []}
        return _sign(text, ans, self.lexicon, glosser, self.client, self.recognizer, self.lang, kind="text")
```

In `src/isharati/app/server.py`, factor the replay and the error mapping out of `ask` and add the endpoint:

```python
def _replay(saved: Path, lang: str):
    """A saved run made by this pipeline with this lexicon, or None (then it is regenerated)."""
    if not saved.exists():
        return None
    report = json.loads(saved.read_text(encoding="utf-8"))
    if report.get("version") == REPORT_VERSION and report.get("stamp") == stamp(lang):
        return {**report, "status": "signed", "id": saved.parent.name, "cached": True,
                "dropped_unsupported": report.get("dropped_unsupported", [])}
    return None


def _fail(e: Exception):
    if "free-models-per-day" in str(e):
        raise HTTPException(503, "The free language-model quota for today is used up. Try again tomorrow, "
                                 "or ask a question that has already been signed.")
    raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/ask")
def ask(body: Ask):  # sync: FastAPI runs it in a worker thread, so the page stays responsive
    q = body.question.strip()
    if not q or len(q) > 300:
        raise HTTPException(400, "Ask a question of 1 to 300 characters.")
    if body.lang not in LANGS:
        raise HTTPException(400, "lang must be one of " + ", ".join(LANGS))
    if not body.fresh and (r := _replay(OUT / question_id(q, body.lang) / "report.json", body.lang)):
        return r
    try:
        return pipeline(body.lang).run(q)
    except Exception as e:  # shown on the page rather than as a bare 500
        _fail(e)


class SignText(BaseModel):
    text: str
    lang: str = "en"
    fresh: bool = False


@app.post("/api/sign")
def sign(body: SignText):
    """Studio: the user's own text in sign language, without retrieval or answer generation."""
    text = body.text.strip()
    if not text or len(text) > 1000:
        raise HTTPException(400, "Write a text of 1 to 1,000 characters.")
    if body.lang not in LANGS:
        raise HTTPException(400, "lang must be one of " + ", ".join(LANGS))
    if not body.fresh and (r := _replay(OUT / question_id(text, body.lang, "text") / "report.json", body.lang)):
        return r
    try:
        return pipeline(body.lang).sign_text(text)
    except ValueError as e:
        if "nothing in the lexicon" in str(e):
            raise HTTPException(422, "None of these words has a recorded sign yet.")
        _fail(e)
    except Exception as e:
        _fail(e)
```

(The existing `ask` body is replaced by the version above; its behaviour is unchanged.)

- [ ] **Step 4: Run the tests**

Run: `py -m pytest tests/test_languages.py tests/test_app.py -v`
Expected: all PASS (including the earlier quota test from the previous session, which now goes through `_fail`).

- [ ] **Step 5: Live check** (needs `data/`): `PYTHONPATH=src py -c "from isharati.pipeline import Pipeline; r = Pipeline('en').sign_text('Prayer is the pillar of the religion.'); print(r['glosser'], r['glosses'])"`.
Expected: a gloss list such as `['PRAYER', ..., 'RELIGION']` and backend `llm`, or `rule` if the quota is used up.

- [ ] **Step 6: Commit**

```bash
git add src/isharati/glossing/llm.py src/isharati/pipeline.py src/isharati/app/server.py tests/
git commit -m "Text to sign: /api/sign signs the user's own text without answering; English falls back to rules" -m "<attribution lines>"
```

---

### Task 4: Lexicon figures (`stats.json`) and `/api/stats`

**Files:**
- Create: `scripts/eval/app_stats.py`, `src/isharati/app/static/stats.json` (generated)
- Modify: `src/isharati/app/server.py`
- Test: `tests/test_app.py` (append)

**Interfaces:**
- Consumes: `results/eda.json`; lexicon files; `scripts/hub/publish.py:LICENCES`; `wslp_enabled()` (Task 1); `avatars()` in `server.py`.
- Produces: `stats.json` with shape
  `{"languages": [{"code", "name", "sign_language", "signs", "glosses", "passages", "coverage", "signs_by_source"?}], "licences": [{"source", "licence", "use"}], "created"}`;
  `GET /api/stats` → the same plus `"avatars": int`, with Urdu `signs` reduced to `signs_by_source["cislr"]` when WSLP is off.

- [ ] **Step 1: Write the failing test** — append to `tests/test_app.py`:

```python
def test_stats_reflects_the_wslp_flag(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    stats = {"languages": [{"code": "en", "signs": 10}, {"code": "ur", "signs": 7, "signs_by_source": {"cislr": 3, "wslp": 4}}],
             "licences": [], "created": "x"}
    f = tmp_path / "stats.json"
    f.write_text(json.dumps(stats), encoding="utf-8")
    monkeypatch.setattr(server, "STATS", f)
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "1")
    on = c.get("/api/stats").json()
    assert {l["code"]: l["signs"] for l in on["languages"]} == {"en": 10, "ur": 7} and on["avatars"] >= 6
    monkeypatch.setenv("ISHARATI_URDU_WSLP", "0")
    off = c.get("/api/stats").json()
    assert {l["code"]: l["signs"] for l in off["languages"]}["ur"] == 3


def test_generated_stats_file_is_complete():
    from isharati.app import server
    s = json.loads(server.STATS.read_text(encoding="utf-8"))
    assert [l["code"] for l in s["languages"]] == ["en", "ar", "tr", "ur"]
    for l in s["languages"]:
        assert l["signs"] > 0 and l["passages"] > 0 and 0 < l["coverage"] < 1
    assert any("WSLP" in r["source"] for r in s["licences"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -m pytest tests/test_app.py -v -k stats`
Expected: FAIL (`AttributeError: module ... has no attribute 'STATS'`).

- [ ] **Step 3: Implement.** Create `scripts/eval/app_stats.py`:

```python
"""The figures the landing page shows, from the data: signs and distinct glosses per lexicon, approved passages, and
coverage (content-word token coverage of the Qur'an + hadith corpus, results/eda.json; Urdu through ISL's English
glosses). Writes src/isharati/app/static/stats.json, which the Space serves (it has no results/ folder).

  py scripts/eval/app_stats.py
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "hub")]
from isharati.config import DATA  # noqa: E402
from publish import LICENCES  # noqa: E402

OUT = ROOT / "src" / "isharati" / "app" / "static" / "stats.json"
LANGS = [  # code, language, sign language, lexicon files, eda.json key for coverage, eda.json key for passages
    ("en", "English", "ASL", [DATA / "asl" / "lexicon" / "lexicon_all.jsonl"], "en", "en"),
    ("ar", "Arabic", "ArSL", [DATA / "lexicon_v2" / "lexicon_qa.jsonl"], "ar", "ar"),
    ("tr", "Turkish", "TİD", [DATA / "lexicon_tr" / "dictionaries" / "entries_tid_dictionary.jsonl"], "tr", "tr"),
    ("ur", "Urdu", "ISL", [DATA / "lexicon_isl" / "cislr" / "entries.jsonl",
                           DATA / "lexicon_isl" / "wslp" / "entries.jsonl"], "isl_on_en", "ur"),
]


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    eda = json.loads((ROOT / "results" / "eda.json").read_text(encoding="utf-8"))
    langs = []
    for code, name, sl, files, cov_key, pas_key in LANGS:
        by_source = {f.parent.name: [r for r in rows(f) if not r.get("is_letter")] for f in files}
        signs = [r for rs in by_source.values() for r in rs]
        entry = {"code": code, "name": name, "sign_language": sl, "signs": len(signs),
                 "glosses": len({r["gloss"].rstrip("0123456789").strip().lower() for r in signs}),
                 "passages": eda[pas_key]["passages"],
                 "coverage": round(eda[cov_key]["all"]["content_token_coverage"], 4)}
        if code == "ur":
            entry["signs_by_source"] = {k: len(v) for k, v in by_source.items()}
        langs.append(entry)
    out = {"languages": langs, "licences": [{"source": s, "licence": l, "use": u} for s, l, u in LICENCES],
           "created": date.today().isoformat()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["languages"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
```

Run it: `py scripts/eval/app_stats.py`. Expected output: four languages; `en` signs 3,213 and coverage 0.6316, `ar`
signs 3,978 and 0.5326, `tr` signs 1,373 and 0.1593, `ur` signs 8,019 (cislr 3,964 + wslp 4,055) and 0.5109. If
`publish.py` fails to import (it imports `huggingface_hub` at module level and that is missing), install it with
`py -m pip install huggingface_hub`.

In `src/isharati/app/server.py`, add after `STATIC = ...`:

```python
STATS = STATIC / "stats.json"  # scripts/eval/app_stats.py
```

and add the endpoint after `avatars()`:

```python
@app.get("/api/stats")
def stats():
    """The landing page's figures: per-language signs, passages and coverage, the licences, and the avatar count."""
    from isharati.lexicon.isl import wslp_enabled
    s = json.loads(STATS.read_text(encoding="utf-8"))
    for lang in s["languages"]:
        if lang.get("signs_by_source") and not wslp_enabled():
            lang["signs"] = lang["signs_by_source"]["cislr"]
    s["avatars"] = len(avatars())
    return s
```

- [ ] **Step 4: Run the tests**

Run: `py -m pytest tests/test_app.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/app_stats.py src/isharati/app/static/stats.json src/isharati/app/server.py tests/test_app.py
git commit -m "Stats: generated lexicon and coverage figures, served at /api/stats" -m "<attribution lines>"
```

---

### Task 5: FastAPI serves the React build; old page at `/classic`

**Files:**
- Modify: `src/isharati/app/server.py`
- Test: `tests/test_app.py` (append; update `test_page_and_static_assets`)

**Interfaces:**
- Consumes: `isharati.config.ROOT`.
- Produces: `server.WEB_DIST: Path` (env `ISHARATI_WEB_DIST`, default `ROOT / "web" / "dist"`); routes `GET /` (React
  `index.html` when built, else the classic page), `GET /classic`, `GET /assets/{path}`, `GET /clips/{path}`,
  `GET /favicon.svg`, `GET /record`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_app.py`:

```python
def _dist(tmp_path):
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "clips").mkdir()
    (d / "index.html").write_text("<div id=root>react</div>", encoding="utf-8")
    (d / "record.html").write_text("recorder", encoding="utf-8")
    (d / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (d / "clips" / "clips.json").write_text("[]", encoding="utf-8")
    return d


def test_react_build_served_with_classic_fallback(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "WEB_DIST", _dist(tmp_path))
    assert "react" in c.get("/").text
    assert "Isharati" in c.get("/classic").text
    assert c.get("/assets/app.js").status_code == 200
    assert c.get("/clips/clips.json").json() == []
    assert c.get("/record").text == "recorder"
    assert c.get("/assets/..%2F..%2Findex.html").status_code == 404
    assert c.get("/assets/missing.js").status_code == 404


def test_classic_page_when_not_built(tmp_path, monkeypatch):
    server, c = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "WEB_DIST", tmp_path / "nothing")
    assert "Isharati" in c.get("/").text
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -m pytest tests/test_app.py -v -k "react or classic"`
Expected: FAIL (`AttributeError: ... 'WEB_DIST'`).

- [ ] **Step 3: Implement.** In `server.py` add `import os` to the top imports, `from isharati.config import ROOT` next to
the other `isharati` imports, and after `STATS = ...`:

```python
WEB_DIST = Path(os.environ.get("ISHARATI_WEB_DIST", ROOT / "web" / "dist"))  # the React landing page (web/, Vite)
```

Replace `@app.get("/") def page()` with:

```python
@app.get("/")
def page():
    index = WEB_DIST / "index.html"
    return FileResponse(index if index.is_file() else PAGE)


@app.get("/classic")
def classic():
    return FileResponse(PAGE)


def _from_dist(sub: str, name: str):
    base = (WEB_DIST / sub).resolve()
    path = (base / name).resolve()
    if not path.is_file() or base not in path.parents:  # no escaping the build folder
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/assets/{name:path}")
def assets(name: str):
    return _from_dist("assets", name)


@app.get("/clips/{name:path}")
def clips(name: str):
    return _from_dist("clips", name)


@app.get("/favicon.svg")
def favicon():
    return _from_dist(".", "favicon.svg")


@app.get("/record")
def record():
    """The headless clip recorder (scripts/avatars/record_clips.py)."""
    return _from_dist(".", "record.html")
```

Update the module docstring's first line to mention the landing page:
`"""The Isharati web application: the React landing page (web/, built to web/dist) with the translator, and the API it
calls. The previous single-file page stays at /classic.`

- [ ] **Step 4: Run the tests**

Run: `py -m pytest tests/test_app.py -v`
Expected: all PASS (`test_page_and_static_assets` still passes because `web/dist` does not exist yet, so `/` falls back).

- [ ] **Step 5: Commit**

```bash
git add src/isharati/app/server.py tests/test_app.py
git commit -m "Server: serve the React build at /, the old page at /classic" -m "<attribution lines>"
```

---

### Task 6: `web/` scaffold, theme and four-language interface

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/index.html`, `web/record.html`,
  `web/public/favicon.svg`, `web/src/main.tsx`, `web/src/App.tsx` (placeholder sections replaced in Task 11),
  `web/src/styles.css`, `web/src/test-setup.ts`, `web/src/static.d.ts`, `web/src/types.ts`,
  `web/src/i18n/i18n.tsx`, `web/src/i18n/{en,ar,tr,ur}.json`, `web/src/i18n/i18n.test.tsx`
- Modify: `.gitignore` (add `web/node_modules/`, `web/dist/`)

**Interfaces:**
- Consumes: `/api/*` (proxied to `http://127.0.0.1:8765` in dev), `@static/*` → `src/isharati/app/static/*`.
- Produces: `I18nProvider`, `useI18n(): { ui: Lang; setUi(l: Lang): void; t(key: string): string; dir: "ltr" | "rtl" }`,
  `RTL: Set<Lang>`, `LANG_LIST: Lang[]`; types `Lang`, `Segment`, `Source`, `Report`, `PoseFrames`, `LangStats`,
  `Stats`, `Clip` (in `src/types.ts`); module typings for `@static/avatar.js`, `@static/outfits.js`, `@static/stats.json`.

- [ ] **Step 1: Create the package.** `web/package.json`:

```json
{
  "name": "isharati-web",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "test": "vitest run",
    "preview": "vite preview"
  },
  "dependencies": {
    "@pixiv/three-vrm": "3.1.0",
    "framer-motion": "^11.11.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "three": "0.169.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.2",
    "@testing-library/react": "^16.0.1",
    "@types/node": "^22.7.5",
    "@types/react": "^18.3.11",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.2",
    "jsdom": "^25.0.1",
    "typescript": "^5.6.3",
    "vite": "^5.4.9",
    "vitest": "^2.1.3"
  }
}
```

Run: `cd web && npm install` (creates `package-lock.json`; commit it).

`web/vite.config.ts` — the shared avatar module lives outside `web/`, so its bare imports (`three`, `three/addons/...`,
`@pixiv/three-vrm`) are pointed at `web/node_modules` explicitly:

```ts
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const nm = (p: string) => resolve(__dirname, "node_modules", p);
const api = "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: "@static", replacement: resolve(__dirname, "../src/isharati/app/static") },
      { find: /^three\/addons\/(.*)$/, replacement: nm("three/examples/jsm/$1") },
      { find: /^three$/, replacement: nm("three/build/three.module.js") },
      { find: /^@pixiv\/three-vrm$/, replacement: nm("@pixiv/three-vrm/lib/three-vrm.module.js") },
    ],
  },
  server: {
    fs: { allow: [".."] },
    proxy: { "/api": api, "/pose": api, "/video": api, "/static": api },
  },
  build: {
    rollupOptions: { input: { main: resolve(__dirname, "index.html"), record: resolve(__dirname, "record.html") } },
  },
  test: { environment: "jsdom", setupFiles: ["./src/test-setup.ts"], globals: true },
});
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "strict": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "skipLibCheck": true,
    "noEmit": true,
    "types": ["node", "vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src", "vite.config.ts"]
}
```

`web/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Isharati · إشارتي</title>
  <meta name="description" content="Questions about Islam answered from the Qur'an and hadith, and signed in ASL, ArSL, TİD and ISL.">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Kufi+Arabic:wght@400;700&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap" rel="stylesheet">
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
```

`web/record.html`:

```html
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Isharati clip recorder</title>
<style>html,body{margin:0;background:#0b1d33}canvas{display:block;width:480px;height:600px}</style></head>
<body><canvas id="c"></canvas><script type="module" src="/src/record.ts"></script></body>
</html>
```

`web/public/favicon.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#081526"/>
<g stroke="#7ef0d3" stroke-width="4" stroke-linecap="round" fill="none"><circle cx="32" cy="16" r="7"/>
<path d="M32 23v20M18 30h28M18 30l-4 18M46 30l8-14"/></g></svg>
```

Add a temporary `web/src/record.ts` containing `export {};` (Task 12 replaces it) so the build input exists.

`.gitignore`: append

```
web/node_modules/
web/dist/
```

- [ ] **Step 2: Types and module typings.** `web/src/types.ts`:

```ts
export type Lang = "en" | "ar" | "tr" | "ur";

export interface Segment { label: string; kind: "sign" | "missing" | "fingerspell"; start_s: number; end_s: number }
export interface Source { reference: string; url: string }
export interface Report {
  status: "signed" | "referred" | "unanswered";
  id?: string; question?: string; answer?: string; reason?: string;
  sources?: Source[]; glosses?: string[]; missing_signs?: string[]; segments?: Segment[];
  coverage?: number; llm?: string | null; cached?: boolean; dropped_unsupported?: string[];
  mode?: "answer" | "text"; glosser?: string;
}
export interface PoseFrames { fps: number; frames: number[][][] }
export interface LangStats {
  code: Lang; name: string; sign_language: string; signs: number; glosses: number; passages: number;
  coverage: number; signs_by_source?: Record<string, number>;
}
export interface Licence { source: string; licence: string; use: string }
export interface Stats { languages: LangStats[]; licences: Licence[]; created?: string; avatars?: number }
export interface Clip {
  avatar: string; lang: Lang; sign_language: string; glosses: string[];
  webm: string; mp4: string; poster: string; seconds: number;
}
```

`web/src/static.d.ts` (ambient declarations cannot import relative modules, so the shapes are written inline;
`state.tsx` treats the JSON as `Stats`):

```ts
// The avatar module shared with the classic page (src/isharati/app/static), imported through the @static alias.
declare module "@static/avatar.js" {
  export class SignAvatar {
    constructor(canvas: HTMLCanvasElement);
    load(url: string): Promise<SignAvatar>;
    resize(): void;
    setFrames(frames: { fps: number; frames: number[][][] } | null): void;
    render(t: number, dt?: number): void;
    renderer: { dispose(): void; setClearColor(color: number, alpha: number): void };
    vrm: unknown;
  }
}
declare module "@static/outfits.js" {
  export const PRESETS: Record<"original" | "hijab" | "shemagh", Record<string, unknown>>;
  export function applyOutfit(avatar: unknown, outfit: Record<string, unknown>): void;
}
declare module "@static/stats.json" {
  const stats: any;  // generated by scripts/eval/app_stats.py; typed as Stats where it is used
  export default stats;
}
```

- [ ] **Step 3: Write the failing i18n test.** `web/src/test-setup.ts`:

```ts
import "@testing-library/jest-dom/vitest";

class NoopObserver { observe() {} unobserve() {} disconnect() {} takeRecords() { return []; } }
// jsdom has neither; framer-motion's useInView and the canvases use them
(globalThis as any).IntersectionObserver ??= NoopObserver;
(globalThis as any).ResizeObserver ??= NoopObserver;
```

`web/src/i18n/i18n.test.tsx`:

```tsx
import { act, render, screen } from "@testing-library/react";
import ar from "./ar.json";
import en from "./en.json";
import tr from "./tr.json";
import ur from "./ur.json";
import { I18nProvider, useI18n } from "./i18n";

function Probe() {
  const { t, setUi, dir } = useI18n();
  return <div><p>{t("hero.cta")}</p><p data-testid="dir">{dir}</p><p>{t("no.such.key")}</p>
    <button onClick={() => setUi("ar")}>ar</button><button onClick={() => setUi("tr")}>tr</button></div>;
}

test("every language has every English key", () => {
  for (const [name, strings] of Object.entries({ ar, tr, ur })) {
    const missing = Object.keys(en).filter((k) => !(k in strings));
    expect(missing, name).toEqual([]);
  }
});

test("switching to Arabic sets right-to-left on the document", () => {
  render(<I18nProvider initial="en"><Probe /></I18nProvider>);
  expect(screen.getByText(en["hero.cta"])).toBeInTheDocument();
  act(() => screen.getByText("ar").click());
  expect(screen.getByText(ar["hero.cta"])).toBeInTheDocument();
  expect(document.documentElement.dir).toBe("rtl");
  expect(document.documentElement.lang).toBe("ar");
  act(() => screen.getByText("tr").click());
  expect(document.documentElement.dir).toBe("ltr");
  expect(screen.getByText("no.such.key")).toBeInTheDocument();   // unknown keys show the key, not a crash
});
```

Run: `cd web && npx vitest run src/i18n` — Expected: FAIL (`Cannot find module './i18n'`).

- [ ] **Step 4: Implement i18n.** `web/src/i18n/i18n.tsx`:

```tsx
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import type { Lang } from "../types";
import ar from "./ar.json";
import en from "./en.json";
import tr from "./tr.json";
import ur from "./ur.json";

const STRINGS: Record<Lang, Record<string, string>> = { en, ar, tr, ur };
export const LANG_LIST: Lang[] = ["en", "ar", "tr", "ur"];
export const RTL = new Set<Lang>(["ar", "ur"]);
const KEY = "isharati.ui";

interface I18n { ui: Lang; setUi: (l: Lang) => void; t: (key: string) => string; dir: "ltr" | "rtl" }
const Ctx = createContext<I18n | null>(null);

function detect(): Lang {
  const isLang = (x: unknown): x is Lang => LANG_LIST.includes(x as Lang);
  const param = new URLSearchParams(location.search).get("ui");
  if (isLang(param)) return param;
  try { const saved = localStorage.getItem(KEY); if (isLang(saved)) return saved; } catch { /* private mode */ }
  const nav = (navigator.language || "en").slice(0, 2);
  return isLang(nav) ? nav : "en";
}

export function I18nProvider({ children, initial }: { children: ReactNode; initial?: Lang }) {
  const [ui, setUi] = useState<Lang>(initial ?? detect());
  const dir = RTL.has(ui) ? "rtl" : "ltr";
  useEffect(() => {
    document.documentElement.lang = ui;
    document.documentElement.dir = dir;
    try { localStorage.setItem(KEY, ui); } catch { /* private mode */ }
  }, [ui, dir]);
  const t = useCallback((key: string) => STRINGS[ui][key] ?? STRINGS.en[key] ?? key, [ui]);
  return <Ctx.Provider value={{ ui, setUi, t, dir }}>{children}</Ctx.Provider>;
}

export function useI18n(): I18n {
  const v = useContext(Ctx);
  if (!v) throw new Error("useI18n outside I18nProvider");
  return v;
}
```

`web/src/i18n/en.json`:

```json
{
  "nav.try": "Try it", "nav.how": "How it works", "nav.lexicon": "Lexicon", "nav.avatars": "Avatars", "nav.sources": "Sources",
  "hero.kicker": "AI in service of Islamic knowledge",
  "hero.title": "Islamic knowledge, in sign language",
  "hero.lead": "Ask a question about Islam. Isharati answers from the Qur'an and authenticated hadith, cites its sources, and signs the answer with a 3D avatar.",
  "hero.cta": "Try it now",
  "hero.languages": "languages", "hero.signs": "recorded signs", "hero.passages": "approved passages",
  "try.title": "Ask, and watch it signed",
  "try.lead": "Pick a language, ask a question, and Isharati answers only from approved sources.",
  "try.placeholder": "Ask a question about Islam…", "try.go": "Sign it",
  "try.wait": "Finding the answer in the sources and assembling the signs… (up to a minute)",
  "try.answer": "Answer from approved sources", "try.gloss": "Sign glosses", "try.video": "Signed video",
  "try.avatar": "Avatar", "try.keypoints": "Keypoints", "try.play": "Play", "try.pause": "Pause",
  "try.export": "Export video", "try.exporting": "Recording…", "try.speed": "Speed",
  "try.coverage": "Coverage", "try.missingNote": "red: no sign recorded yet (held in the video)",
  "try.referred": "Questions about rulings are referred to a qualified scholar.",
  "try.unanswered": "No approved source answers this question: ",
  "try.quota": "The free language-model quota for today is used up. Try one of the example questions, which are already signed.",
  "try.html": "The server sent a web page instead of an answer. Reload the page, or open the Space in its own tab.",
  "try.error": "Something went wrong: ", "try.cached": "replayed from an earlier run",
  "try.avatarFail": "The avatar could not load here, so the keypoints are shown.", "try.pick": "Choose an avatar",
  "how.title": "How it works",
  "how.1.t": "Your question", "how.1.b": "In English, Arabic, Turkish or Urdu. Questions about rulings go to a scholar, never to the model.",
  "how.2.t": "Approved sources", "how.2.b": "Hybrid search over the Qur'an and graded hadith from HadeethEnc, in your language.",
  "how.3.t": "A cited answer", "how.3.b": "Two to four short sentences, each tied to the passages it comes from; unsupported sentences are dropped.",
  "how.4.t": "Sign glosses", "how.4.b": "The answer becomes a sequence of signs, using only signs that exist in that sign language's lexicon.",
  "how.5.t": "Signed pose and avatar", "how.5.b": "Recorded sign poses are stitched together and drive a 3D avatar or a keypoint skeleton.",
  "lex.title": "The lexicon", "lex.lead": "Real recorded signs for each sign language, measured against the whole Qur'an and hadith corpus.",
  "lex.signs": "signs", "lex.passages": "passages", "lex.coverage": "of content words covered",
  "lex.note": "Coverage counts how many content words in the Qur'an and hadith have a recorded sign. Words without one are shown in red and held in the video.",
  "av.title": "Choose who signs", "av.lead": "Five realistic avatars and one stylised one, all driven by the same recorded signs.",
  "av.use": "Use this avatar",
  "avatar.stylised": "Stylised", "avatar.female_06": "Woman, white hijab", "avatar.female_10": "Woman, black abaya",
  "avatar.male_15": "Man, thobe and cap", "avatar.male_19": "Man, red shemagh", "avatar.male_21": "Man, white shemagh",
  "src.title": "Sources and licences", "src.lead": "Every dataset Isharati draws on, with its licence.",
  "src.source": "Source", "src.licence": "Licence", "src.use": "Used for",
  "team.title": "Team", "team.fatimah": "Fatimah Emad Eldin — AI research and engineering",
  "team.asmaa": "Dr. Asmaa Al-Mirghani — Islamic content",
  "foot.note": "Isharati · إشارتي — answers come only from approved sources; it does not issue rulings."
}
```

`web/src/i18n/ar.json`:

```json
{
  "nav.try": "جرّبها", "nav.how": "كيف تعمل", "nav.lexicon": "المعجم", "nav.avatars": "الشخصيات", "nav.sources": "المصادر",
  "hero.kicker": "ذكاء اصطناعي في خدمة المعرفة الإسلامية",
  "hero.title": "المعرفة الإسلامية بلغة الإشارة",
  "hero.lead": "اطرح سؤالًا عن الإسلام، فتجيبك إشارتي من القرآن الكريم والأحاديث الصحيحة مع ذكر المصادر، ثم تترجم الإجابة إلى لغة الإشارة بشخصية ثلاثية الأبعاد.",
  "hero.cta": "جرّبها الآن",
  "hero.languages": "لغات", "hero.signs": "إشارة مسجّلة", "hero.passages": "نصًّا معتمدًا",
  "try.title": "اسأل وشاهد الإجابة بالإشارة",
  "try.lead": "اختر اللغة واطرح سؤالك، وتجيب إشارتي من المصادر المعتمدة فقط.",
  "try.placeholder": "اطرح سؤالًا عن الإسلام…", "try.go": "ترجم بالإشارة",
  "try.wait": "نبحث عن الإجابة في المصادر ونركّب الإشارات… (قد يستغرق دقيقة)",
  "try.answer": "الإجابة من المصادر المعتمدة", "try.gloss": "مسرد الإشارات", "try.video": "فيديو الإشارة",
  "try.avatar": "الشخصية", "try.keypoints": "النقاط المفصلية", "try.play": "تشغيل", "try.pause": "إيقاف",
  "try.export": "تصدير الفيديو", "try.exporting": "جارٍ التسجيل…", "try.speed": "السرعة",
  "try.coverage": "التغطية", "try.missingNote": "الأحمر: لا توجد إشارة مسجّلة بعد (ثبات في الفيديو)",
  "try.referred": "أسئلة الأحكام الشرعية تُحال إلى عالم مختص.",
  "try.unanswered": "لا يوجد مصدر معتمد يجيب عن هذا السؤال: ",
  "try.quota": "نفدت حصة النموذج المجانية لهذا اليوم. جرّب أحد الأسئلة المقترحة، فهي مترجمة مسبقًا.",
  "try.html": "أرسل الخادم صفحة ويب بدلًا من الإجابة. أعد تحميل الصفحة أو افتح التطبيق في نافذة مستقلة.",
  "try.error": "حدث خطأ: ", "try.cached": "معاد من تشغيل سابق",
  "try.avatarFail": "تعذّر تحميل الشخصية هنا، لذا تُعرض النقاط المفصلية.", "try.pick": "اختر الشخصية",
  "how.title": "كيف تعمل",
  "how.1.t": "سؤالك", "how.1.b": "بالعربية أو الإنجليزية أو التركية أو الأردية. أسئلة الأحكام تُحال إلى عالم، لا إلى النموذج.",
  "how.2.t": "مصادر معتمدة", "how.2.b": "بحث هجين في القرآن الكريم والأحاديث المصنّفة من موسوعة الأحاديث، بلغتك.",
  "how.3.t": "إجابة موثّقة", "how.3.b": "من جملتين إلى أربع جمل قصيرة، كل جملة مربوطة بنصوصها؛ وتُحذف الجمل غير المدعومة.",
  "how.4.t": "مسرد الإشارات", "how.4.b": "تتحول الإجابة إلى تسلسل إشارات، من الإشارات الموجودة في معجم لغة الإشارة فقط.",
  "how.5.t": "وضعيات وشخصية", "how.5.b": "تُدمج وضعيات الإشارات المسجّلة لتحريك شخصية ثلاثية الأبعاد أو هيكل من النقاط.",
  "lex.title": "المعجم", "lex.lead": "إشارات حقيقية مسجّلة لكل لغة إشارة، مقيسة على نصوص القرآن والحديث كاملة.",
  "lex.signs": "إشارة", "lex.passages": "نصًّا", "lex.coverage": "من الكلمات الدلالية مغطّاة",
  "lex.note": "التغطية هي نسبة الكلمات الدلالية في القرآن والحديث التي لها إشارة مسجّلة. الكلمات بلا إشارة تظهر بالأحمر وتثبت في الفيديو.",
  "av.title": "اختر من يترجم", "av.lead": "خمس شخصيات واقعية وشخصية كرتونية، تحرّكها الإشارات المسجّلة نفسها.",
  "av.use": "استخدم هذه الشخصية",
  "avatar.stylised": "كرتونية", "avatar.female_06": "امرأة بحجاب أبيض", "avatar.female_10": "امرأة بعباءة سوداء",
  "avatar.male_15": "رجل بثوب وطاقية", "avatar.male_19": "رجل بشماغ أحمر", "avatar.male_21": "رجل بشماغ أبيض",
  "src.title": "المصادر والتراخيص", "src.lead": "كل مجموعة بيانات تعتمد عليها إشارتي، مع ترخيصها.",
  "src.source": "المصدر", "src.licence": "الترخيص", "src.use": "الاستخدام",
  "team.title": "الفريق", "team.fatimah": "فاطمة عماد الدين — أبحاث وهندسة الذكاء الاصطناعي",
  "team.asmaa": "د. أسماء الميرغني — المحتوى الإسلامي",
  "foot.note": "إشارتي — الإجابات من المصادر المعتمدة فقط، ولا تُصدر فتاوى."
}
```

`web/src/i18n/tr.json`:

```json
{
  "nav.try": "Dene", "nav.how": "Nasıl çalışır", "nav.lexicon": "Sözlük", "nav.avatars": "Avatarlar", "nav.sources": "Kaynaklar",
  "hero.kicker": "İslami bilginin hizmetinde yapay zekâ",
  "hero.title": "İşaret dilinde İslami bilgi",
  "hero.lead": "İslam hakkında bir soru sorun. Isharati, Kur'an ve sahih hadislerden kaynak göstererek cevap verir ve cevabı 3B bir avatarla işaret diline çevirir.",
  "hero.cta": "Hemen dene",
  "hero.languages": "dil", "hero.signs": "kayıtlı işaret", "hero.passages": "onaylı metin",
  "try.title": "Sorun, işaretle izleyin",
  "try.lead": "Bir dil seçin ve sorunuzu sorun; Isharati yalnızca onaylı kaynaklardan cevap verir.",
  "try.placeholder": "İslam hakkında bir soru sorun…", "try.go": "İşaretle",
  "try.wait": "Cevap kaynaklarda aranıyor ve işaretler birleştiriliyor… (bir dakikayı bulabilir)",
  "try.answer": "Onaylı kaynaklardan cevap", "try.gloss": "İşaret glosları", "try.video": "İşaret videosu",
  "try.avatar": "Avatar", "try.keypoints": "Eklem noktaları", "try.play": "Oynat", "try.pause": "Duraklat",
  "try.export": "Videoyu indir", "try.exporting": "Kaydediliyor…", "try.speed": "Hız",
  "try.coverage": "Kapsam", "try.missingNote": "kırmızı: henüz kayıtlı işaret yok (videoda beklenir)",
  "try.referred": "Hüküm soruları yetkin bir âlime yönlendirilir.",
  "try.unanswered": "Bu soruyu cevaplayan onaylı bir kaynak yok: ",
  "try.quota": "Bugünkü ücretsiz model kotası doldu. Önceden işaretlenmiş örnek sorulardan birini deneyin.",
  "try.html": "Sunucu cevap yerine bir web sayfası gönderdi. Sayfayı yenileyin veya uygulamayı ayrı bir sekmede açın.",
  "try.error": "Bir hata oluştu: ", "try.cached": "önceki bir çalıştırmadan",
  "try.avatarFail": "Avatar burada yüklenemedi; eklem noktaları gösteriliyor.", "try.pick": "Avatar seçin",
  "how.title": "Nasıl çalışır",
  "how.1.t": "Sorunuz", "how.1.b": "Türkçe, Arapça, İngilizce veya Urduca. Hüküm soruları modele değil, bir âlime yönlendirilir.",
  "how.2.t": "Onaylı kaynaklar", "how.2.b": "Kendi dilinizde Kur'an ve HadeethEnc'in derecelendirilmiş hadisleri üzerinde karma arama.",
  "how.3.t": "Kaynaklı cevap", "how.3.b": "İki ila dört kısa cümle, her biri dayandığı metinlere bağlı; desteklenmeyen cümleler çıkarılır.",
  "how.4.t": "İşaret glosları", "how.4.b": "Cevap, yalnızca o işaret dilinin sözlüğünde bulunan işaretlerden oluşan bir diziye dönüşür.",
  "how.5.t": "İşaretli poz ve avatar", "how.5.b": "Kayıtlı işaret pozları birleştirilir ve 3B bir avatarı veya eklem iskeletini hareket ettirir.",
  "lex.title": "Sözlük", "lex.lead": "Her işaret dili için gerçek kayıtlı işaretler, tüm Kur'an ve hadis metinleriyle ölçüldü.",
  "lex.signs": "işaret", "lex.passages": "metin", "lex.coverage": "içerik kelimesi kapsanıyor",
  "lex.note": "Kapsam, Kur'an ve hadislerdeki içerik kelimelerinin kaçının kayıtlı bir işareti olduğunu gösterir. İşareti olmayan kelimeler kırmızıyla gösterilir ve videoda beklenir.",
  "av.title": "Kim işaretlesin", "av.lead": "Beş gerçekçi ve bir stilize avatar; hepsi aynı kayıtlı işaretlerle hareket eder.",
  "av.use": "Bu avatarı kullan",
  "avatar.stylised": "Stilize", "avatar.female_06": "Kadın, beyaz başörtülü", "avatar.female_10": "Kadın, siyah abaya",
  "avatar.male_15": "Erkek, cüppe ve takke", "avatar.male_19": "Erkek, kırmızı puşi", "avatar.male_21": "Erkek, beyaz puşi",
  "src.title": "Kaynaklar ve lisanslar", "src.lead": "Isharati'nin kullandığı her veri seti ve lisansı.",
  "src.source": "Kaynak", "src.licence": "Lisans", "src.use": "Kullanım",
  "team.title": "Ekip", "team.fatimah": "Fatimah Emad Eldin — yapay zekâ araştırması ve mühendisliği",
  "team.asmaa": "Dr. Asmaa Al-Mirghani — İslami içerik",
  "foot.note": "Isharati · إشارتي — cevaplar yalnızca onaylı kaynaklardan gelir; fetva vermez."
}
```

`web/src/i18n/ur.json`:

```json
{
  "nav.try": "آزمائیں", "nav.how": "یہ کیسے کام کرتا ہے", "nav.lexicon": "لغت", "nav.avatars": "اوتار", "nav.sources": "ماخذ",
  "hero.kicker": "اسلامی علم کی خدمت میں مصنوعی ذہانت",
  "hero.title": "اشاروں کی زبان میں اسلامی علم",
  "hero.lead": "اسلام کے بارے میں سوال پوچھیں۔ اشارتی قرآن اور صحیح احادیث سے حوالوں کے ساتھ جواب دیتا ہے، اور جواب کو تین جہتی اوتار کے ذریعے اشاروں کی زبان میں پیش کرتا ہے۔",
  "hero.cta": "ابھی آزمائیں",
  "hero.languages": "زبانیں", "hero.signs": "ریکارڈ شدہ اشارے", "hero.passages": "مستند متون",
  "try.title": "پوچھیں اور اشاروں میں دیکھیں",
  "try.lead": "زبان منتخب کریں اور سوال پوچھیں؛ اشارتی صرف مستند ماخذ سے جواب دیتا ہے۔",
  "try.placeholder": "اسلام کے بارے میں سوال پوچھیں…", "try.go": "اشارے میں دکھائیں",
  "try.wait": "ماخذ میں جواب تلاش کیا جا رہا ہے اور اشارے جوڑے جا رہے ہیں… (ایک منٹ تک)",
  "try.answer": "مستند ماخذ سے جواب", "try.gloss": "اشاروں کی فہرست", "try.video": "اشاروں کی ویڈیو",
  "try.avatar": "اوتار", "try.keypoints": "جوڑوں کے نقاط", "try.play": "چلائیں", "try.pause": "روکیں",
  "try.export": "ویڈیو محفوظ کریں", "try.exporting": "ریکارڈنگ…", "try.speed": "رفتار",
  "try.coverage": "احاطہ", "try.missingNote": "سرخ: ابھی کوئی ریکارڈ شدہ اشارہ نہیں (ویڈیو میں رکا رہتا ہے)",
  "try.referred": "شرعی احکام کے سوالات کسی مستند عالم کے سپرد کیے جاتے ہیں۔",
  "try.unanswered": "کوئی مستند ماخذ اس سوال کا جواب نہیں دیتا: ",
  "try.quota": "آج کا مفت ماڈل کوٹہ ختم ہو گیا ہے۔ مثال کے سوالات آزمائیں، وہ پہلے سے اشاروں میں موجود ہیں۔",
  "try.html": "سرور نے جواب کے بجائے ایک ویب صفحہ بھیجا۔ صفحہ دوبارہ لوڈ کریں یا ایپ کو الگ ٹیب میں کھولیں۔",
  "try.error": "کچھ غلط ہو گیا: ", "try.cached": "پچھلے رن سے دہرایا گیا",
  "try.avatarFail": "اوتار یہاں لوڈ نہیں ہو سکا، اس لیے نقاط دکھائے جا رہے ہیں۔", "try.pick": "اوتار منتخب کریں",
  "how.title": "یہ کیسے کام کرتا ہے",
  "how.1.t": "آپ کا سوال", "how.1.b": "اردو، عربی، انگریزی یا ترکی میں۔ احکام کے سوالات ماڈل کے بجائے عالم کے پاس جاتے ہیں۔",
  "how.2.t": "مستند ماخذ", "how.2.b": "آپ کی زبان میں قرآن اور ہدیث انسائیکلوپیڈیا کی درجہ بند احادیث پر مخلوط تلاش۔",
  "how.3.t": "حوالہ شدہ جواب", "how.3.b": "دو سے چار مختصر جملے، ہر ایک اپنے متون سے جڑا ہوا؛ غیر تائید شدہ جملے نکال دیے جاتے ہیں۔",
  "how.4.t": "اشاروں کی فہرست", "how.4.b": "جواب اشاروں کی ترتیب میں بدلتا ہے، صرف ان اشاروں سے جو اس اشاراتی زبان کی لغت میں موجود ہیں۔",
  "how.5.t": "اشاراتی پوز اور اوتار", "how.5.b": "ریکارڈ شدہ اشاروں کے پوز جوڑ کر تین جہتی اوتار یا نقاط کے ڈھانچے کو حرکت دی جاتی ہے۔",
  "lex.title": "لغت", "lex.lead": "ہر اشاراتی زبان کے حقیقی ریکارڈ شدہ اشارے، پورے قرآن اور حدیث کے متن پر ناپے گئے۔",
  "lex.signs": "اشارے", "lex.passages": "متون", "lex.coverage": "مفہومی الفاظ کا احاطہ",
  "lex.note": "احاطہ بتاتا ہے کہ قرآن اور حدیث کے کتنے مفہومی الفاظ کا ریکارڈ شدہ اشارہ موجود ہے۔ بغیر اشارے والے الفاظ سرخ دکھائے جاتے ہیں اور ویڈیو میں رکے رہتے ہیں۔",
  "av.title": "کون اشارہ کرے", "av.lead": "پانچ حقیقت پسند اور ایک کارٹون اوتار، سب ایک ہی ریکارڈ شدہ اشاروں سے چلتے ہیں۔",
  "av.use": "یہ اوتار استعمال کریں",
  "avatar.stylised": "کارٹون", "avatar.female_06": "خاتون، سفید حجاب", "avatar.female_10": "خاتون، سیاہ عبایا",
  "avatar.male_15": "مرد، ثوب اور ٹوپی", "avatar.male_19": "مرد، سرخ رومال", "avatar.male_21": "مرد، سفید رومال",
  "src.title": "ماخذ اور لائسنس", "src.lead": "ہر ڈیٹا سیٹ جس پر اشارتی انحصار کرتا ہے، اس کے لائسنس کے ساتھ۔",
  "src.source": "ماخذ", "src.licence": "لائسنس", "src.use": "استعمال",
  "team.title": "ٹیم", "team.fatimah": "فاطمہ عماد الدین — مصنوعی ذہانت کی تحقیق اور انجینئرنگ",
  "team.asmaa": "ڈاکٹر اسماء المیرغنی — اسلامی مواد",
  "foot.note": "اشارتی · إشارتي — جوابات صرف مستند ماخذ سے؛ یہ فتویٰ جاری نہیں کرتا۔"
}
```

- [ ] **Step 5: Theme, entry and a placeholder App.** `web/src/styles.css`:

```css
:root {
  --bg: #050b16; --bg2: #081526; --panel: #0b1d33; --panel2: #0f2742; --line: #1f4870;
  --ink: #e8f1ff; --muted: #9db3cf; --glow: #7ef0d3; --glow2: #8fb8ff; --accent: #2dd4bf; --accent2: #60a5fa;
  --missing: #ff8a80; --missing-bg: #3a1c22; --now: #ffe08a;
  --radius: 16px; --font: "Inter", system-ui, sans-serif;
}
:lang(ar) { --font: "Noto Kufi Arabic", "Inter", sans-serif; }
:lang(ur) { --font: "Noto Nastaliq Urdu", "Inter", sans-serif; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; background: var(--bg); }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font); line-height: 1.6; }
a { color: var(--glow); }
.wrap { width: min(1180px, 100% - 32px); margin-inline: auto; }
section { padding-block: 96px; }
section h2 { font-size: clamp(28px, 4vw, 44px); margin: 0 0 12px; }
.lead { color: var(--muted); max-width: 720px; margin: 0 0 32px; }
.grad { background: linear-gradient(90deg, var(--glow), var(--glow2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.btn { display: inline-flex; align-items: center; gap: 8px; border: 0; border-radius: 999px; padding: 12px 22px;
  font: 600 16px var(--font); cursor: pointer; color: #04121f; background: linear-gradient(90deg, var(--accent), var(--accent2));
  box-shadow: 0 0 24px #2dd4bf55; text-decoration: none; }
.btn.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); box-shadow: none; }
.btn:disabled { opacity: .55; cursor: wait; }
.panel { background: linear-gradient(180deg, var(--panel), var(--bg2)); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px; }
.label { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }
/* nav */
.nav { position: sticky; top: 0; z-index: 10; backdrop-filter: blur(12px); background: #050b16cc; border-bottom: 1px solid #1f487055; }
.nav .wrap { display: flex; align-items: center; gap: 20px; height: 64px; }
.nav .logo { font-weight: 800; font-size: 20px; text-decoration: none; color: var(--ink); }
.nav .links { display: flex; gap: 18px; margin-inline-start: auto; }
.nav .links a { color: var(--muted); text-decoration: none; font-size: 14px; }
.nav .links a:hover { color: var(--ink); }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
.seg button { background: transparent; color: var(--muted); border: 0; padding: 6px 12px; font: 600 13px var(--font); cursor: pointer; }
.seg button.on { background: var(--panel2); color: var(--glow); }
/* hero */
.hero { position: relative; overflow: hidden; padding-block: 72px 96px;
  background: radial-gradient(ellipse at 70% 30%, #12385a 0%, #081526 55%, var(--bg) 100%); }
.hero .wrap { display: grid; grid-template-columns: 1.1fr 1fr; gap: 40px; align-items: center; }
.hero h1 { font-size: clamp(36px, 6vw, 68px); line-height: 1.05; margin: 8px 0 18px; }
.hero .kicker { color: var(--glow); font-weight: 600; letter-spacing: .04em; }
.hero .stats { display: flex; gap: 36px; margin-top: 40px; flex-wrap: wrap; }
.hero .stats b { display: block; font-size: 36px; color: var(--glow); }
.hero .stage { position: relative; aspect-ratio: 4 / 5; }
.hero .stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.reel { display: flex; gap: 10px; margin-top: 14px; }
.reel video { width: 33%; border-radius: 12px; border: 1px solid var(--line); background: var(--panel); }
/* translator */
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.tabs button { background: var(--panel); color: var(--muted); border: 1px solid var(--line); border-radius: 999px; padding: 8px 16px; font: 600 14px var(--font); cursor: pointer; }
.tabs button.on { color: #04121f; background: var(--glow); border-color: var(--glow); }
.ask { display: flex; gap: 10px; }
.ask input { flex: 1; min-width: 0; background: var(--bg2); color: var(--ink); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; font: 17px var(--font); }
.ask input:focus { outline: 2px solid var(--accent); }
.examples { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 24px; }
.examples button { background: transparent; color: var(--glow); border: 1px solid var(--line); border-radius: 999px; padding: 6px 12px; font: 14px var(--font); cursor: pointer; }
.result { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.answer { font-size: 19px; }
.sources a { display: block; font-size: 14px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font: 600 13px ui-monospace, monospace; padding: 5px 9px; border-radius: 8px; border: 1px solid var(--line); background: var(--bg2); cursor: pointer; color: var(--ink); }
.chip.missing { color: var(--missing); background: var(--missing-bg); border-color: #5a2a33; }
.chip.now { background: var(--now); color: #1b1300; border-color: var(--now); box-shadow: 0 0 12px #ffe08a88; }
.status { color: var(--muted); margin: 8px 0 16px; }
.status.err { color: var(--missing); }
.viewer { position: relative; aspect-ratio: 4 / 5; border-radius: 12px; overflow: hidden; background: radial-gradient(circle at 50% 35%, #12385a, var(--bg2)); }
.viewer canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.transport { display: flex; gap: 10px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
.transport input[type=range] { flex: 1; accent-color: var(--accent); }
.transport select { background: var(--bg2); color: var(--ink); border: 1px solid var(--line); border-radius: 8px; padding: 6px; }
.pickers { display: flex; gap: 8px; margin-top: 12px; overflow-x: auto; }
.picker { flex: none; width: 64px; height: 80px; border-radius: 10px; border: 2px solid var(--line); background: var(--panel) center / cover; cursor: pointer; color: var(--muted); font-size: 11px; }
.picker.on { border-color: var(--glow); box-shadow: 0 0 14px #7ef0d366; }
/* how it works */
.steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }
.step .n { width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center; font-weight: 800; color: #04121f; background: var(--glow); box-shadow: 0 0 18px #7ef0d388; margin-bottom: 12px; }
.step h3 { margin: 0 0 6px; font-size: 18px; }
.step p { color: var(--muted); margin: 0; font-size: 14px; }
/* lexicon */
.langs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.lang .big { font-size: 40px; font-weight: 800; color: var(--glow); line-height: 1.1; }
.bar { height: 8px; border-radius: 99px; background: var(--bg2); overflow: hidden; margin: 14px 0 6px; }
.bar > div { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.note { color: var(--muted); font-size: 14px; margin-top: 24px; }
/* avatars */
.gallery { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.clip { padding: 0; overflow: hidden; }
.clip video, .clip .poster { display: block; width: 100%; aspect-ratio: 4 / 5; object-fit: cover; background: var(--bg2); }
.clip .meta { padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.clip .meta small { color: var(--muted); }
/* sources, footer */
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: start; padding: 10px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
footer { padding: 40px 0 60px; color: var(--muted); font-size: 14px; border-top: 1px solid var(--line); }
@media (max-width: 900px) {
  .hero .wrap, .result { grid-template-columns: 1fr; }
  .steps { grid-template-columns: 1fr 1fr; }
  .langs, .gallery { grid-template-columns: 1fr 1fr; }
  .nav .links { display: none; }
}
@media (max-width: 560px) { .langs, .gallery, .steps { grid-template-columns: 1fr; } .ask { flex-direction: column; } }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
```

`web/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { I18nProvider } from "./i18n/i18n";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode><I18nProvider><App /></I18nProvider></StrictMode>,
);
```

`web/src/App.tsx` (placeholder until Task 11):

```tsx
import { useI18n } from "./i18n/i18n";

export default function App() {
  const { t } = useI18n();
  return <main className="wrap"><h1 className="grad">{t("hero.title")}</h1></main>;
}
```

- [ ] **Step 6: Run the tests and the build**

Run: `cd web && npx vitest run && npm run build`
Expected: the two i18n tests PASS; `vite build` writes `web/dist/index.html`, `web/dist/record.html` and
`web/dist/favicon.svg`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore web/
git commit -m "Web: Vite + React scaffold, dark theme, four-language interface with RTL" -m "<attribution lines>"
```

---

### Task 7: API client with readable errors

**Files:**
- Create: `web/src/api.ts`, `web/src/api.test.ts`

**Interfaces:**
- Consumes: types from `src/types.ts`.
- Produces: `class ApiError extends Error { kind: "html" | "quota" | "server" | "network" }`;
  `ask(question: string, lang: Lang): Promise<Report>`; `signText(text: string, lang: Lang): Promise<Report>`;
  `getPose(id: string): Promise<PoseFrames>`;
  `getStats(): Promise<Stats>`; `getAvatars(): Promise<string[]>`; `getClips(): Promise<Clip[]>` (never rejects; `[]` on any failure).

- [ ] **Step 1: Write the failing tests** — `web/src/api.test.ts`:

```ts
import { ApiError, ask, getClips, signText } from "./api";

const reply = (body: string, status: number, type: string) =>
  vi.fn().mockResolvedValue(new Response(body, { status, headers: { "content-type": type } }));

afterEach(() => vi.unstubAllGlobals());

test("an HTML page instead of JSON is an 'html' error", async () => {
  vi.stubGlobal("fetch", reply("<!DOCTYPE html><p>sign in</p>", 200, "text/html"));
  await expect(ask("q", "en")).rejects.toMatchObject({ kind: "html" });
});

test("the free-quota 503 is a 'quota' error", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({ detail: "quota used up" }), 503, "application/json"));
  await expect(ask("q", "en")).rejects.toMatchObject({ kind: "quota" });
});

test("other failures carry the server's detail", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({ detail: "GlossError: boom" }), 500, "application/json"));
  const err = await ask("q", "tr").catch((e) => e);
  expect(err).toBeInstanceOf(ApiError);
  expect(err.kind).toBe("server");
  expect(err.message).toContain("boom");
});

test("a network failure is a 'network' error", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
  await expect(ask("q", "ur")).rejects.toMatchObject({ kind: "network" });
});

test("a signed report is returned as is", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({ status: "signed", id: "abc" }), 200, "application/json"));
  await expect(ask("q", "ar")).resolves.toMatchObject({ status: "signed", id: "abc" });
});

test("signing a text posts it to /api/sign, not /api/ask", async () => {
  const f = reply(JSON.stringify({ status: "signed", mode: "text", id: "tenabc" }), 200, "application/json");
  vi.stubGlobal("fetch", f);
  await expect(signText("Prayer is light.", "en")).resolves.toMatchObject({ mode: "text" });
  expect(f.mock.calls[0][0]).toBe("/api/sign");
  expect(JSON.parse(f.mock.calls[0][1].body)).toEqual({ text: "Prayer is light.", lang: "en" });
});

test("missing clips give an empty list", async () => {
  vi.stubGlobal("fetch", reply("Not Found", 404, "text/plain"));
  await expect(getClips()).resolves.toEqual([]);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && npx vitest run src/api.test.ts` — Expected: FAIL (`Cannot find module './api'`).

- [ ] **Step 3: Implement** `web/src/api.ts`:

```ts
import type { Clip, Lang, PoseFrames, Report, Stats } from "./types";

export type ApiErrorKind = "html" | "quota" | "server" | "network";

export class ApiError extends Error {
  constructor(public kind: ApiErrorKind, message: string) { super(message); this.name = "ApiError"; }
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try { r = await fetch(url, init); } catch (e) { throw new ApiError("network", String(e)); }
  // a private Space's sign-in page or a proxy error page arrives as HTML
  if (!(r.headers.get("content-type") || "").includes("json")) throw new ApiError("html", `HTTP ${r.status}`);
  const body = await r.json();
  if (r.status === 503) throw new ApiError("quota", body.detail || "quota");
  if (!r.ok) throw new ApiError("server", body.detail || r.statusText);
  return body as T;
}

export const ask = (question: string, lang: Lang) =>
  call<Report>("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
                             body: JSON.stringify({ question, lang }) });
export const signText = (text: string, lang: Lang) =>  // Studio: the user's own text, no answer generation
  call<Report>("/api/sign", { method: "POST", headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ text, lang }) });
export const getPose = (id: string) => call<PoseFrames>(`/pose/${id}.json`);
export const getStats = () => call<Stats>("/api/stats");
export const getAvatars = () => call<string[]>("/api/avatars");
export const getClips = () => call<Clip[]>("/clips/clips.json").catch(() => [] as Clip[]);
```

- [ ] **Step 4: Run the tests** — `cd web && npx vitest run` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/api.ts web/src/api.test.ts
git commit -m "Web: API client that turns HTML pages, quota and network failures into typed errors" -m "<attribution lines>"
```

---

### Task 8: Player clock, counters and the glowing keypoint canvas

**Files:**
- Create: `web/src/player.ts`, `web/src/components/Counter.tsx`, `web/src/components/KeypointCanvas.tsx`,
  `web/src/components/Counter.test.tsx`, `web/src/components/KeypointCanvas.test.ts`

**Interfaces:**
- Produces: `usePlayer(duration: number, opts?: { loop?: boolean; autoplay?: boolean }) => { t: number; playing: boolean; speed: number; play(): void; pause(): void; toggle(): void; seek(t: number): void; setSpeed(s: number): void }`;
  `<Counter value={number} format?="int" | "pct" duration?={seconds} />`;
  `poseBounds(frames: number[][][]) => { minX: number; maxX: number; minY: number; maxY: number }`;
  `<KeypointCanvas frames={PoseFrames | null} time={number} canvasRef?={RefObject<HTMLCanvasElement>} />`.

- [ ] **Step 1: Write the failing tests.** `web/src/components/Counter.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { I18nProvider } from "../i18n/i18n";
import { Counter } from "./Counter";

test("shows the final value with grouping when not animated", () => {
  render(<I18nProvider initial="en"><Counter value={3213} duration={0} /></I18nProvider>);
  expect(screen.getByText("3,213")).toBeInTheDocument();
});

test("formats a share as a whole percentage", () => {
  render(<I18nProvider initial="en"><Counter value={0.6316} format="pct" duration={0} /></I18nProvider>);
  expect(screen.getByText("63%")).toBeInTheDocument();
});
```

`web/src/components/KeypointCanvas.test.ts`:

```ts
import { poseBounds } from "./KeypointCanvas";

const frame = (x: number, y: number) => Array.from({ length: 50 }, () => [x, y, 0]);

test("bounds span every frame and ignore non-finite values", () => {
  const f1 = frame(-1, -2), f2 = frame(1, 3);
  f2[10] = [NaN, Infinity, 0];
  expect(poseBounds([f1, f2])).toEqual({ minX: -1, maxX: 1, minY: -2, maxY: 3 });
});

test("a single still frame still has a usable box", () => {
  const b = poseBounds([frame(0, 0)]);
  expect(b.maxX - b.minX).toBeGreaterThan(0);
  expect(b.maxY - b.minY).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && npx vitest run src/components` — Expected: FAIL (modules not found).

- [ ] **Step 3: Implement.** `web/src/player.ts`:

```ts
import { useCallback, useEffect, useRef, useState } from "react";

/** A media clock for poses: the avatar, the keypoints and the gloss chips all read t (seconds). */
export function usePlayer(duration: number, opts: { loop?: boolean; autoplay?: boolean } = {}) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(!!opts.autoplay);
  const [speed, setSpeed] = useState(0.75);
  const tRef = useRef(0);
  useEffect(() => { tRef.current = 0; setT(0); setPlaying(!!opts.autoplay); }, [duration, opts.autoplay]);
  useEffect(() => {
    if (!playing || duration <= 0) return;
    let raf = 0, last = performance.now();
    const step = (now: number) => {
      let next = tRef.current + ((now - last) / 1000) * speed;
      last = now;
      if (next >= duration) {
        if (opts.loop) next %= duration;
        else { next = duration; setPlaying(false); }
      }
      tRef.current = next;
      setT(next);
      if (next < duration || opts.loop) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, duration, speed, opts.loop]);
  const seek = useCallback((s: number) => { tRef.current = Math.max(0, Math.min(duration, s)); setT(tRef.current); },
                           [duration]);
  const play = useCallback(() => { if (tRef.current >= duration) seek(0); setPlaying(true); }, [duration, seek]);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => (playing ? pause() : play()), [playing, play, pause]);
  return { t, playing, speed, play, pause, toggle, seek, setSpeed };
}
```

`web/src/components/Counter.tsx`:

```tsx
import { animate, useInView } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "../i18n/i18n";

const LOCALE = { en: "en", ar: "ar-u-nu-latn", tr: "tr", ur: "ur-u-nu-latn" } as const;

export function Counter({ value, format = "int", duration = 1.6 }: { value: number; format?: "int" | "pct"; duration?: number }) {
  const { ui } = useI18n();
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [shown, setShown] = useState(duration === 0 ? value : 0);
  useEffect(() => {
    if (duration === 0) { setShown(value); return; }
    if (!inView) return;
    const c = animate(0, value, { duration, ease: "easeOut", onUpdate: setShown });
    return () => c.stop();
  }, [inView, value, duration]);
  const nf = new Intl.NumberFormat(LOCALE[ui], format === "pct" ? { style: "percent", maximumFractionDigits: 0 }
                                                              : { maximumFractionDigits: 0 });
  return <span ref={ref} className="counter">{nf.format(format === "pct" ? shown : Math.round(shown))}</span>;
}
```

`web/src/components/KeypointCanvas.tsx`:

```tsx
import { useEffect, useMemo, useRef, type RefObject } from "react";
import type { PoseFrames } from "../types";

// 50 joints: 0 nose, 1 neck, 2-4 right shoulder/elbow/wrist, 5-7 left, 8-28 left hand, 29-49 right hand (MediaPipe)
const BODY = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [7, 8], [4, 29]];
const HAND = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
              [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

export function poseBounds(frames: number[][][]) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const f of frames) for (const [x, y] of f) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  if (maxX - minX < 1e-3) { minX -= 1; maxX += 1; }
  if (maxY - minY < 1e-3) { minY -= 1; maxY += 1; }
  return { minX, maxX, minY, maxY };
}

export function KeypointCanvas({ frames, time, canvasRef }: { frames: PoseFrames | null; time: number;
                                                             canvasRef?: RefObject<HTMLCanvasElement> }) {
  const own = useRef<HTMLCanvasElement>(null);
  const ref = canvasRef ?? own;
  const box = useMemo(() => (frames?.frames.length ? poseBounds(frames.frames) : null), [frames]);
  useEffect(() => {
    const c = ref.current;
    if (!c || !frames || !box || !frames.frames.length) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = c.clientWidth * dpr, h = c.clientHeight * dpr;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const g = c.getContext("2d");
    if (!g) return;
    const f = frames.frames[Math.min(frames.frames.length - 1, Math.max(0, Math.round(time * frames.fps)))];
    const pad = 0.12, sx = (w * (1 - 2 * pad)) / (box.maxX - box.minX), sy = (h * (1 - 2 * pad)) / (box.maxY - box.minY);
    const s = Math.min(sx, sy);
    const ox = w / 2 - ((box.minX + box.maxX) / 2) * s, oy = h / 2 - ((box.minY + box.maxY) / 2) * s;
    const P = (i: number) => [f[i][0] * s + ox, f[i][1] * s + oy] as const;
    const ok = (i: number) => Number.isFinite(f[i]?.[0]) && Number.isFinite(f[i]?.[1]);
    g.clearRect(0, 0, w, h);
    g.lineCap = "round";
    g.shadowColor = "#7ef0d3";
    g.shadowBlur = 14 * dpr;
    const lines = (pairs: number[][], base: number, width: number, color: string) => {
      g.strokeStyle = color; g.lineWidth = width * dpr; g.beginPath();
      for (const [a, b] of pairs) {
        if (!ok(a + base) || !ok(b + base)) continue;
        const [x1, y1] = P(a + base), [x2, y2] = P(b + base);
        g.moveTo(x1, y1); g.lineTo(x2, y2);
      }
      g.stroke();
    };
    lines(BODY, 0, 4, "#7ef0d3");
    lines(HAND, 8, 2, "#8fb8ff");
    lines(HAND, 29, 2, "#8fb8ff");
    g.fillStyle = "#e8fffa";
    for (let i = 0; i < 50; i++) if (ok(i)) { const [x, y] = P(i); g.beginPath(); g.arc(x, y, (i < 8 ? 3.5 : 2) * dpr, 0, 7); g.fill(); }
    if (ok(0) && ok(1)) {  // the head: a ring around the nose, sized by the neck length
      const [nx, ny] = P(0), [kx, ky] = P(1);
      g.strokeStyle = "#7ef0d3"; g.lineWidth = 3 * dpr; g.beginPath();
      g.arc(nx, ny, Math.hypot(nx - kx, ny - ky) * 0.55, 0, 7); g.stroke();
    }
  }, [frames, time, box, ref]);
  return <canvas ref={ref} aria-label="keypoints" />;
}
```

- [ ] **Step 4: Run the tests** — `cd web && npx vitest run` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/player.ts web/src/components/
git commit -m "Web: pose clock, animated counters, glowing keypoint canvas" -m "<attribution lines>"
```

---

### Task 9: Avatar view and picker, sharing the classic page's avatar module

**Files:**
- Create: `web/src/state.tsx`, `web/src/components/AvatarView.tsx`, `web/src/components/AvatarPicker.tsx`,
  `web/src/components/StyleSwitch.tsx`, `web/src/components/StyleSwitch.test.tsx`,
  `web/src/components/avatars.ts`, `web/src/components/avatars.test.ts`
- Modify: `web/src/styles.css` (style switch), `web/src/i18n/{en,ar,tr,ur}.json` (style and outfit keys below)

**Interfaces:**
- Consumes: `SignAvatar`, `PRESETS`, `applyOutfit` (`@static/*`); `getStats`, `getAvatars`, `getClips` (Task 7); fallback `@static/stats.json`.
- Produces: `AVATAR_KEYS: Record<string, string>` (file → i18n key), `DEFAULT_AVATAR = "rocketbox_female_06.vrm"`,
  `type AvatarStyle = "cartoon" | "realistic"`, `type Outfit = "hijab" | "shemagh" | "original"`, `STYLES`, `OUTFITS`,
  `styleOf(model): AvatarStyle`, `avatarsFor(style, available): string[]`,
  `avatarForStyle(style, last: Partial<Record<AvatarStyle, string>>, available): string`,
  `pickAvatar(saved: string | null, available: string[]): string`, `outfitFor(model: string, outfit?: Outfit)`;
  `AppStateProvider`, `useAppState(): { avatar; setAvatar(m); avatars: string[]; style: AvatarStyle; setStyle(s); outfit: Outfit; setOutfit(o); stats: Stats; clips: Clip[] }`;
  `<AvatarView model={string} outfit?={Outfit} frames={PoseFrames | null} time={number} onError={() => void} canvasRef?={RefObject<HTMLCanvasElement>} />`;
  `<AvatarPicker value={string} onChange={(m: string) => void} />` (shows the avatars of `value`'s style);
  `<StyleSwitch />` (Cartoon | Realistic, then the realistic picker or the cartoon outfits).

**Interface strings to add in this task** (append to each JSON file from Task 6; the i18n completeness test enforces
all four):

| key | en | ar | tr | ur |
|---|---|---|---|---|
| `style.label` | Avatar style | نمط الشخصية | Avatar tarzı | اوتار کا انداز |
| `style.cartoon` | Cartoon | كرتونية | Çizgi film | کارٹون |
| `style.realistic` | Realistic | واقعية | Gerçekçi | حقیقت پسند |
| `style.cartoonNote` | A stylised VRM avatar; choose its outfit. | شخصية VRM مرسومة؛ اختر لباسها. | Stilize bir VRM avatarı; kıyafetini seçin. | ایک کارٹون VRM اوتار؛ اس کا لباس منتخب کریں۔ |
| `style.realisticNote` | Five Microsoft Rocketbox characters, converted in Blender; same signs, same motion. | خمس شخصيات من Microsoft Rocketbox حُوّلت في Blender؛ الإشارات والحركة نفسها. | Blender'da dönüştürülmüş beş Microsoft Rocketbox karakteri; aynı işaretler, aynı hareket. | Blender میں تبدیل کیے گئے مائیکروسافٹ Rocketbox کے پانچ کردار؛ وہی اشارے، وہی حرکت۔ |
| `outfit.label` | Outfit | اللباس | Kıyafet | لباس |
| `outfit.hijab` | Hijab | حجاب | Başörtüsü | حجاب |
| `outfit.shemagh` | Shemagh and thobe | شماغ وثوب | Puşi ve cüppe | رومال اور ثوب |
| `outfit.original` | Original | الأصلي | Orijinal | اصل |

- [ ] **Step 1: Write the failing test** — `web/src/components/avatars.test.ts`:

```ts
import { AVATAR_KEYS, DEFAULT_AVATAR, avatarForStyle, avatarsFor, outfitFor, pickAvatar, styleOf } from "./avatars";

const all = Object.keys(AVATAR_KEYS);

test("a saved avatar is kept when it still exists", () => {
  expect(pickAvatar("rocketbox_male_19.vrm", all)).toBe("rocketbox_male_19.vrm");
});

test("a saved avatar that no longer exists falls back to the default", () => {
  expect(pickAvatar("old_model.vrm", all)).toBe(DEFAULT_AVATAR);
  expect(pickAvatar(null, all)).toBe(DEFAULT_AVATAR);
});

test("without the default, the first available avatar is used", () => {
  expect(pickAvatar(null, ["avatar.vrm"])).toBe("avatar.vrm");
});

test("realistic avatars keep their own clothes; the cartoon one wears the chosen outfit (hijab by default)", () => {
  expect(outfitFor("rocketbox_female_06.vrm", "shemagh")).toEqual({ style: "none" });
  expect(outfitFor("avatar.vrm")).toMatchObject({ style: "hijab" });
  expect(outfitFor("avatar.vrm", "shemagh")).toMatchObject({ style: "shemagh" });
  expect(outfitFor("avatar.vrm", "original")).toEqual({ style: "none" });
});

test("styles split the avatars into cartoon and realistic", () => {
  expect(styleOf("avatar.vrm")).toBe("cartoon");
  expect(styleOf("rocketbox_male_19.vrm")).toBe("realistic");
  expect(avatarsFor("realistic", all)).toHaveLength(5);
  expect(avatarsFor("cartoon", all)).toEqual(["avatar.vrm"]);
});

test("switching style returns to the last avatar used in that style", () => {
  const last = { cartoon: "avatar.vrm", realistic: "rocketbox_male_21.vrm" };
  expect(avatarForStyle("realistic", last, all)).toBe("rocketbox_male_21.vrm");
  expect(avatarForStyle("realistic", { cartoon: "avatar.vrm", realistic: "gone.vrm" }, all)).toBe(DEFAULT_AVATAR);
  expect(avatarForStyle("cartoon", { realistic: DEFAULT_AVATAR }, all)).toBe("avatar.vrm");
});
```

- [ ] **Step 2: Run to verify it fails** — `cd web && npx vitest run src/components/avatars.test.ts` — Expected: FAIL.

- [ ] **Step 3: Implement.** `web/src/components/avatars.ts`:

```ts
export const DEFAULT_AVATAR = "rocketbox_female_06.vrm";

export const AVATAR_KEYS: Record<string, string> = {
  "rocketbox_female_06.vrm": "avatar.female_06",
  "rocketbox_female_10.vrm": "avatar.female_10",
  "rocketbox_male_15.vrm": "avatar.male_15",
  "rocketbox_male_19.vrm": "avatar.male_19",
  "rocketbox_male_21.vrm": "avatar.male_21",
  "avatar.vrm": "avatar.stylised",
};

export function pickAvatar(saved: string | null, available: string[]): string {
  if (saved && available.includes(saved)) return saved;
  if (available.includes(DEFAULT_AVATAR)) return DEFAULT_AVATAR;
  return available[0] ?? DEFAULT_AVATAR;
}

// Two styles to choose between: the cartoon (pixiv VRM sample) and the realistic Microsoft Rocketbox avatars
// converted in Blender. Same signs, same retargeting; only the model changes.
export type AvatarStyle = "cartoon" | "realistic";
export type Outfit = "hijab" | "shemagh" | "original";
export const STYLES: AvatarStyle[] = ["cartoon", "realistic"];
export const OUTFITS: Outfit[] = ["hijab", "shemagh", "original"];
export const styleOf = (model: string): AvatarStyle => (model.startsWith("rocketbox_") ? "realistic" : "cartoon");
export const avatarsFor = (style: AvatarStyle, available: string[]) => available.filter((m) => styleOf(m) === style);

export function avatarForStyle(style: AvatarStyle, last: Partial<Record<AvatarStyle, string>>, available: string[]) {
  const pool = avatarsFor(style, available);
  const remembered = last[style];
  if (remembered && pool.includes(remembered)) return remembered;
  if (pool.includes(DEFAULT_AVATAR)) return DEFAULT_AVATAR;
  return pool[0] ?? pickAvatar(null, available);
}

// same presets as the classic page (static/outfits.js PRESETS); Rocketbox avatars come dressed modestly
const PRESET = {
  original: { style: "none" },
  hijab: { style: "hijab", head: "#2f3e5c", clothes: "#6d5a78", sleeves: true },
  shemagh: { style: "shemagh", head: "#c1272d", clothes: "#f3f0e8", sleeves: true },
};
export const outfitFor = (model: string, outfit: Outfit = "hijab") =>
  styleOf(model) === "realistic" ? PRESET.original : PRESET[outfit];
```

`web/src/state.tsx`:

```tsx
import fallbackStats from "@static/stats.json";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getAvatars, getClips, getStats } from "./api";
import { AVATAR_KEYS, OUTFITS, avatarForStyle, pickAvatar, styleOf, type AvatarStyle, type Outfit } from "./components/avatars";
import type { Clip, Stats } from "./types";

interface AppState {
  avatar: string; setAvatar: (m: string) => void; avatars: string[];
  style: AvatarStyle; setStyle: (s: AvatarStyle) => void;     // cartoon or realistic, switchable at any time
  outfit: Outfit; setOutfit: (o: Outfit) => void;             // the cartoon avatar's clothes
  stats: Stats; clips: Clip[];
}
const Ctx = createContext<AppState | null>(null);
const KEY = "isharati.avatar", OUTFIT_KEY = "isharati.outfit";
const read = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const write = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [avatars, setAvatars] = useState<string[]>(Object.keys(AVATAR_KEYS));
  const [avatar, setAvatarState] = useState(() => pickAvatar(read(KEY), Object.keys(AVATAR_KEYS)));
  const [last, setLast] = useState<Partial<Record<AvatarStyle, string>>>(() => ({ [styleOf(avatar)]: avatar }));
  const [outfit, setOutfitState] = useState<Outfit>(() => {
    const o = read(OUTFIT_KEY) as Outfit | null;
    return o && OUTFITS.includes(o) ? o : "hijab";
  });
  const [stats, setStats] = useState<Stats>(fallbackStats);  // the built-in figures until /api/stats answers
  const [clips, setClips] = useState<Clip[]>([]);
  useEffect(() => {
    getStats().then(setStats).catch(() => {});
    getClips().then(setClips);
    getAvatars().then((list) => { setAvatars(list); setAvatarState((cur) => pickAvatar(cur, list)); }).catch(() => {});
  }, []);
  const setAvatar = (m: string) => {
    setAvatarState(m); write(KEY, m);
    setLast((l) => ({ ...l, [styleOf(m)]: m }));
  };
  const setStyle = (s: AvatarStyle) => setAvatar(avatarForStyle(s, last, avatars));
  const setOutfit = (o: Outfit) => { setOutfitState(o); write(OUTFIT_KEY, o); };
  return (
    <Ctx.Provider value={{ avatar, setAvatar, avatars, style: styleOf(avatar), setStyle, outfit, setOutfit, stats, clips }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAppState(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppState outside AppStateProvider");
  return v;
}
```

`web/src/components/AvatarView.tsx`:

```tsx
import { SignAvatar } from "@static/avatar.js";
import { applyOutfit } from "@static/outfits.js";
import { useEffect, useRef, useState, type RefObject } from "react";
import type { PoseFrames } from "../types";
import { outfitFor, type Outfit } from "./avatars";

/** The 3D avatar (three.js + VRM) driven by pose frames at time t. One canvas; a new model replaces the old one. */
export function AvatarView({ model, outfit = "hijab", frames, time, onError, canvasRef }: {
  model: string; outfit?: Outfit; frames: PoseFrames | null; time: number; onError: () => void;
  canvasRef?: RefObject<HTMLCanvasElement>;
}) {
  const own = useRef<HTMLCanvasElement>(null);
  const ref = canvasRef ?? own;
  const [av, setAv] = useState<SignAvatar | null>(null);
  const last = useRef(performance.now());
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let alive = true, made: SignAvatar | null = null;
    (async () => {
      try {
        made = await new SignAvatar(canvas).load("/static/" + model);
        applyOutfit(made, outfitFor(model, outfit));
        if (alive) setAv(made); else made.renderer.dispose();
      } catch (e) {
        console.error("avatar", e);
        if (alive) onError();
      }
    })();
    return () => { alive = false; made?.renderer.dispose(); setAv(null); };
  }, [model]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { av?.setFrames(frames); av?.render(time, 1 / 30); }, [av, frames]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (av) { applyOutfit(av, outfitFor(model, outfit)); av.render(time); } }, [av, outfit]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!av) return;
    const now = performance.now();
    av.render(time, Math.min(0.1, (now - last.current) / 1000));
    last.current = now;
  }, [av, time]);
  useEffect(() => {
    const c = ref.current;
    if (!c || !av) return;
    const ro = new ResizeObserver(() => { av.resize(); av.render(time); });
    ro.observe(c);
    return () => ro.disconnect();
  }, [av]);  // eslint-disable-line react-hooks/exhaustive-deps
  return <canvas ref={ref} aria-label="avatar" />;
}
```

`web/src/components/AvatarPicker.tsx`:

```tsx
import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";
import { AVATAR_KEYS, avatarsFor, styleOf } from "./avatars";

/** The avatars of one style, as thumbnails (each avatar's clip poster once clips exist). */
export function AvatarPicker({ value, onChange }: { value: string; onChange: (m: string) => void }) {
  const { t } = useI18n();
  const { avatars, clips } = useAppState();
  return (
    <div className="pickers" role="radiogroup" aria-label={t("try.pick")}>
      {avatarsFor(styleOf(value), avatars).map((m) => {
        const poster = clips.find((c) => c.avatar === m)?.poster;
        const name = t(AVATAR_KEYS[m] ?? m);
        return (
          <button key={m} type="button" role="radio" aria-checked={m === value} title={name}
                  className={"picker" + (m === value ? " on" : "")} onClick={() => onChange(m)}
                  style={poster ? { backgroundImage: `url(${poster})` } : undefined}>
            {poster ? "" : name}
          </button>
        );
      })}
    </div>
  );
}
```

`web/src/components/StyleSwitch.tsx` — the demo switch between the cartoon and the realistic (Blender-converted)
avatars. The pose clock is outside it, so switching mid-answer keeps signing from the same moment:

```tsx
import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";
import { AvatarPicker } from "./AvatarPicker";
import { OUTFITS, STYLES } from "./avatars";

export function StyleSwitch() {
  const { t } = useI18n();
  const { avatar, setAvatar, style, setStyle, outfit, setOutfit } = useAppState();
  return (
    <div className="style-switch">
      <div className="seg" role="radiogroup" aria-label={t("style.label")}>
        {STYLES.map((s) => (
          <button key={s} role="radio" aria-checked={s === style} className={s === style ? "on" : ""}
                  onClick={() => setStyle(s)}>{t(`style.${s}`)}</button>
        ))}
      </div>
      <p className="note">{t(`style.${style}Note`)}</p>
      {style === "realistic"
        ? <AvatarPicker value={avatar} onChange={setAvatar} />
        : (
          <div className="seg" role="radiogroup" aria-label={t("outfit.label")}>
            {OUTFITS.map((o) => (
              <button key={o} role="radio" aria-checked={o === outfit} className={o === outfit ? "on" : ""}
                      onClick={() => setOutfit(o)}>{t(`outfit.${o}`)}</button>
            ))}
          </div>
        )}
    </div>
  );
}
```

`web/src/components/StyleSwitch.test.tsx`:

```tsx
import { act, render, screen } from "@testing-library/react";
import en from "../i18n/en.json";
import { I18nProvider } from "../i18n/i18n";
import { AppStateProvider, useAppState } from "../state";
import { StyleSwitch } from "./StyleSwitch";

function Current() { const { avatar, outfit } = useAppState(); return <p data-testid="cur">{avatar}|{outfit}</p>; }

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("x", { status: 404, headers: { "content-type": "text/plain" } })));
});
afterEach(() => vi.unstubAllGlobals());

test("switching style swaps the avatar and back, remembering the realistic choice", async () => {
  await act(async () => {
    render(<I18nProvider initial="en"><AppStateProvider><StyleSwitch /><Current /></AppStateProvider></I18nProvider>);
  });
  expect(screen.getByTestId("cur").textContent).toBe("rocketbox_female_06.vrm|hijab");  // realistic by default
  act(() => screen.getByTitle(en["avatar.male_19"]).click());
  act(() => screen.getByText(en["style.cartoon"]).click());
  expect(screen.getByTestId("cur").textContent).toBe("avatar.vrm|hijab");
  act(() => screen.getByText(en["outfit.shemagh"]).click());
  expect(screen.getByTestId("cur").textContent).toBe("avatar.vrm|shemagh");
  act(() => screen.getByText(en["style.realistic"]).click());
  expect(screen.getByTestId("cur").textContent).toBe("rocketbox_male_19.vrm|shemagh");
});
```

Add to `web/src/styles.css`:

```css
.style-switch { margin-top: 14px; display: grid; gap: 8px; }
.style-switch .note { margin: 0; }
```

- [ ] **Step 4: Run the tests and the build** — `cd web && npx vitest run && npm run build`
Expected: tests PASS; the build succeeds and bundles `avatar.js` from `src/isharati/app/static` with three from
`web/node_modules` (look for `three.module` in the build output's module list, or check that `web/dist/assets/*.js`
contains `VRMLoaderPlugin`: `grep -l VRMLoaderPlugin web/dist/assets/*.js`).

- [ ] **Step 5: Commit**

```bash
git add web/src/state.tsx web/src/components/
git commit -m "Web: avatar view and picker on the shared avatar module, app state with stats and clips" -m "<attribution lines>"
```

---

### Task 10: The translator section

**Files:**
- Create: `web/src/components/GlossChips.tsx`, `web/src/components/GlossChips.test.tsx`, `web/src/sections/Translator.tsx`,
  `web/src/sections/translator.ts`, `web/src/sections/translator.test.ts`
- Modify: `web/src/styles.css` (append the Studio textarea rule below), `web/src/i18n/{en,ar,tr,ur}.json` (keys below)

**Studio (text to sign) strings to add in this task:**

| key | en | ar | tr | ur |
|---|---|---|---|---|
| `nav.studio` | Studio | الاستوديو | Stüdyo | اسٹوڈیو |
| `try.modeAsk` | Ask a question | اطرح سؤالًا | Soru sor | سوال پوچھیں |
| `try.modeText` | Translate my text | ترجم نصّي | Metnimi çevir | میرا متن ترجمہ کریں |
| `try.textLead` | Write or paste your own content; it is signed as written, with no answer generated. | اكتب محتواك أو الصقه؛ يُترجم إلى الإشارة كما هو، دون توليد إجابة. | Kendi içeriğinizi yazın veya yapıştırın; cevap üretilmeden, yazıldığı gibi işaretlenir. | اپنا مواد لکھیں یا چسپاں کریں؛ کوئی جواب بنائے بغیر، جیسا لکھا ہے ویسا اشاروں میں ہوگا۔ |
| `try.textPlaceholder` | Your text, in the chosen language… | نصّك باللغة المختارة… | Seçilen dilde metniniz… | منتخب زبان میں آپ کا متن… |
| `try.signText` | Translate to sign | ترجم إلى الإشارة | İşarete çevir | اشاروں میں ترجمہ کریں |
| `try.yourText` | Your text | نصّك | Metniniz | آپ کا متن |
| `try.unchecked` | Your text — not checked against the Qur'an and hadith sources. | نصّك — لم يُراجَع على مصادر القرآن والحديث. | Metniniz — Kur'an ve hadis kaynaklarıyla karşılaştırılmadı. | آپ کا متن — قرآن و حدیث کے ماخذ سے جانچا نہیں گیا۔ |
| `try.ruleGlosser` | glossed word by word (the language model was unavailable) | تُرجم كلمةً كلمة (النموذج اللغوي غير متاح) | kelime kelime çevrildi (dil modeli kullanılamadı) | لفظ بہ لفظ ترجمہ (لسانی ماڈل دستیاب نہیں تھا) |

```css
.ask.studio { align-items: flex-start; }
.ask textarea { flex: 1; min-width: 0; resize: vertical; background: var(--bg2); color: var(--ink); border: 1px solid var(--line);
  border-radius: 12px; padding: 14px 16px; font: 17px/1.6 var(--font); }
.ask textarea:focus { outline: 2px solid var(--accent); }
```

**Interfaces:**
- Consumes: `ask`, `getPose`, `ApiError` (Task 7); `usePlayer`, `KeypointCanvas` (Task 8); `AvatarView`, `StyleSwitch`, `useAppState` (Task 9); `useI18n`.
- Produces: `<GlossChips segments={Segment[]} time={number} onSeek={(s: number) => void} />`;
  `SIGN_LANGS: { code: Lang; label: string }[]`, `EXAMPLES: Record<Lang, string[]>`, `textDir(lang: Lang): "ltr" | "rtl"`,
  `errorKey(e: unknown): string` (i18n key); `<Translator />` rendered with `id="try"`.

- [ ] **Step 1: Write the failing tests.** `web/src/components/GlossChips.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import type { Segment } from "../types";
import { GlossChips } from "./GlossChips";

const segs: Segment[] = [
  { label: "ALLAH", kind: "sign", start_s: 0.2, end_s: 1.0 },
  { label: "be", kind: "missing", start_s: 1.3, end_s: 1.7 },
  { label: "PRAYER", kind: "sign", start_s: 2.0, end_s: 3.0 },
];

test("the chip being signed now is highlighted; missing signs are red", () => {
  render(<GlossChips segments={segs} time={2.5} onSeek={() => {}} />);
  expect(screen.getByText("PRAYER")).toHaveClass("now");
  expect(screen.getByText("ALLAH")).not.toHaveClass("now");
  expect(screen.getByText("be")).toHaveClass("missing");
});

test("clicking a chip seeks to its start", () => {
  const onSeek = vi.fn();
  render(<GlossChips segments={segs} time={0} onSeek={onSeek} />);
  screen.getByText("PRAYER").click();
  expect(onSeek).toHaveBeenCalledWith(2.0);
});

test("no segments renders an empty list", () => {
  const { container } = render(<GlossChips segments={[]} time={0} onSeek={() => {}} />);
  expect(container.querySelectorAll(".chip")).toHaveLength(0);
});
```

`web/src/sections/translator.test.ts`:

```ts
import { ApiError } from "../api";
import { EXAMPLES, SIGN_LANGS, errorKey, textDir } from "./translator";

test("text direction follows the sign language, not the interface", () => {
  expect(textDir("en")).toBe("ltr");
  expect(textDir("tr")).toBe("ltr");
  expect(textDir("ar")).toBe("rtl");
  expect(textDir("ur")).toBe("rtl");
});

test("four sign languages, each with examples", () => {
  expect(SIGN_LANGS.map((s) => s.code)).toEqual(["en", "ar", "tr", "ur"]);
  for (const s of SIGN_LANGS) expect(EXAMPLES[s.code].length).toBeGreaterThanOrEqual(3);
});

test("errors map to localised messages", () => {
  expect(errorKey(new ApiError("html", "x"))).toBe("try.html");
  expect(errorKey(new ApiError("quota", "x"))).toBe("try.quota");
  expect(errorKey(new ApiError("server", "x"))).toBe("try.error");
  expect(errorKey(new Error("x"))).toBe("try.error");
});
```

- [ ] **Step 2: Run to verify they fail** — `cd web && npx vitest run src/components/GlossChips.test.tsx src/sections` — Expected: FAIL.

- [ ] **Step 3: Implement.** `web/src/components/GlossChips.tsx`:

```tsx
import type { Segment } from "../types";

export function GlossChips({ segments, time, onSeek }: { segments: Segment[]; time: number; onSeek: (s: number) => void }) {
  return (
    <div className="chips">
      {segments.map((s, i) => {
        const now = time >= s.start_s && time < s.end_s;
        return (
          <button key={i} type="button" className={"chip" + (s.kind === "missing" ? " missing" : "") + (now ? " now" : "")}
                  onClick={() => onSeek(s.start_s)}>{s.label}</button>
        );
      })}
    </div>
  );
}
```

`web/src/sections/translator.ts`:

```ts
import { ApiError } from "../api";
import { RTL } from "../i18n/i18n";
import type { Lang } from "../types";

export const SIGN_LANGS: { code: Lang; label: string }[] = [
  { code: "en", label: "English → ASL" },
  { code: "ar", label: "العربية → ArSL" },
  { code: "tr", label: "Türkçe → TİD" },
  { code: "ur", label: "اردو → ISL" },
];

// written in the question's language; the English and Arabic ones are already signed and replay instantly
export const EXAMPLES: Record<Lang, string[]> = {
  en: ["What are the pillars of Islam?", "What is Ramadan?", "How many prayers a day?"],
  ar: ["ما هو الإسلام؟", "ما هي الزكاة؟", "كم عدد الصلوات في اليوم؟"],
  tr: ["Ramazan nedir?", "Namaz nedir?", "Zekât nedir?"],
  ur: ["رمضان کیا ہے؟", "نماز کیا ہے؟", "زکوٰۃ کیا ہے؟"],
};

export const textDir = (lang: Lang) => (RTL.has(lang) ? "rtl" : "ltr");

export function errorKey(e: unknown): string {
  if (e instanceof ApiError && e.kind === "html") return "try.html";
  if (e instanceof ApiError && e.kind === "quota") return "try.quota";
  return "try.error";
}
```

`web/src/sections/Translator.tsx`:

```tsx
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { ask, getPose, signText } from "../api";
import { AvatarView } from "../components/AvatarView";
import { GlossChips } from "../components/GlossChips";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { StyleSwitch } from "../components/StyleSwitch";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { Lang, PoseFrames, Report } from "../types";
import { EXAMPLES, SIGN_LANGS, errorKey, textDir } from "./translator";

type Status = { kind: "idle" } | { kind: "wait" } | { kind: "error"; text: string } | { kind: "info"; text: string };

export function Translator() {
  const { t, ui } = useI18n();
  const { avatar, outfit } = useAppState();
  const [lang, setLang] = useState<Lang>(ui);
  // "ask": a question answered from the sources; "text": Studio, the user's own text signed as written
  const [kind, setKind] = useState<"ask" | "text">(() => (location.hash === "#studio" ? "text" : "ask"));
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [report, setReport] = useState<Report | null>(null);
  const [frames, setFrames] = useState<PoseFrames | null>(null);
  const [mode, setMode] = useState<"avatar" | "keypoints">("avatar");
  const [avatarFailed, setAvatarFailed] = useState(false);
  const [exporting, setExporting] = useState(false);
  const avatarCanvas = useRef<HTMLCanvasElement>(null);
  const keypointCanvas = useRef<HTMLCanvasElement>(null);
  const duration = frames ? frames.frames.length / frames.fps : 0;
  const player = usePlayer(duration, { autoplay: true });

  useEffect(() => { if (!report) setLang(ui); }, [ui, report]);  // follow the interface until something is asked
  useEffect(() => {                                              // a shareable link: /?q=...&lang=tr#try
    const p = new URLSearchParams(location.search), q0 = p.get("q"), l0 = p.get("lang") as Lang | null;
    if (q0) { const l = SIGN_LANGS.some((s) => s.code === l0) ? l0! : ui; setLang(l); setQ(q0); run(q0, l, "ask"); }
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {                                              // the nav's Studio link (#studio) opens the text tab
    const onHash = () => { if (location.hash === "#studio") { switchKind("text"); document.getElementById("try")?.scrollIntoView(); } };
    onHash();
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  function switchKind(k: "ask" | "text") {
    setKind(k); setQ(""); setReport(null); setFrames(null); setStatus({ kind: "idle" });
  }

  async function run(question: string, l: Lang, as: "ask" | "text" = kind) {
    if (!question.trim()) return;
    setStatus({ kind: "wait" }); setReport(null); setFrames(null);
    try {
      const r = as === "text" ? await signText(question.trim(), l) : await ask(question.trim(), l);
      if (r.status === "referred") return setStatus({ kind: "info", text: t("try.referred") });
      if (r.status === "unanswered") return setStatus({ kind: "info", text: t("try.unanswered") + (r.reason ?? "") });
      setReport(r);
      setStatus({ kind: "idle" });
      if (r.id) setFrames(await getPose(r.id).catch(() => null));
    } catch (e) {
      setStatus({ kind: "error", text: t(errorKey(e)) + (errorKey(e) === "try.error" ? String((e as Error).message) : "") });
    }
  }

  async function exportVideo() {
    const canvas = (mode === "avatar" && !avatarFailed ? avatarCanvas : keypointCanvas).current;
    if (!canvas || !duration) return;
    const type = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm"].find((x) => MediaRecorder.isTypeSupported(x))!;
    const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: type, videoBitsPerSecond: 6e6 });
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob(chunks, { type }));
      a.download = `isharati_${lang}.${type.startsWith("video/mp4") ? "mp4" : "webm"}`;
      a.click();
      setExporting(false);
    };
    setExporting(true);
    player.seek(0); player.play(); rec.start();
    setTimeout(() => rec.stop(), (duration / player.speed) * 1000 + 300);
  }

  const showAvatar = mode === "avatar" && !avatarFailed;
  const dir = textDir(lang);
  return (
    <section id="try">
      <div className="wrap">
        <h2>{t("try.title")}</h2>
        <p className="lead">{t(kind === "text" ? "try.textLead" : "try.lead")}</p>
        <div className="seg" role="tablist" style={{ marginBottom: 16 }}>
          <button role="tab" aria-selected={kind === "ask"} className={kind === "ask" ? "on" : ""}
                  onClick={() => switchKind("ask")}>{t("try.modeAsk")}</button>
          <button role="tab" aria-selected={kind === "text"} className={kind === "text" ? "on" : ""}
                  onClick={() => switchKind("text")}>{t("try.modeText")}</button>
        </div>
        <div className="tabs" role="tablist">
          {SIGN_LANGS.map((s) => (
            <button key={s.code} role="tab" aria-selected={s.code === lang} className={s.code === lang ? "on" : ""}
                    onClick={() => { setLang(s.code); setQ(""); setReport(null); setFrames(null); setStatus({ kind: "idle" }); }}>
              {s.label}
            </button>
          ))}
        </div>
        <form className={kind === "text" ? "ask studio" : "ask"} onSubmit={(e) => { e.preventDefault(); run(q, lang); }}>
          {kind === "text"
            ? <textarea value={q} dir={dir} lang={lang} maxLength={1000} rows={5} placeholder={t("try.textPlaceholder")}
                        onChange={(e) => setQ(e.target.value)} aria-label={t("try.textPlaceholder")} />
            : <input value={q} dir={dir} lang={lang} maxLength={300} placeholder={t("try.placeholder")}
                     onChange={(e) => setQ(e.target.value)} aria-label={t("try.placeholder")} />}
          <button className="btn" disabled={status.kind === "wait"}>{t(kind === "text" ? "try.signText" : "try.go")}</button>
        </form>
        {kind === "text"
          ? <p className="note">{q.length} / 1000</p>
          : <div className="examples" dir={dir}>
              {EXAMPLES[lang].map((x) => <button key={x} type="button" onClick={() => { setQ(x); run(x, lang); }}>{x}</button>)}
            </div>}
        {status.kind === "wait" && <p className="status">{t("try.wait")}</p>}
        {(status.kind === "error" || status.kind === "info") &&
          <p className={"status" + (status.kind === "error" ? " err" : "")}>{status.text}</p>}
        <AnimatePresence>
          {report && (
            <motion.div className="result" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div style={{ display: "grid", gap: 20, alignContent: "start" }}>
                <div className="panel">
                  <p className="label">{t(report.mode === "text" ? "try.yourText" : "try.answer")}</p>
                  <p className="answer" dir={dir} lang={lang}>{report.answer}</p>
                  {report.mode === "text"
                    ? <p className="note">{t("try.unchecked")}{report.glosser === "rule" ? ` · ${t("try.ruleGlosser")}` : ""}</p>
                    : <div className="sources">
                        {report.sources?.map((s) => <a key={s.url} href={s.url} target="_blank" rel="noopener">{s.reference}</a>)}
                      </div>}
                </div>
                <div className="panel">
                  <p className="label">{t("try.gloss")}</p>
                  <div dir={dir} lang={lang}>
                    <GlossChips segments={report.segments ?? []} time={player.t} onSeek={(s) => { player.seek(s); player.play(); }} />
                  </div>
                  <p className="note">
                    {t("try.coverage")} {Math.round((report.coverage ?? 0) * 100)}% · {t("try.missingNote")}
                    {report.cached ? ` · ${t("try.cached")}` : ""}
                  </p>
                </div>
              </div>
              <div className="panel">
                <p className="label">{t("try.video")}</p>
                <div className="seg" style={{ marginBottom: 12 }}>
                  <button className={mode === "avatar" ? "on" : ""} onClick={() => setMode("avatar")}>{t("try.avatar")}</button>
                  <button className={mode === "keypoints" ? "on" : ""} onClick={() => setMode("keypoints")}>{t("try.keypoints")}</button>
                </div>
                <div className="viewer">
                  {showAvatar
                    ? <AvatarView model={avatar} outfit={outfit} frames={frames} time={player.t} canvasRef={avatarCanvas}
                                  onError={() => setAvatarFailed(true)} />
                    : <KeypointCanvas frames={frames} time={player.t} canvasRef={keypointCanvas} />}
                </div>
                {avatarFailed && mode === "avatar" && <p className="status">{t("try.avatarFail")}</p>}
                <div className="transport">
                  <button className="btn ghost" onClick={player.toggle}>{player.playing ? t("try.pause") : t("try.play")}</button>
                  <input type="range" min={0} max={1000} value={duration ? Math.round((player.t / duration) * 1000) : 0}
                         onChange={(e) => player.seek((+e.target.value / 1000) * duration)} aria-label="position" />
                  <select value={player.speed} onChange={(e) => player.setSpeed(+e.target.value)} aria-label={t("try.speed")}>
                    {[0.5, 0.75, 1].map((s) => <option key={s} value={s}>{s}×</option>)}
                  </select>
                  <button className="btn" disabled={exporting || !duration} onClick={exportVideo}>
                    {exporting ? t("try.exporting") : t("try.export")}
                  </button>
                </div>
                {mode === "avatar" && <StyleSwitch />}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Run the tests and the build** — `cd web && npx vitest run && npm run build` — Expected: PASS, build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/GlossChips.tsx web/src/components/GlossChips.test.tsx web/src/sections/
git commit -m "Web: translator with four sign languages, synced gloss chips, avatar/keypoint viewer and export" -m "<attribution lines>"
```

---

### Task 11: Landing sections and page assembly

**Files:**
- Create: `web/src/sections/Nav.tsx`, `Hero.tsx`, `HowItWorks.tsx`, `Lexicon.tsx`, `Avatars.tsx`, `Sources.tsx`,
  `Footer.tsx`, `web/src/components/ClipCard.tsx`, `web/src/App.test.tsx`; placeholder `web/src/assets/hero-pose.json`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: everything above; `hero-pose.json` (`PoseFrames`, replaced with a real pose by Task 12).
- Produces: the full page; section ids `try`, `how`, `lexicon`, `avatars`, `sources`.

- [ ] **Step 1: Write the failing test** — `web/src/App.test.tsx`:

```tsx
import { act, render, screen } from "@testing-library/react";
import App from "./App";
import en from "./i18n/en.json";
import { I18nProvider } from "./i18n/i18n";

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("nope", { status: 404, headers: { "content-type": "text/plain" } })));
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as any;
});
afterEach(() => vi.unstubAllGlobals());

test("the page renders every section, with built-in figures when the API is unreachable", async () => {
  await act(async () => { render(<I18nProvider initial="en"><App /></I18nProvider>); });
  for (const key of ["hero.title", "try.title", "how.title", "lex.title", "av.title", "src.title"]) {
    expect(screen.getAllByText(en[key as keyof typeof en]).length).toBeGreaterThan(0);
  }
  for (const id of ["try", "how", "lexicon", "avatars", "sources"]) expect(document.getElementById(id)).not.toBeNull();
  expect(screen.getAllByText(/TİD/).length).toBeGreaterThan(0);   // Turkish is offered
  expect(screen.getAllByText(/ISL/).length).toBeGreaterThan(0);   // and Urdu
});
```

(`AvatarView` is not mounted until an answer exists, so three.js never runs in jsdom.)

- [ ] **Step 2: Run to verify it fails** — `cd web && npx vitest run src/App.test.tsx` — Expected: FAIL (sections missing).

- [ ] **Step 3: Implement.** Create a placeholder `web/src/assets/hero-pose.json` so the build works before Task 12:

```json
{"fps": 25, "frames": [[[0,-1.2,0],[0,0,0],[-0.5,0,0],[-0.6,0.6,0],[-0.6,1.1,0],[0.5,0,0],[0.6,0.6,0],[0.6,1.1,0]]]}
```

(Only 8 joints: `KeypointCanvas` skips joints that are missing, so this draws the body only. Task 12 replaces it.)

`web/src/sections/Nav.tsx`:

```tsx
import { LANG_LIST, useI18n } from "../i18n/i18n";

const NAMES = { en: "EN", ar: "ع", tr: "TR", ur: "اردو" } as const;

export function Nav() {
  const { t, ui, setUi } = useI18n();
  return (
    <nav className="nav">
      <div className="wrap">
        <a className="logo" href="#top">Isharati <span className="grad">إشارتي</span></a>
        <div className="links">
          {(["try", "studio", "how", "lexicon", "avatars", "sources"] as const).map((id) => <a key={id} href={`#${id}`}>{t(`nav.${id}`)}</a>)}
        </div>
        <div className="seg" role="group" aria-label="language">
          {LANG_LIST.map((l) => <button key={l} className={l === ui ? "on" : ""} onClick={() => setUi(l)}>{NAMES[l]}</button>)}
        </div>
      </div>
    </nav>
  );
}
```

`web/src/sections/Hero.tsx`:

```tsx
import { motion } from "framer-motion";
import heroPose from "../assets/hero-pose.json";
import { Counter } from "../components/Counter";
import { KeypointCanvas } from "../components/KeypointCanvas";
import { useI18n } from "../i18n/i18n";
import { usePlayer } from "../player";
import { useAppState } from "../state";
import type { PoseFrames } from "../types";

const pose = heroPose as PoseFrames;

export function Hero() {
  const { t } = useI18n();
  const { stats, clips } = useAppState();
  const player = usePlayer(pose.frames.length / pose.fps, { loop: true, autoplay: true });
  const signs = stats.languages.reduce((n, l) => n + l.signs, 0);
  const passages = stats.languages.reduce((n, l) => n + l.passages, 0);
  return (
    <header className="hero" id="top">
      <div className="wrap">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
          <div className="kicker">{t("hero.kicker")}</div>
          <h1 className="grad">{t("hero.title")}</h1>
          <p className="lead">{t("hero.lead")}</p>
          <a className="btn" href="#try">{t("hero.cta")} ↓</a>
          <div className="stats">
            <div><b><Counter value={stats.languages.length} /></b>{t("hero.languages")}</div>
            <div><b><Counter value={signs} /></b>{t("hero.signs")}</div>
            <div><b><Counter value={passages} /></b>{t("hero.passages")}</div>
          </div>
        </motion.div>
        <motion.div initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 1, delay: 0.2 }}>
          <div className="stage"><KeypointCanvas frames={pose} time={player.t} /></div>
          {clips.length > 0 && (
            <div className="reel">
              {clips.slice(0, 3).map((c) => (
                <video key={c.webm} muted loop autoPlay playsInline poster={c.poster}>
                  <source src={c.webm} type="video/webm" /><source src={c.mp4} type="video/mp4" />
                </video>
              ))}
            </div>
          )}
        </motion.div>
      </div>
    </header>
  );
}
```

`web/src/sections/HowItWorks.tsx`:

```tsx
import { motion } from "framer-motion";
import { useI18n } from "../i18n/i18n";

export function HowItWorks() {
  const { t } = useI18n();
  return (
    <section id="how">
      <div className="wrap">
        <h2>{t("how.title")}</h2>
        <div className="steps">
          {[1, 2, 3, 4, 5].map((n) => (
            <motion.div key={n} className="step panel" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true, amount: 0.4 }} transition={{ duration: 0.5, delay: n * 0.12 }}>
              <div className="n">{n}</div>
              <h3>{t(`how.${n}.t`)}</h3>
              <p>{t(`how.${n}.b`)}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
```

`web/src/sections/Lexicon.tsx`:

```tsx
import { motion } from "framer-motion";
import { Counter } from "../components/Counter";
import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";

export function Lexicon() {
  const { t } = useI18n();
  const { stats } = useAppState();
  return (
    <section id="lexicon">
      <div className="wrap">
        <h2>{t("lex.title")}</h2>
        <p className="lead">{t("lex.lead")}</p>
        <div className="langs">
          {stats.languages.map((l) => (
            <motion.div key={l.code} className="lang panel" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}>
              <p className="label">{l.name} → {l.sign_language}</p>
              <div className="big"><Counter value={l.signs} /></div>
              <div>{t("lex.signs")} · <Counter value={l.passages} /> {t("lex.passages")}</div>
              <div className="bar">
                <motion.div initial={{ width: 0 }} whileInView={{ width: `${l.coverage * 100}%` }} viewport={{ once: true }}
                            transition={{ duration: 1.4, ease: "easeOut" }} />
              </div>
              <div><b><Counter value={l.coverage} format="pct" /></b> {t("lex.coverage")}</div>
            </motion.div>
          ))}
        </div>
        <p className="note">{t("lex.note")}</p>
      </div>
    </section>
  );
}
```

`web/src/components/ClipCard.tsx`:

```tsx
import { useInView } from "framer-motion";
import { useEffect, useRef } from "react";
import type { Clip } from "../types";

const touch = typeof matchMedia !== "undefined" && matchMedia("(hover: none)").matches;

/** An avatar's signing clip: plays on hover, or when scrolled into view on touch screens. */
export function ClipCard({ clip }: { clip: Clip }) {
  const ref = useRef<HTMLVideoElement>(null);
  const inView = useInView(ref, { amount: 0.6 });
  useEffect(() => {
    if (!touch || !ref.current) return;
    if (inView) ref.current.play().catch(() => {}); else ref.current.pause();
  }, [inView]);
  return (
    <video ref={ref} muted loop playsInline preload="metadata" poster={clip.poster}
           onMouseEnter={(e) => e.currentTarget.play().catch(() => {})} onMouseLeave={(e) => e.currentTarget.pause()}>
      <source src={clip.webm} type="video/webm" /><source src={clip.mp4} type="video/mp4" />
    </video>
  );
}
```

`web/src/sections/Avatars.tsx`:

```tsx
import { motion } from "framer-motion";
import { AVATAR_KEYS, STYLES, avatarsFor } from "../components/avatars";
import { ClipCard } from "../components/ClipCard";
import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";

export function Avatars() {
  const { t } = useI18n();
  const { avatars, clips, setAvatar } = useAppState();
  return (
    <section id="avatars">
      <div className="wrap">
        <h2>{t("av.title")}</h2>
        <p className="lead">{t("av.lead")}</p>
        {STYLES.map((style) => (
          <div key={style} style={{ marginBottom: 40 }}>
            <h3>{t(`style.${style}`)}</h3>
            <p className="note" style={{ marginTop: 0 }}>{t(`style.${style}Note`)}</p>
            <div className="gallery">
              {avatarsFor(style, avatars).map((m, i) => {
            const clip = clips.find((c) => c.avatar === m);
            return (
              <motion.div key={m} className="clip panel" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
                          viewport={{ once: true }} transition={{ delay: i * 0.08 }}>
                {clip ? <ClipCard clip={clip} /> : <div className="poster" />}
                <div className="meta">
                  <div>{t(AVATAR_KEYS[m] ?? m)}<br /><small>{clip ? `${clip.sign_language} · ${clip.glosses.join(" · ")}` : ""}</small></div>
                  <a className="btn ghost" href="#try" onClick={() => setAvatar(m)}>{t("av.use")}</a>
                </div>
              </motion.div>
            );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
```

`web/src/sections/Sources.tsx`:

```tsx
import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";

export function Sources() {
  const { t } = useI18n();
  const { stats } = useAppState();
  return (
    <section id="sources">
      <div className="wrap">
        <h2>{t("src.title")}</h2>
        <p className="lead">{t("src.lead")}</p>
        <div className="panel" style={{ overflowX: "auto" }}>
          <table>
            <thead><tr><th>{t("src.source")}</th><th>{t("src.licence")}</th><th>{t("src.use")}</th></tr></thead>
            <tbody dir="ltr">{stats.licences.map((l) => <tr key={l.source}><td>{l.source}</td><td>{l.licence}</td><td>{l.use}</td></tr>)}</tbody>
          </table>
        </div>
        <h2 style={{ marginTop: 64 }}>{t("team.title")}</h2>
        <p>{t("team.fatimah")}<br />{t("team.asmaa")}</p>
      </div>
    </section>
  );
}
```

`web/src/sections/Footer.tsx`:

```tsx
import { useI18n } from "../i18n/i18n";

export function Footer() {
  const { t } = useI18n();
  return <footer><div className="wrap">{t("foot.note")} · <a href="/classic">classic</a></div></footer>;
}
```

Replace `web/src/App.tsx`:

```tsx
import { Avatars } from "./sections/Avatars";
import { Footer } from "./sections/Footer";
import { Hero } from "./sections/Hero";
import { HowItWorks } from "./sections/HowItWorks";
import { Lexicon } from "./sections/Lexicon";
import { Nav } from "./sections/Nav";
import { Sources } from "./sections/Sources";
import { Translator } from "./sections/Translator";
import { AppStateProvider } from "./state";

export default function App() {
  return (
    <AppStateProvider>
      <Nav />
      <Hero />
      <main>
        <Translator />
        <HowItWorks />
        <Lexicon />
        <Avatars />
        <Sources />
      </main>
      <Footer />
    </AppStateProvider>
  );
}
```

- [ ] **Step 4: Run the tests and the build** — `cd web && npx vitest run && npm run build` — Expected: all PASS; build succeeds.

- [ ] **Step 5: Look at it.** Start the API (`PYTHONPATH=src py -m isharati.app.server`, port 8765) and in another
shell `cd web && npx vite --port 5173`; open `http://localhost:5173/`, switch the interface to Arabic and Urdu
(layout flips) and back, click an English example, and confirm the avatar signs and the chips light up. Note anything
broken and fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "Web: landing page sections — hero, how it works, lexicon figures, avatar gallery, sources" -m "<attribution lines>"
```

---

### Task 12: Avatar signing clips and the hero pose

**Files:**
- Create: `scripts/avatars/clip_poses.py`, `scripts/avatars/record_clips.py`; replace `web/src/record.ts`
- Generated (commit): `web/public/clips/poses/*.json`, `web/public/clips/*.webm`, `*.mp4`, `*.jpg`, `web/public/clips/clips.json`, `web/src/assets/hero-pose.json`

**Interfaces:**
- Consumes: `pipeline.load_lexicon(lang)` (Task 3); `MissingAwarePoser`; `GlossResult`, `GlossItem`; the built `/record` page (Task 5); ffmpeg on PATH.
- Produces: `clips.json` = `Clip[]` (see `web/src/types.ts`), paths rooted at `/clips/`.

- [ ] **Step 1: Write `scripts/avatars/clip_poses.py`:**

```python
"""Short stitched phrases for the landing page's avatar clips and hero figure: each avatar signs a different phrase
in a different sign language. Writes web/public/clips/poses/<id>.json and web/src/assets/hero-pose.json.

  py scripts/avatars/clip_poses.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import load_lexicon  # noqa: E402
from isharati.pose.poser import MissingAwarePoser  # noqa: E402
from isharati.types import FPS, GlossItem, GlossResult  # noqa: E402

OUT = ROOT / "web" / "public" / "clips" / "poses"
HERO = ROOT / "web" / "src" / "assets" / "hero-pose.json"
# avatar, language, candidate glosses in order of preference; the first 2-3 the lexicon has are used
CLIPS = [
    ("rocketbox_female_06.vrm", "en", ["ALLAH", "PRAY", "PRAYER", "FIVE", "DAY"]),
    ("rocketbox_female_10.vrm", "ar", ["الصلاة", "الله", "مسلم", "يوم"]),
    ("rocketbox_male_15.vrm", "tr", ["allah", "namaz", "dua", "cami", "oruç", "müslüman"]),
    ("rocketbox_male_19.vrm", "ur", ["ALLAH", "PRAYER", "GOD", "PEACE", "MOSQUE"]),
    ("rocketbox_male_21.vrm", "ar", ["الزكاة", "فقير", "مال", "رمضان", "الصوم"]),
    ("avatar.vrm", "en", ["PEACE", "WELCOME", "HELLO", "QURAN", "LEARN"]),
]
SIGN_LANGUAGE = {"en": "ASL", "ar": "ArSL", "tr": "TİD", "ur": "ISL"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lexicons, plan = {}, []
    for avatar, lang, candidates in CLIPS:
        lex = lexicons.setdefault(lang, load_lexicon(lang))
        found = [lex.lookup(g).gloss for g in candidates if lex.lookup(g)][:3]
        if len(found) < 2:
            raise SystemExit(f"{avatar}: fewer than two of {candidates} are in the {lang} lexicon")
        pose, _ = MissingAwarePoser(lex).pose(GlossResult([GlossItem(g) for g in found], "clip"))
        cid = f"{avatar.removesuffix('.vrm')}_{lang}"
        frames = [[[round(float(v), 4) for v in j] for j in f] for f in pose]
        (OUT / f"{cid}.json").write_text(json.dumps({"fps": FPS, "frames": frames}), encoding="utf-8")
        plan.append({"id": cid, "avatar": avatar, "lang": lang, "sign_language": SIGN_LANGUAGE[lang],
                     "glosses": found, "seconds": round(len(pose) / FPS, 2)})
        print(cid, found, f"{len(pose) / FPS:.1f}s")
    (OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    hero = OUT / f"{plan[0]['id']}.json"                      # the hero figure signs the first (English) phrase
    HERO.write_text(hero.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
```

Run: `py scripts/avatars/clip_poses.py`. Expected: six lines, each with 2–3 glosses and 3–6 s. If a clip is longer
than 6 s, drop its third gloss (edit the candidate list) and re-run; if a language has fewer than two candidates,
add common words from that lexicon (`grep -m20 '"gloss"' <lexicon file>`).

- [ ] **Step 2: Replace `web/src/record.ts`** (the recorder page; renders frame by frame so timing is exact):

```ts
import { SignAvatar } from "@static/avatar.js";
import { applyOutfit } from "@static/outfits.js";
import { outfitFor } from "./components/avatars";
import type { PoseFrames } from "./types";

declare global { interface Window { __ready?: boolean; __count?: number; __error?: string; __frame?: (i: number) => string } }

(async () => {
  try {
    const p = new URLSearchParams(location.search);
    const model = p.get("model")!, poseUrl = p.get("pose")!;
    const canvas = document.getElementById("c") as HTMLCanvasElement;
    const av = await new SignAvatar(canvas).load("/static/" + model);
    applyOutfit(av, outfitFor(model));
    av.renderer.setClearColor(0x0b1d33, 1);                  // the site's panel colour, not transparent
    const frames: PoseFrames = await (await fetch(poseUrl)).json();
    av.setFrames(frames);
    window.__count = frames.frames.length;
    window.__frame = (i: number) => { av.render(i / frames.fps, 1 / frames.fps); return canvas.toDataURL("image/jpeg", 0.9); };
    window.__ready = true;
  } catch (e) {
    window.__error = String(e);
  }
})();
```

- [ ] **Step 3: Write `scripts/avatars/record_clips.py`:**

```python
"""Record each planned clip on its avatar (headless Chromium, the built /record page, one JPEG per frame) and encode
WebM (VP9) + MP4 (H.264) + a poster with ffmpeg. Needs web/dist built and the app running on --base.

  cd web && npm run build && cd ..
  PYTHONPATH=src py -m isharati.app.server            # another shell (port 8765)
  py scripts/avatars/record_clips.py [--base http://127.0.0.1:8765]
"""
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "web" / "public" / "clips"
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
MAX_BYTES = 1_500_000


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def main():
    plan = json.loads((CLIPS / "poses" / "plan.json").read_text(encoding="utf-8"))
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for c in plan:
            page = browser.new_page(viewport={"width": 480, "height": 600})
            page.goto(f"{BASE}/record?model={c['avatar']}&pose=/clips/poses/{c['id']}.json")
            page.wait_for_function("window.__ready || window.__error", timeout=180_000)
            if page.evaluate("window.__error"):
                raise SystemExit(f"{c['id']}: {page.evaluate('window.__error')}")
            n = page.evaluate("window.__count")
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(n):
                    data = page.evaluate(f"window.__frame({i})").split(",", 1)[1]
                    Path(tmp, f"{i:04d}.jpg").write_bytes(base64.b64decode(data))
                frames = str(Path(tmp, "%04d.jpg"))
                webm, mp4, poster = CLIPS / f"{c['id']}.webm", CLIPS / f"{c['id']}.mp4", CLIPS / f"{c['id']}.jpg"
                ffmpeg("-framerate", "25", "-i", frames, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "38",
                       "-pix_fmt", "yuv420p", "-an", str(webm))
                ffmpeg("-framerate", "25", "-i", frames, "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4))
                Path(tmp, f"{n // 2:04d}.jpg").replace(poster)
            page.close()
            for f in (webm, mp4):
                if f.stat().st_size > MAX_BYTES:
                    raise SystemExit(f"{f.name} is {f.stat().st_size} bytes, over {MAX_BYTES}: raise -crf and re-run")
            out.append({**{k: c[k] for k in ("avatar", "lang", "sign_language", "glosses", "seconds")},
                        "webm": f"/clips/{webm.name}", "mp4": f"/clips/{mp4.name}", "poster": f"/clips/{poster.name}"})
            print(c["id"], n, "frames", webm.stat().st_size, mp4.stat().st_size)
        browser.close()
    (CLIPS / "clips.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in CLIPS.rglob("*") if f.is_file())
    print("clips folder:", total, "bytes")
    if total > 15_000_000:
        raise SystemExit("clips folder over 15 MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Record.** Run in order:
  1. `cd web && npm run build && cd ..` (puts the pose JSONs and the recorder into `web/dist`)
  2. Start the server in another shell: `PYTHONPATH=src py -m isharati.app.server`. If port 8765 is taken by an
     older copy, use `PORT=8766` and pass `--base http://127.0.0.1:8766`.
  3. `py scripts/avatars/record_clips.py`
  4. `cd web && npm run build` again (copies the clips into `web/dist`).

  Expected: six clips printed, each file ≤ 1.5 MB, folder ≤ 15 MB. Open two `.webm` files and one `.mp4` in a browser
  and check the avatar signs smoothly against the navy background. Open `http://127.0.0.1:8765/#avatars`: each card
  shows a poster and plays on hover; the hero shows the real keypoint figure and a reel of three clips.

- [ ] **Step 5: Commit**

```bash
git add scripts/avatars/clip_poses.py scripts/avatars/record_clips.py web/src/record.ts web/src/assets/hero-pose.json web/public/clips
git commit -m "Clips: every avatar signing a phrase in ASL, ArSL, TİD or ISL, recorded headless and encoded with ffmpeg" -m "<attribution lines>"
```

---

### Task 13: Build the React app in the Space image

**Files:**
- Modify: `deploy/space/Dockerfile`, `scripts/hub/deploy_space.py:25-31`, `deploy/space/README.md` (one line on the landing page)
- Test: `tests/test_hub_deploy.py` (append)

**Interfaces:**
- Consumes: `web/` (Tasks 6–12), `ISHARATI_LANGS`, `ISHARATI_URDU_WSLP`.
- Produces: a staged folder with `Dockerfile`, `requirements.txt`, `README.md`, `src/isharati/**`, `web/**` (no `node_modules`, no `dist`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_hub_deploy.py`:

```python
def test_stage_includes_the_web_app_without_build_output():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("deploy_space", root / "scripts" / "hub" / "deploy_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.stage()
    assert (out / "web" / "package.json").exists() and (out / "web" / "package-lock.json").exists()
    assert (out / "web" / "public" / "clips" / "clips.json").exists()
    assert not (out / "web" / "node_modules").exists() and not (out / "web" / "dist").exists()
    docker = (out / "Dockerfile").read_text(encoding="utf-8")
    assert "npm ci" in docker and "ISHARATI_LANGS=en,ar,tr,ur" in docker and "ISHARATI_URDU_WSLP=1" in docker
```

- [ ] **Step 2: Run to verify it fails** — `py -m pytest tests/test_hub_deploy.py -v -k stage` — Expected: FAIL (`web/package.json` missing).

- [ ] **Step 3: Implement.** In `scripts/hub/deploy_space.py`, add to `stage()` before `return out`:

```python
    shutil.copytree(ROOT / "web", out / "web",  # the React landing page; built inside the image (Dockerfile)
                    ignore=shutil.ignore_patterns("node_modules", "dist", "*.log"))
```

Replace `deploy/space/Dockerfile` with:

```dockerfile
# Isharati on a Hugging Face Docker Space: the React landing page (web/, built here) and the FastAPI app
# (isharati.app.server) with its data fetched from the private dataset at start-up (isharati.hub), models through
# OpenRouter (then NVIDIA NIM / Groq when their keys are set), question embeddings by HF inference.

# 1 · the landing page
FROM node:22-slim AS web
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# the avatar module and the figures are shared with the Python app
COPY src/isharati/app/static/avatar.js src/isharati/app/static/outfits.js src/isharati/app/static/stats.json /build/src/isharati/app/static/
RUN npm run build

# 2 · the app
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=7860 \
    ISHARATI_DATA_DIR=/home/user/data ISHARATI_OUT_DIR=/home/user/data/out \
    ISHARATI_LLM=openrouter ISHARATI_EMBED=hf ISHARATI_LANGS=en,ar,tr,ur ISHARATI_URDU_WSLP=1 \
    ISHARATI_WEB_DIST=/home/user/app/web/dist \
    PYTHONPATH=/home/user/app/src
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user src ./src
COPY --chown=user --from=web /build/web/dist ./web/dist

EXPOSE 7860
CMD ["python", "-m", "isharati.app.server"]
```

Add to `deploy/space/README.md` (below its front matter) the line:
`The landing page is a React app in web/, built by the Dockerfile's first stage; the previous page is at /classic.`

- [ ] **Step 4: Run the tests** — `py -m pytest tests/test_hub_deploy.py -v` — Expected: all PASS.

- [ ] **Step 5: Optional local image check** (only if Docker is installed: `docker --version`):
`py scripts/hub/deploy_space.py --dry-run` prints the staging folder; `docker build -t isharati <that folder>`.
Expected: both stages build. Skip if Docker is not installed and say so in the commit body.

- [ ] **Step 6: Commit**

```bash
git add deploy/space/ scripts/hub/deploy_space.py tests/test_hub_deploy.py
git commit -m "Deploy: build the React landing page in the Space image; Turkish and Urdu data" -m "<attribution lines>"
```

---

### Task 14: End-to-end check, screenshots, README, deploy

**Files:**
- Create: `scripts/web/e2e.py`
- Modify: `README.md` (screenshots, languages, how to run the front end), `docs/paper/figures/ui_en.png`,
  `docs/paper/figures/ui_ar.png`, new `docs/paper/figures/landing.png`, `docs/paper/main.pdf` (rebuilt)

**Interfaces:**
- Consumes: the built app on a local server; cached English (`What are the pillars of Islam?`) and Arabic
  (`ما هو الإسلام؟`) answers in `data/asl/out`.

- [ ] **Step 1: Write `scripts/web/e2e.py`:**

```python
"""End-to-end check of the landing page against a running app, plus the screenshots the README and report use.

  py scripts/web/e2e.py [--base http://127.0.0.1:8765] [--live]    # --live also asks a Turkish and an Urdu question
"""
import sys
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "paper" / "figures"
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
CASES = [("en", "What are the pillars of Islam?", "ui_en"), ("ar", "ما هو الإسلام؟", "ui_ar")]
if "--live" in sys.argv:
    CASES += [("tr", "Ramazan nedir?", None), ("ur", "رمضان کیا ہے؟", None)]
READY = "document.querySelector('.result canvas') && !document.querySelector('.status.err')"


def main():
    failures = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{BASE}/?ui=en")
        page.wait_for_timeout(2500)
        for sid in ("try", "how", "lexicon", "avatars", "sources"):
            if not page.query_selector(f"#{sid}"):
                failures.append(f"missing section #{sid}")
        page.screenshot(path=str(FIG / "landing.png"))
        playing = page.evaluate("Promise.all([...document.querySelectorAll('.reel video')].map(v => v.play().then(() => v.readyState)))")
        if not playing or min(playing) < 2:
            failures.append(f"hero clips not playing: {playing}")
        for lang, q, shot in CASES:
            page.goto(f"{BASE}/?ui={lang}&lang={lang}&q={urllib.parse.quote(q)}#try")
            try:
                page.wait_for_function(READY, timeout=240_000)
            except Exception:
                failures.append(f"{lang}: no signed result ({page.inner_text('#try')[:200]!r})")
                continue
            page.wait_for_timeout(4000)  # avatar loads and plays into a sign
            if shot:
                page.locator("#try").screenshot(path=str(FIG / f"{shot}.png"))
        page.goto(f"{BASE}/?ui=en&lang=en&q={urllib.parse.quote(CASES[0][1])}#try")
        page.wait_for_function(READY, timeout=120_000)
        pickers = page.query_selector_all(".picker")
        if len(pickers) < 5:
            failures.append(f"only {len(pickers)} realistic avatars in the picker")
        else:
            pickers[3].click()
            page.wait_for_timeout(4000)
        # the demo switch: cartoon (with an outfit change) and back to realistic, mid-answer
        page.click(".style-switch .seg button:nth-child(1)")
        page.wait_for_timeout(4000)
        page.click("text=Shemagh and thobe")
        page.wait_for_timeout(1500)
        page.locator("#try").screenshot(path=str(FIG / "ui_cartoon.png"))
        page.click(".style-switch .seg button:nth-child(2)")
        page.wait_for_timeout(4000)
        if not page.query_selector(".style-switch .picker.on"):
            failures.append("switching back to realistic did not restore an avatar")
        # Studio: the user's own text, signed without an answer being generated
        page.goto(f"{BASE}/?ui=en#studio")
        page.wait_for_selector("textarea", timeout=20_000)
        page.fill("textarea", "Prayer is the pillar of the religion.")
        page.click("form.ask button")
        try:
            page.wait_for_function(READY, timeout=240_000)
            if page.query_selector(".sources a"):
                failures.append("text to sign showed sources")
            page.wait_for_timeout(4000)
            page.locator("#try").screenshot(path=str(FIG / "ui_studio.png"))
        except Exception:
            failures.append(f"studio: no signed result ({page.inner_text('#try')[:200]!r})")
        if errors:
            failures.append(f"page errors: {errors}")
        b.close()
    print("\n".join(failures) or "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it.** With the app running on 8765 serving the built `web/dist`:
`py scripts/web/e2e.py --live`. Expected: `all checks passed`, plus `landing.png`, `ui_en.png` and `ui_ar.png`
written. Look at all three screenshots before going on. If the live Turkish or Urdu question fails because of the
LLM quota, re-run without `--live` and report the live check as not done.

- [ ] **Step 3: Full test pass.** `py -m pytest -q` (only the four known environment failures) and
`cd web && npx vitest run && npm run build`.

- [ ] **Step 4: README and report.** In `README.md`: put `docs/paper/figures/landing.png` under the title image; list
all four language pairs (English → ASL, Arabic → ArSL, Turkish → TİD, Urdu → ISL) with the figures from
`src/isharati/app/static/stats.json`; describe the two modes (Ask a question; Studio — translate your own text, with
`docs/paper/figures/ui_studio.png`) and the cartoon / realistic style switch (`docs/paper/figures/ui_cartoon.png`); replace the "Run the app" commands with:

```bash
PYTHONPATH=src python -m isharati.app.server     # API + built page on http://127.0.0.1:8765
cd web && npm install && npm run dev             # front-end development on http://localhost:5173 (proxies the API)
cd web && npm run build                          # production build to web/dist, served by the app at /
```

Check every path the README names still exists (`ls` each). Rebuild the report:
`cd docs/paper && latexmk -xelatex -interaction=nonstopmode -quiet main.tex`, then delete `main.xdv`.

- [ ] **Step 5: Commit and push**

```bash
git add scripts/web/e2e.py README.md docs/paper/figures/ docs/paper/main.pdf
git commit -m "Landing page: end-to-end check, new screenshots, README" -m "<attribution lines>"
git push origin main
```

- [ ] **Step 6: Deploy.** `env -u HF_TOKEN py scripts/hub/deploy_space.py --update`; wait until the Space runtime
is `RUNNING` (`HfApi().get_space_runtime("FatimahEmadEldin/isharati-app").stage`), then with the token check:
`GET /` returns the React page (contains `id="root"`), `GET /api/stats` lists four languages, `GET /clips/clips.json`
lists six clips, and `POST /api/ask` with `{"question": "Ramazan nedir?", "lang": "tr"}` returns `status: signed`
(or the quota 503). The first build takes longer (the Node stage and about 0.3 GB more data). Report the results.
