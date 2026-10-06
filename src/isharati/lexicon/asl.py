"""English gloss -> ASL sign lookup over an ASL lexicon.jsonl (same SignEntry rows and pose contract as Isharati),
with A-Z letter signs for fingerspelling. Same interface as isharati.lexicon.Lexicon, so StitchPoser, the glossers and
the back-translation recogniser work unchanged."""
import json
from pathlib import Path

import numpy as np

from isharati.text.english import candidates, normalize_en
from isharati.pose.keypoints import pin_collapsed_hands, repair_body
from isharati.lexicon.arabic import MissingLetterSign
from isharati.types import SignEntry


def gloss_key(gloss: str) -> str:
    """ASL Citizen glosses carry variant digits (EAT1, EAT2) and separators; the key is the plain English word(s)."""
    g = gloss.replace("_", " ").replace("-", " ")
    return normalize_en("".join(ch for ch in g if not ch.isdigit()))


class ASLLexicon:
    _key = staticmethod(gloss_key)          # gloss -> lookup key (overridden for Turkish)
    _candidates = staticmethod(candidates)  # token -> keys to try, longest first
    _letter_key = staticmethod(str.lower)   # letter gloss -> key (Turkish: I -> ı, İ -> i)

    @staticmethod
    def _spell_chars(word: str) -> str:     # the letters of a word to fingerspell, as letter keys
        return normalize_en(word).replace(" ", "")

    def __init__(self, entries: list[SignEntry], root: Path):
        self.root = Path(root)
        self._signs: dict[str, SignEntry] = {}
        for e in entries:
            if not e.is_letter:
                self._signs.setdefault(self._key(e.gloss), e)  # first variant wins (the builder orders them)
        self._letters = {self._letter_key(e.gloss): e for e in entries if e.is_letter}
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

    def fingerspell(self, word: str) -> list[SignEntry]:
        out = []
        for ch in self._spell_chars(word):
            if ch not in self._letters:
                raise MissingLetterSign(ch)
            out.append(self._letters[ch])
        return out

    def add_letters(self, path: Path) -> int:
        """Letter signs (is_letter rows, keypoints_path relative to this lexicon's root) from another entries file,
        e.g. one signer's alphabet (alphabet/entries.jsonl): they replace any letter of the same name. Word signs are
        untouched, so a single-letter word gloss (WLASL «A») stays a word sign. Returns the number of letters added."""
        path = Path(path)
        if not path.exists():
            return 0
        rows = [SignEntry(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        letters = {self._letter_key(e.gloss): e for e in rows if e.is_letter}
        self._letters.update(letters)
        return len(letters)

    def add_words(self, path: Path) -> int:
        """Word signs from another entries file (keypoints_path relative to this lexicon's root), e.g. fingerspelled
        names (names/entries.jsonl, scripts/lexicon/alphabet/names.py). A gloss the lexicon already signs keeps its
        sign. A row with "variant_of" (another recording of a sign, kept for recognition, e.g. AUTSL's other signers)
        is not a gloss and is skipped. Returns the number of words added."""
        path = Path(path)
        if not path.exists():
            return 0
        added = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("variant_of"):
                    continue
                e = SignEntry(**{k: v for k, v in row.items() if k in SignEntry.__dataclass_fields__})
                if not e.is_letter and self._key(e.gloss) not in self._signs:
                    self._signs[self._key(e.gloss)] = e
                    added += 1
        return added

    def keypoints(self, entry: SignEntry) -> np.ndarray:
        if entry.keypoints_path not in self._cache:
            self._cache[entry.keypoints_path] = repair_body(pin_collapsed_hands(np.load(self.root / entry.keypoints_path)))
        return self._cache[entry.keypoints_path]

    def sign_entries(self) -> list[SignEntry]:
        return list(self._signs.values())

    def gloss_keys(self) -> list[str]:
        return sorted(self._signs, key=lambda k: (-len(k.split()), -len(k)))


ALPHABET = Path("alphabet") / "entries.jsonl"  # one signer's A-Z (scripts/lexicon/alphabet/build.py)
NAMES = Path("names") / "entries.jsonl"        # names spelled from that alphabet (scripts/lexicon/alphabet/names.py)


def asl_lexicon(path: Path | None = None) -> ASLLexicon:
    """The ASL lexicon the app uses: lexicon_all.jsonl (scripts/lexicon/asl/merge.py), or lexicon.jsonl before the
    first merge, with the letters of the A-Z alphabet source for fingerspelling."""
    from isharati.config import DATA
    lex_dir = DATA / "asl" / "lexicon"
    if path is None:
        path = lex_dir / "lexicon_all.jsonl"
        path = path if path.exists() else lex_dir / "lexicon.jsonl"
    lex = ASLLexicon.load(path)
    lex.add_letters(Path(path).parent / ALPHABET)
    lex.add_words(Path(path).parent / NAMES)  # fingerspelled names (Sahaba), where no sign exists
    return lex
