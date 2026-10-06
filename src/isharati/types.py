"""Shared data contracts (see spec §3)."""
from dataclasses import dataclass, field

N_JOINTS = 50
FPS = 25


@dataclass(frozen=True)
class SourceCard:
    title: str
    reference: str
    grade: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class Lesson:
    lesson_id: str
    title: str
    text_ar: str
    meaning_simplified: str
    source_card: SourceCard
    level: str            # "A" | "B" | "C" | "D"
    review_status: str    # "approved" | "pending"


@dataclass(frozen=True)
class SignEntry:
    gloss: str
    sign_id: str
    dataset: str
    keypoints_path: str   # relative to the lexicon file's directory
    review_status: str    # "approved" | "pending"
    is_religious: bool
    is_letter: bool = False
    verification_score: float | None = None


@dataclass(frozen=True)
class GlossItem:
    text: str
    oov: bool = False


@dataclass(frozen=True)
class AlignRow:
    word: str
    normalized: str
    method: str           # "exact" | "stem" | "llm" | "fingerspell" | "stopword"
    gloss: str = ""
    sign_id: str = ""
    review_status: str = ""


@dataclass
class GlossResult:
    items: list[GlossItem]
    backend: str
    trace: list[AlignRow] = field(default_factory=list)

    @property
    def glosses(self) -> list[str]:
        return [i.text for i in self.items if not i.oov]

    @property
    def oov(self) -> list[str]:
        return [i.text for i in self.items if i.oov]


@dataclass(frozen=True)
class Segment:
    label: str
    start: int
    end: int              # exclusive
    kind: str             # "sign" | "fingerspell" | "transition"


@dataclass
class BTResult:
    recognized: list[str] = field(default_factory=list)
    gloss_acc: float = 0.0
    wer: float = 1.0
