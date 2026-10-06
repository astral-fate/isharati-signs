import json

import numpy as np
import pytest

SIGNS = {"وضوء": "r1", "وجه": "r2", "غسل": "r3", "يد": "r4", "صلاة": "r5", "إسلام": "r6", "صدق": "r7", "سلام": "r8"}
PENDING = {"صلاة"}
RELIGIOUS = {"وضوء", "صلاة", "إسلام"}
LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويةأؤئءإآى")


def _clip(seed: int, T: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(1, 50, 3)).astype(np.float32)
    drift = np.linspace(0, 1, T, dtype=np.float32)[:, None, None] * rng.normal(size=(1, 50, 3)).astype(np.float32)
    return (base + drift).astype(np.float32)


def _row(gloss, sid, religious, letter, status):
    return {"gloss": gloss, "sign_id": sid, "dataset": "test", "keypoints_path": f"signs/{sid}.npy",
            "review_status": status, "is_religious": religious, "is_letter": letter}


@pytest.fixture
def lexicon_dir(tmp_path):
    (tmp_path / "signs").mkdir()
    rows = []
    for i, (g, sid) in enumerate(SIGNS.items()):
        np.save(tmp_path / "signs" / f"{sid}.npy", _clip(i))
        rows.append(_row(g, sid, g in RELIGIOUS, False, "pending" if g in PENDING else "approved"))
    for j, ch in enumerate(LETTERS):
        sid = f"L{j}"
        np.save(tmp_path / "signs" / f"{sid}.npy", _clip(100 + j, T=10))
        rows.append(_row(ch, sid, False, True, "approved"))
    (tmp_path / "lexicon.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp_path


@pytest.fixture
def lexicon(lexicon_dir):
    from isharati.lexicon.arabic import Lexicon
    return Lexicon.load(lexicon_dir / "lexicon.jsonl")
