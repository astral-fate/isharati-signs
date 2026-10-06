"""Proof that the alphabets work as letters: fingerspell test names with each language's Lexicon and
isharati.pose.fingerspell (the Arabic fingerspelled.py steps). Static letters give their HOLD-frame hold, the motion
letters of alphabet/sources.json (ASL J Z; TİD Ç Ğ İ J Ö Ş Ü) their whole movement; one signer, so the right hand
throughout and nothing mirrored. Writes F:/alphabets/<lang>/test_words/<word>.npy, <word>_face.npz and a skeleton
strip <word>.jpg (one frame per letter); nothing goes into the lexicon.
  PYTHONPATH=src python scripts/lexicon/alphabet/fingerspell_test.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
from isharati.lexicon.asl import asl_lexicon  # noqa: E402
from isharati.lexicon.turkish import tid_lexicon  # noqa: E402
from isharati.pose import fingerspell  # noqa: E402
from isharati.pose.keypoints import LH, RH, L_WR, R_WR  # noqa: E402

WORDS = {"asl": (asl_lexicon, DATA / "asl" / "lexicon", DATA / "face" / "asl", ["ALI", "AISHA"]),
         "tid": (tid_lexicon, DATA / "lexicon_tr", DATA / "face" / "tr", ["ALİ", "AYŞE"])}
BONES = [(0, 1), (2, 3), (3, 4), (5, 6), (6, 7), (2, 5), (1, 2), (1, 5)]
FINGERS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
           (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def draw(frame, label, size=220):
    img = np.full((size, size, 3), 255, np.uint8)
    p = frame[:, :2]
    to = lambda q: (int(size / 2 + q[0] * size * 0.32), int(size * 0.28 + q[1] * size * 0.32))
    for a, b in BONES:
        cv2.line(img, to(p[a]), to(p[b]), (120, 120, 120), 2)
    for sl, wr, col in ((RH, R_WR, (40, 40, 220)), (LH, L_WR, (220, 120, 40))):
        h = p[sl]
        for a, b in FINGERS:
            cv2.line(img, to(h[a]), to(h[b]), col, 1)
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img


def main():
    for lang, (load, lex_dir, faces, words) in WORDS.items():
        lex = load()
        src = json.loads((lex_dir / "alphabet" / "sources.json").read_text(encoding="utf-8"))
        motion = {r["letter"] for r in src["letters"] if r["kind"] == "motion"}
        out = Path(r"F:\alphabets") / lang / "test_words"
        out.mkdir(parents=True, exist_ok=True)
        for word in words:
            entries = lex.fingerspell(word)
            clip, face, hand, hands, lengths = fingerspell.build(entries, lex.keypoints, faces, motion=motion,
                                                                 hand=src["signer_hand"])
            np.save(out / f"{word}.npy", clip)
            if face is not None:
                fingerspell.save_face(out / f"{word}_face.npz", face, len(clip), f"fingerspelled from {src['url']}")
            t, tiles = 0, []
            for e, n in zip(entries, lengths):
                tiles.append(draw(clip[t + n // 2], f"{e.gloss} ({n}f)"))
                t += n + fingerspell.TRANSITION
            cv2.imwrite(str(out / f"{word}.jpg"), np.hstack(tiles))
            print(f"{lang} {word}: {' '.join(e.gloss for e in entries)} ({' '.join(e.sign_id for e in entries)}) -> "
                  f"{len(clip)} frames; frames per letter {[f'{e.gloss}:{n}' for e, n in zip(entries, lengths)]} + "
                  f"{fingerspell.TRANSITION}-frame transitions; hand {hand} (detected per letter: {hands}), "
                  f"face {'yes' if face is not None else 'no'}")


if __name__ == "__main__":
    main()
