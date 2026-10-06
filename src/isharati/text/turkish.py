"""Turkish text normalisation: Turkish casing (I -> ı, İ -> i) before lower-casing, letters only."""
import re
import unicodedata

TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def tr_norm(text):
    t = unicodedata.normalize("NFC", text or "").translate(TR_LOWER).lower()
    return re.sub(r"[^\wçğıöşüâîû]+", " ", t).strip()
