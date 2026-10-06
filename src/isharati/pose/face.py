"""The signer's face for an assembled sentence, alongside the [T,50,3] pose:

  blend   (T,52) blendshape scores (ARKit names): drive an avatar's face
  points  (T,160,3) face contour landmarks (lips, eyes, irises, brows, nose, face oval; face_contours.json) in the
          pose's own frame: drawn by the skeleton view

Each sign's face comes from data/face/<lexicon>/<sign_id>.npz (scripts/face/attach.py: the same frames as the
sign's clip). Points get the same height rescaling that proportions.fix gives the clip at load. In the stitched
sentence a sign's face sits on the sign's frames, transitions blend from one sign's last face to the next one's first,
and the rest pose at the start and end relaxes to a neutral expression (blend 0) holding the nearest face shape.
Signs with no face recorded yet are NaN: the avatar keeps the expression it had, the skeleton draws no face.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from isharati.config import DATA
from isharati.pose import proportions

FACE = DATA / "face"
LEXICON_DIR = {"en": "asl", "ar": "ar", "tr": "tr", "ur": "ur"}  # sign language lexicon per question language
CONTOURS = json.loads((Path(__file__).with_name("face_contours.json")).read_text(encoding="utf-8"))
POINTS = np.array(CONTOURS["points"])


@lru_cache(maxsize=4096)
def load(lang: str, sign_id: str):
    """(blend [T,52], contour points [T,160,3], names) for one sign, or None if its face has not been extracted."""
    path = FACE / LEXICON_DIR.get(lang, lang) / f"{sign_id}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    face = z["face"]  # the full 478-point mesh, or the 160 contour points already (the published dataset)
    pts = (face if face.shape[1] == len(POINTS) else face[:, POINTS]).astype(np.float32)
    if "ratio" in z.files:  # as proportions.fix scales the clip: heights about the neck (the origin) by REF / ratio
        r = float(z["ratio"])
        if np.isfinite(r) and r > 0:
            pts[..., 1] *= proportions.REF / r
    return z["blend"].astype(np.float32), pts, [str(n) for n in z["blend_names"]]


def _fill_gaps(rows, segments, length):
    """Transitions: linear blend where both neighbours are known."""
    for s in segments:
        if s.kind == "transition" and 0 < s.start and s.end < length:
            a, b = rows[s.start - 1], rows[s.end]
            if np.isfinite(a).all() and np.isfinite(b).all():
                w = np.linspace(0, 1, s.end - s.start + 2, dtype=np.float32)[1:-1]
                w = w.reshape(-1, *([1] * a.ndim))
                rows[s.start:s.end] = (1 - w) * a + w * b


def track(segments, length: int, lang: str, lexicon, loader=None):
    """{"blend", "points", "names"} for an assembled pose (NaN where unknown), or None if no sign has a face."""
    blend = np.full((length, 52), np.nan, np.float32)
    points = np.full((length, len(POINTS), 3), np.nan, np.float32)
    names = None
    for s in segments:
        if s.kind != "sign":
            continue
        e = lexicon.lookup(s.label)
        f = (loader or load)(lang, e.sign_id) if e is not None else None
        if f is None:
            continue
        b, p, names = f
        n = min(len(b), s.end - s.start)
        blend[s.start:s.start + n], points[s.start:s.start + n] = b[:n], p[:n]
    if names is None:
        return None
    signs = [s for s in segments if s.kind != "transition"]
    first, last = signs[0].start, signs[-1].end
    blend[:first], blend[last:] = 0.0, 0.0                       # rest pose: neutral expression
    known = np.flatnonzero(np.isfinite(points[:, 0, 0]))
    if known.size:                                               # rest pose: hold the nearest face shape
        points[:first] = points[known[0]]
        points[last:] = points[known[-1]]
    _fill_gaps(blend, segments, length)
    _fill_gaps(points, segments, length)
    return {"blend": blend, "points": points, "names": names}


def as_json(rows, digits=3):
    """Rows for JSON: NaN -> None. Works for (T,K) and (T,K,D) arrays."""
    a = np.asarray(rows, dtype=np.float64).round(digits)
    out = a.astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()
