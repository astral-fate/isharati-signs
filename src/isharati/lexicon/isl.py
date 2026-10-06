"""Indian Sign Language lexicon for Urdu answers: CISLR (AFL-3.0) and, while their flags are on, WSLP
(ISHARATI_URDU_WSLP; CC BY-NC-ND 4.0) and the ISLRTC dictionary words and A-Z manual alphabet (ISHARATI_URDU_ISLRTC;
Govt of India YouTube channel, no open licence; scripts/lexicon/isl/islrtc.py, isl_alphabet.py). Keep both off for
any public deployment. Glosses are English; an earlier source wins a tie (CISLR, then WSLP, then ISLRTC), so turning a
flag off only removes signs. The alphabet's rows are letters (is_letter), which the lexicon uses for fingerspelling;
rows marked "variant_of" (more recordings of a sign) are skipped.

Pakistan Sign Language (same Indo-Pakistani family, the sign language of most Urdu speakers) joins while
ISHARATI_URDU_PSL is on: the FESF / Deaf Reach PSL dictionary (psl.org.pk; (c) FESF, no open licence, research and
demo only; scripts/lexicon/isl/psl.py). Its rows carry English glosses plus the Urdu word ("urdu") and
"sign_language": "PSL"; it loads last, so a PSL sign only fills glosses no ISL source has."""
import json
import os
from dataclasses import fields
from pathlib import Path

from isharati.config import DATA
from isharati.lexicon.asl import ASLLexicon
from isharati.types import SignEntry

ISL_ROOT = DATA / "lexicon_isl"
SOURCES = ("cislr", "wslp", "islrtc", "alphabet", "psl")


def _flag(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "no", "off", "")


def wslp_enabled() -> bool:
    return _flag("ISHARATI_URDU_WSLP")


def islrtc_enabled() -> bool:
    return _flag("ISHARATI_URDU_ISLRTC")


def psl_enabled() -> bool:
    return _flag("ISHARATI_URDU_PSL")


def entries_files(root: Path | None = None, wslp: bool | None = None, islrtc: bool | None = None,
                  psl: bool | None = None) -> list[Path]:
    root = Path(root or ISL_ROOT)
    on = {"cislr": True, "wslp": wslp_enabled() if wslp is None else wslp,
          "islrtc": islrtc_enabled() if islrtc is None else islrtc, "psl": psl_enabled() if psl is None else psl}
    on["alphabet"] = on["islrtc"]
    return [root / s / "entries.jsonl" for s in SOURCES if on[s] and (root / s / "entries.jsonl").exists()]


def isl_lexicon(root: Path | None = None, wslp: bool | None = None, islrtc: bool | None = None,
                psl: bool | None = None) -> ASLLexicon:
    root = Path(root or ISL_ROOT)
    rows = [json.loads(l) for f in entries_files(root, wslp, islrtc, psl)
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    # a row with "variant_of" is another recording of a sign already listed: kept in the files, not used here
    keep = {f.name for f in fields(SignEntry)}
    return ASLLexicon([SignEntry(**{k: v for k, v in r.items() if k in keep}) for r in rows if not r.get("variant_of")],
                      root)  # keypoints_path is relative to lexicon_isl/
