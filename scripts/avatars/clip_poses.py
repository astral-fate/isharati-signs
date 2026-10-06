"""Short stitched phrases for the landing page's avatar clips and hero figure: each avatar signs a different phrase
in a different sign language. Writes web/public/clips/poses/<id>.json and web/src/assets/hero-pose.json.

  py scripts/avatars/clip_poses.py
"""
import json
import os
import sys
from pathlib import Path

# Committed clips must never contain poses without an open licence: WSLP (CC BY-NC-ND), ISLRTC (not stated) and the
# FESF PSL dictionary (all rights reserved). Force the Urdu lexicon to CISLR only, before it loads.
os.environ["ISHARATI_URDU_WSLP"] = "0"
os.environ["ISHARATI_URDU_ISLRTC"] = "0"
os.environ["ISHARATI_URDU_PSL"] = "0"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import load_lexicon  # noqa: E402
from isharati.pose.poser import MissingAwarePoser  # noqa: E402
from isharati.types import FPS, GlossItem, GlossResult  # noqa: E402

OUT = ROOT / "web" / "public" / "clips" / "poses"
HERO = ROOT / "web" / "src" / "assets" / "hero-pose.json"
# avatar, language, candidate glosses in order of preference; the first three the lexicon has are used (fewer if the phrase would run past 6 s)
CLIPS = [
    ("rocketbox_female_06.vrm", "en", ["ALLAH", "PRAYER"]),
    ("rocketbox_female_10.vrm", "ar", ["الله", "الصلاة"]),
    ("rocketbox_male_15.vrm", "tr", ["merhaba", "barış"]),
    ("rocketbox_male_19.vrm", "ur", ["ALLAH", "MOSQUE"]),
    ("rocketbox_male_21.vrm", "ar", ["الزكاة", "فقير", "مال"]),
    ("avatar.vrm", "en", ["PEACE", "HELLO"]),
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
