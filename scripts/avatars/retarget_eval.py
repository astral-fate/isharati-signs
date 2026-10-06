"""How faithfully the 3D avatar reproduces a sign's keypoints: placement relative to the face and body, and palm facing.

Each sign is played on the avatar in headless Chromium (the /record page), and after every frame the avatar's bone
positions are read back (SignAvatar.probe) and compared with the keypoints that drove it, per hand:

  height    fingertip height above the neck, in units of the nose's height above the neck (0 = neck level, 1 = nose
            level). Signer: index-finger DIP joint (21-point hand, 7) vs nose and shoulder midpoint; avatar: the
            IndexDistal bone vs the eyes lowered by 0.1 shoulder widths and the shoulder-joint midpoint.
  side      fingertip offset from the midline, in shoulder widths
  contact   closest frontal-plane approach of the fingertip to the nose over the sign, in shoulder widths (signs made at
            the mouth or chin are near 0.1-0.3; a hand left at the neck or chest is not)
  palm      angle between the palm normals (wrist->middle knuckle x index->little knuckle, mirrored for the right hand),
            in degrees: 0 = facing the same way, 180 = flipped

Only frames where the hand is up (signer fingertip above the lower chest) are scored; a resting hand says little.
Output: results/retarget_eval_<tag>.{json,md}
  PYTHONPATH=src py -m isharati.app.server                # port 8765 (or --base)
  py scripts/avatars/retarget_eval.py <tag> [--base URL] [--model avatar.vrm]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.pipeline import load_lexicon  # noqa: E402
from isharati.types import FPS  # noqa: E402

NOSE, NECK, R_SH, L_SH = 0, 1, 2, 5
HANDS = {"left": 8, "right": 29}
# (language, gloss, where it is made)
SIGNS = [("en", "FAST", "mouth"), ("en", "RAMADAN", "mouth"), ("en", "EAT1", "mouth"), ("en", "MOTHER", "chin"),
         ("en", "FATHER", "forehead"), ("en", "THANK YOU", "chin"), ("en", "GOD", "above head"),
         ("en", "PRAYER", "chest"), ("en", "HAPPY", "chest"), ("en", "SALAH", "chest"), ("en", "QURAN", "chest"),
         ("ar", "صوم", "mouth"), ("ar", "رمضان", "face"), ("ar", "الصلاة", "chest"), ("ar", "الله", "above head")]
OUT = ROOT / "results"
PROBES = ROOT / "web" / "dist" / "clips" / "poses"  # build output, served at /clips/poses/; probe files are removed after


def three(p):
    """Image axes (x right, y down, z away) -> three.js axes, as avatar.js does."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 1], -p[..., 2]], -1)


def palm_normal(wrist, middle, index, little, side):
    a, b = middle - wrist, index - little
    n = np.cross(a, b) if side == "left" else np.cross(b, a)
    return n / (np.linalg.norm(n) + 1e-9)


def signer_measures(f, side):
    k = three(f)
    h = HANDS[side]
    neck, nose = (k[R_SH] + k[L_SH]) / 2, k[NOSE]
    sw = np.linalg.norm(k[R_SH] - k[L_SH])
    tip = k[h + 7]
    return {"height": (tip[1] - neck[1]) / (nose[1] - neck[1]), "side": (tip[0] - neck[0]) / sw,
            "contact": np.linalg.norm((tip - nose)[:2]) / sw,
            "palm": palm_normal(k[h], k[h + 9], k[h + 5], k[h + 17], side)}


def avatar_measures(b, side):
    g = {n: np.asarray(v, float) for n, v in b.items()  # bone positions only (the probe also reports face, reach)
         if isinstance(v, list) and len(v) == 3}
    ls, rs = g["leftUpperArm"], g["rightUpperArm"]
    neck, sw = (ls + rs) / 2, np.linalg.norm(ls - rs)
    eyes = (g["leftEye"] + g["rightEye"]) / 2 if "leftEye" in g and "rightEye" in g else g["head"] + [0, 0.25 * sw, 0]
    nose = eyes - [0, 0.1 * sw, 0]
    tip = g[f"{side}IndexDistal"]
    return {"height": (tip[1] - neck[1]) / (nose[1] - neck[1]), "side": (tip[0] - neck[0]) / sw,
            "contact": np.linalg.norm((tip - nose)[:2]) / sw,
            "palm": palm_normal(g[f"{side}Hand"], g[f"{side}MiddleProximal"], g[f"{side}IndexProximal"],
                                g[f"{side}LittleProximal"], side)}


def evaluate(page, base, model, lang, gloss, lexicons):
    lex = lexicons.setdefault(lang, load_lexicon(lang))
    e = lex.lookup(gloss)
    if e is None:
        return None
    clip = lex.keypoints(e)
    probe = PROBES / f"_retarget_probe_{e.sign_id}.json"  # one file per sign: a reused URL could come from the cache
    probe.write_text(json.dumps({"fps": FPS, "frames": np.round(clip, 4).tolist()}), encoding="utf-8")
    page.goto(f"{base}/record?model={model}&pose=/clips/poses/{probe.name}")
    page.wait_for_function("window.__ready || window.__error", timeout=120000)
    if page.evaluate("window.__error"):
        raise RuntimeError(page.evaluate("window.__error"))
    out = {}
    for side in HANDS:
        rows = []
        for i in range(len(clip)):
            s = signer_measures(clip[i], side)
            if s["height"] < -1.2:  # hand down by the waist: not signing with this hand in this frame
                continue
            a = avatar_measures(page.evaluate(f"window.__probe({i})"), side)
            rows.append((s, a))
        if len(rows) < 3:
            continue
        out[side] = {
            "frames": len(rows),
            "height_signer": round(float(np.median([s["height"] for s, _ in rows])), 2),
            "height_avatar": round(float(np.median([a["height"] for _, a in rows])), 2),
            "height_err": round(float(np.median([abs(s["height"] - a["height"]) for s, a in rows])), 2),
            "side_err": round(float(np.median([abs(s["side"] - a["side"]) for s, a in rows])), 2),
            "contact_signer": round(float(min(s["contact"] for s, _ in rows)), 2),
            "contact_avatar": round(float(min(a["contact"] for _, a in rows)), 2),
            "palm_deg": round(float(np.median([np.degrees(np.arccos(np.clip(s["palm"] @ a["palm"], -1, 1)))
                                              for s, a in rows])), 1),
            "palm_flipped": round(float(np.mean([s["palm"] @ a["palm"] < 0 for s, a in rows])), 2),
        }
    return {"lang": lang, "gloss": gloss, "where": WHERE[(lang, gloss)], "dataset": e.dataset, "hands": out}


WHERE = {(l, g): w for l, g, w in SIGNS}


def main():
    from playwright.sync_api import sync_playwright
    tag = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "current"
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "avatar.vrm"
    results, lexicons = [], {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 480, "height": 480})
        for lang, gloss, _ in SIGNS:
            r = evaluate(page, base, model, lang, gloss, lexicons)
            if r:
                results.append(r)
                print(gloss, json.dumps(r["hands"], ensure_ascii=False), flush=True)
        browser.close()
    for f in PROBES.glob("_retarget_probe_*.json"):
        f.unlink()
    hands = [h for r in results for h in r["hands"].values()]
    summary = {"model": model, "signs": len(results), "hand_tracks": len(hands),
               "height_err": round(float(np.median([h["height_err"] for h in hands])), 2),
               "side_err": round(float(np.median([h["side_err"] for h in hands])), 2),
               "palm_deg": round(float(np.median([h["palm_deg"] for h in hands])), 1),
               "palm_flipped": round(float(np.mean([h["palm_flipped"] for h in hands])), 2)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"retarget_eval_{tag}.json").write_text(json.dumps({"summary": summary, "signs": results}, ensure_ascii=False,
                                                               indent=1), encoding="utf-8")
    md = [f"# Avatar retargeting vs keypoints ({tag}, {model})\n",
          f"Median over {summary['hand_tracks']} hand tracks in {summary['signs']} signs: height error "
          f"{summary['height_err']} (nose heights), side error {summary['side_err']} (shoulder widths), palm angle "
          f"{summary['palm_deg']}°, share of frames with the palm flipped {summary['palm_flipped']:.0%}.\n",
          "| sign | where | hand | height signer → avatar | contact signer → avatar | palm ° | flipped |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r['gloss']} ({r['dataset']}) | {r['where']} | {side} | {h['height_signer']} → {h['height_avatar']} | "
           f"{h['contact_signer']} → {h['contact_avatar']} | {h['palm_deg']} | {h['palm_flipped']:.0%} |"
           for r in results for side, h in r["hands"].items()]
    (OUT / f"retarget_eval_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
