"""End-to-end check of the landing page against a running app, plus the screenshots the README and report use.

  py scripts/web/e2e.py [--base http://127.0.0.1:8765] [--live]    # --live also asks a Turkish and an Urdu question
"""
import shutil
import sys
import tempfile
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "paper" / "figures"
TMP = Path(tempfile.mkdtemp(prefix="isharati_e2e_"))  # screenshots land here; copied to FIG only if every check passes
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8765"
CASES = [("en", "What are the pillars of Islam?", "ui_en"), ("ar", "ما هو الإسلام؟", "ui_ar")]
if "--live" in sys.argv:
    CASES += [("tr", "İslam'ın şartları nelerdir?", None), ("ur", "اسلام کے ارکان کیا ہیں؟", None)]
READY = "document.querySelector('.result canvas') && !document.querySelector('.status.err')"


def shot_section(page, name):
    """Screenshot #try without the sticky nav bar drawn over it."""
    page.evaluate("document.querySelector('nav.nav').style.visibility = 'hidden'")
    page.locator("#try").screenshot(path=str(TMP / f"{name}.png"))
    page.evaluate("document.querySelector('nav.nav').style.visibility = ''")


def main():
    failures = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page.set_default_timeout(120_000)  # software WebGL keeps the browser busy while an avatar plays
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{BASE}/?ui=en", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        for sid in ("try", "how", "lexicon", "avatars", "sources"):
            if not page.query_selector(f"#{sid}"):
                failures.append(f"missing section #{sid}")
        page.screenshot(path=str(TMP / "landing.png"))
        playing = page.evaluate("Promise.all([...document.querySelectorAll('.reel video')].map(v => v.play().then(() => v.readyState)))")
        if not playing or min(playing) < 2:
            failures.append(f"hero clips not playing: {playing}")
        for lang, q, shot in CASES:
            page.goto("about:blank")
            page.goto(f"{BASE}/?ui={lang}&lang={lang}&q={urllib.parse.quote(q)}#try", wait_until="domcontentloaded")
            try:
                page.wait_for_function(READY, timeout=240_000)
            except Exception:
                failures.append(f"{lang}: no signed result ({page.inner_text('#try')[:200]!r})")
                continue
            page.wait_for_timeout(4000)  # avatar loads and plays into a sign
            if shot:
                shot_section(page, shot)
        page.goto(f"{BASE}/?ui=en&lang=en&q={urllib.parse.quote(CASES[0][1])}#try", wait_until="domcontentloaded")
        page.wait_for_function(READY)
        pickers = page.query_selector_all(".picker")
        if len(pickers) < 5:
            failures.append(f"only {len(pickers)} realistic avatars in the picker")
        else:
            pickers[3].click()
            page.wait_for_timeout(4000)
        # the demo switch: cartoon (with an outfit change) and back to realistic, mid-answer
        page.click(".style-switch > .seg button:nth-child(1)")
        page.wait_for_timeout(4000)
        page.click("text=Shemagh and thobe")
        page.wait_for_timeout(1500)
        shot_section(page, "ui_cartoon")
        page.click(".style-switch > .seg button:nth-child(2)")
        page.wait_for_timeout(4000)
        if not page.query_selector(".style-switch .picker.on"):
            failures.append("switching back to realistic did not restore an avatar")
        # Studio: the user's own text, signed without an answer being generated
        page.goto(f"{BASE}/?ui=en#studio", wait_until="domcontentloaded")
        page.wait_for_selector("textarea", timeout=20_000)
        page.fill("textarea", "Prayer is the pillar of the religion.")
        page.click("form.ask button")
        try:
            page.wait_for_function(READY, timeout=240_000)
            if page.query_selector(".sources a"):
                failures.append("text to sign showed sources")
            page.wait_for_timeout(4000)
            shot_section(page, "ui_studio")
        except Exception:
            failures.append(f"studio: no signed result ({page.inner_text('#try')[:200]!r})")
        if errors:
            failures.append(f"page errors: {errors}")
        b.close()
    if not failures:
        for f in TMP.glob("*.png"):
            shutil.copy(f, FIG / f.name)
    shutil.rmtree(TMP, ignore_errors=True)
    print("\n".join(failures) or "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
