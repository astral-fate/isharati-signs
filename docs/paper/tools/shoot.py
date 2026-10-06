"""Screenshots of the avatar for the paper: python shoot.py <out.png> <query> [<out.png> <query> ...] (serve_fig.py must run)."""
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = b.new_page(viewport={"width": 600, "height": 760}, device_scale_factor=2)
    for out, q in zip(sys.argv[1::2], sys.argv[2::2]):
        page.goto(f"http://127.0.0.1:8790/fig.html?{q}")
        page.wait_for_function("document.title.length > 0", timeout=120000)
        if page.title() != "done":
            print(out, page.title(), page.inner_text("body")[:500]); continue
        import base64
        data = page.evaluate("document.getElementById('c').toDataURL('image/png')")
        open(out, "wb").write(base64.b64decode(data.split(",", 1)[1]))
        print(out, "ok")
    b.close()
