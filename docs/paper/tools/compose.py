"""Joins the avatar renders into figure panels on white (run after shoot.py): .venv/Scripts/python compose.py"""
from pathlib import Path
from PIL import Image

FIG = Path(__file__).resolve().parents[1] / "figures"


def flat(name, crop_bottom=0.86):
    im = Image.open(FIG / name).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    bg.alpha_composite(im)
    w, h = bg.size
    return bg.convert("RGB").crop((int(w * 0.08), 0, int(w * 0.92), int(h * crop_bottom)))


def row(names, out, gap=12):
    ims = [flat(n) for n in names]
    w = sum(i.width for i in ims) + gap * (len(ims) - 1)
    m = Image.new("RGB", (w, ims[0].height), "white")
    x = 0
    for i in ims:
        m.paste(i, (x, 0)); x += i.width + gap
    m.thumbnail((3000, 3000))
    m.save(FIG / out, quality=92)


row([f"outfit_{p}.png" for p in ("original", "hijab", "shemagh")], "avatar_outfits.jpg")
row([f"strip_ar_{k}.png" for k in range(6)], "avatar_strip_ar.jpg")
row([f"strip_en_{k}.png" for k in range(6)], "avatar_strip_en.jpg")
print("ok")
