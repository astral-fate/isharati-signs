"""Figures for the Isharati paper, from the project's own outputs (run from D:/islam/isharati):
  .venv/Scripts/python docs/paper/tools/make_figures.py
Inputs: docs/paper/data/ask_{ar,en}.json (two live demo runs), data/asl/out/<id>/pose.npy, results/coverage_en.json,
the inventory CSVs (lengths_*.csv) when present. Output: docs/paper/figures/*.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "docs" / "paper"
FIG = PAPER / "figures"
sys.path.insert(0, str(ROOT))
from isharati.keypoints import L_WR, R_WR  # noqa: E402
from isharati.render_skeleton import BODY, HAND  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
TEAL, AMBER, RED, INK, GREY = "#14b8a6", "#f59e0b", "#dc2626", "#1d1d1b", "#9ca3af"


def run(lang):
    d = json.loads((PAPER / "data" / f"ask_{lang}.json").read_text(encoding="utf-8"))
    pose = np.load(ROOT / "data" / "asl" / "out" / d["id"] / "pose.npy")
    return d, pose


def draw(ax, f):
    """One frame [50,3] in image axes (y down)."""
    for a, b in BODY:
        ax.plot([f[a, 0], f[b, 0]], [f[a, 1], f[b, 1]], color=INK, lw=1.6, solid_capstyle="round")
    for base, col in ((8, AMBER), (29, RED)):
        h = f[base:base + 21]
        if np.isfinite(h).all():
            for a, b in HAND:
                ax.plot([h[a, 0], h[b, 0]], [h[a, 1], h[b, 1]], color=col, lw=0.9)
    ax.scatter(f[:8, 0], f[:8, 1], s=6, color=TEAL, zorder=3)
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(2.2, -1.3); ax.set_aspect("equal"); ax.axis("off")


def strip(lang, labels_en=None, n=6):
    d, pose = run(lang)
    segs = [s for s in d["segments"] if s["kind"] == "sign"]
    if labels_en:
        segs = [s for s in segs if s["label"] in labels_en]
    segs = segs[:n]
    fig, axes = plt.subplots(1, len(segs), figsize=(1.55 * len(segs), 2.0), gridspec_kw={"wspace": 0.02})
    for k, (ax, s) in enumerate(zip(axes, segs)):
        t = int(round((s["start_s"] + s["end_s"]) / 2 * 25))
        draw(ax, pose[min(t, len(pose) - 1)])
        ax.set_title(s["label"] if lang == "en" else f"({k + 1})", fontsize=8)
    fig.savefig(FIG / f"skeleton_strip_{lang}.pdf")
    plt.close(fig)
    return segs


def timeline(lang):
    """Wrist height over the whole answer, with the sign segments: shows the concatenation and the transitions."""
    d, pose = run(lang)
    t = np.arange(len(pose)) / 25
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    for k, s in enumerate(x for x in d["segments"] if x["kind"] == "sign"):
        ax.axvspan(s["start_s"], s["end_s"], color=TEAL, alpha=0.12 if k % 2 else 0.22, lw=0)
        ax.text((s["start_s"] + s["end_s"]) / 2, 1.02, s["label"] if lang == "en" else str(k + 1), fontsize=6,
                ha="center", va="bottom", rotation=90 if lang == "en" else 0, transform=ax.get_xaxis_transform())
    ax.plot(t, -pose[:, R_WR, 1], color=RED, lw=0.9, label="right wrist")
    ax.plot(t, -pose[:, L_WR, 1], color=AMBER, lw=1, label="left wrist")
    ax.set_xlabel("time (s)"); ax.set_ylabel("height (shoulder units)")
    ax.legend(frameon=True, framealpha=0.9, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_xlim(0, t[-1])
    fig.savefig(FIG / f"timeline_{lang}.pdf")
    plt.close(fig)


def coverage_curve():
    r = json.loads((ROOT / "results" / "coverage_en.json").read_text(encoding="utf-8"))
    tot = r["all"]["vocabulary_no_names"]
    base = tot["tokens_covered"]
    gains = np.cumsum([n for _, n in r["top_missing"]])
    k = np.arange(0, len(gains) + 1)
    cov = 100 * (base + np.concatenate([[0], gains])) / tot["tokens"]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(k, cov, color=TEAL, lw=1.6)
    for x in (50, 100, 200, 500):
        ax.plot(x, cov[x], "o", color=INK, ms=3)
        ax.annotate(f"{cov[x]:.1f}%", (x, cov[x]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.axhline(cov[0], color=GREY, lw=0.8, ls="--")
    ax.text(len(k) * 0.62, cov[0] + 0.8, f"today: {cov[0]:.1f}%", fontsize=7, color="#555")
    ax.set_xlabel("most frequent uncovered words given a sign")
    ax.set_ylabel("Qur'an + hadith tokens covered (%)")
    fig.savefig(FIG / "coverage_curve_en.pdf")
    plt.close(fig)


def coverage_bars():
    rows = []
    for lang in ("en", "ar"):
        p = ROOT / "results" / f"coverage_{lang}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            for kind in ("quran", "hadith", "all"):
                s = r[kind]["vocabulary_no_names"]
                rows.append((lang, kind, 100 * s["type_coverage"], 100 * s["token_coverage"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    labels = [f"{'ASL' if l == 'en' else 'ArSL'}\n{k}" for l, k, _, _ in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[2] for r in rows], 0.36, color=GREY, label="distinct words")
    ax.bar(x + 0.18, [r[3] for r in rows], 0.36, color=TEAL, label="word occurrences")
    for i, r in enumerate(rows):
        ax.text(i + 0.18, r[3] + 1, f"{r[3]:.0f}", ha="center", fontsize=7)
        ax.text(i - 0.18, r[2] + 1, f"{r[2]:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=7); ax.set_ylabel("covered (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "coverage_bars.pdf")
    plt.close(fig)


def lengths():
    """Sign durations per source dataset (inventory CSVs: dataset,gloss,frames)."""
    import csv
    inv = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not inv:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=False)
    for ax, lang, title in zip(axes, ("en", "ar"), ("ASL lexicon", "ArSL lexicon")):
        p = inv / f"lengths_{lang}.csv"
        if not p.exists():
            continue
        by = {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            by.setdefault(r["dataset"], []).append(int(float(r["frames"])) / 25)
        names = sorted(by, key=lambda n: -len(by[n]))[:8]
        ax.boxplot([by[n] for n in names], vert=False, showfliers=False, widths=0.6,
                   medianprops={"color": RED})
        ax.set_yticks(range(1, len(names) + 1), [f"{n} ({len(by[n])})" for n in names], fontsize=6.5)
        ax.set_xlabel("sign duration (s)"); ax.set_title(title, fontsize=8)
    fig.savefig(FIG / "sign_lengths.pdf")
    plt.close(fig)


def lexicon_bars():
    """Glosses per source in the merged lexicons."""
    from collections import Counter
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.75})
    for ax, path, title in zip(axes, (ROOT / "data/asl/lexicon/lexicon_all.jsonl", ROOT / "data/lexicon_v2/lexicon_qa.jsonl"),
                               ("ASL (English mode)", "ArSL (Arabic mode)")):
        seen, c = set(), Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                g = e["gloss"].rstrip("0123456789") if "asl" in str(path) else e["gloss"]  # ASL numbers its variants (PRAY2)
                if g not in seen:
                    seen.add(g); c[e["dataset"]] += 1
        names = [n for n, _ in c.most_common()]
        ax.barh(range(len(names)), [c[n] for n in names], color=TEAL)
        for i, n in enumerate(names):
            ax.text(c[n], i, f" {c[n]:,}", va="center", fontsize=6.5)
        ax.set_yticks(range(len(names)), names, fontsize=6.5); ax.invert_yaxis()
        ax.set_title(f"{title}: {len(seen):,} glosses", fontsize=8); ax.set_xlabel("glosses")
        ax.set_xlim(0, max(c.values()) * 1.25)
    fig.savefig(FIG / "lexicon_sources.pdf")
    plt.close(fig)


def eda_figures():
    """Coverage of the Qur'an and hadith content words per language, and which lexicon source supplies the signs."""
    r = json.loads((ROOT / "results" / "eda.json").read_text(encoding="utf-8"))
    rows = [("English / ASL", r["en"]), ("Arabic / ArSL", r["ar"]), ("Turkish / TID\n(partial lexicon)", r["tr"]),
            ("Urdu / ISL\n(on English glosses)", r["isl_on_en"])]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    x = np.arange(len(rows))
    for off, kind, col in ((-0.2, "quran", GREY), (0.2, "hadith", TEAL)):
        vals = [100 * v[kind]["content_token_coverage"] for _, v in rows]
        ax.bar(x + off, vals, 0.38, color=col, label="Qur'an" if kind == "quran" else "hadith")
        for i, v in enumerate(vals):
            ax.text(i + off, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, [n for n, _ in rows], fontsize=7); ax.set_ylim(0, 80)
    ax.set_ylabel("content-word occurrences\nwith a sign (%)"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "coverage_languages.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"wspace": 0.9})
    for ax, k, title in zip(axes, ("en", "ar"), ("English corpus, ASL lexicon", "Arabic corpus, ArSL lexicon")):
        a = r[k]["attribution"]
        names = list(a["by_source"])
        vals = [100 * a["by_source"][n] for n in names]
        ax.barh(range(len(names)), vals, color=TEAL)
        for i, n in enumerate(names):
            ax.text(vals[i], i, f" {vals[i]:.1f}% ({a['glosses_used_by_source'].get(n, 0)} signs)", va="center", fontsize=6)
        ax.set_yticks(range(len(names)), names, fontsize=6.5); ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.6); ax.set_xlabel("share of content-word occurrences (%)")
        ax.set_title(title, fontsize=8)
    fig.savefig(FIG / "source_attribution.pdf"); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    ar = strip("ar")
    print("ar strip:", [s["label"] for s in ar])
    strip("en", labels_en=["ISLAM", "FIVE", "ALLAH", "PRAYER", "HAJJ", "RAMADAN"])
    timeline("ar"); timeline("en")
    coverage_curve(); coverage_bars(); lexicon_bars(); lengths(); eda_figures()
    print(sorted(p.name for p in FIG.glob("*.pdf")))
