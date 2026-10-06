"""Automatic QA of the published sign datasets (the hub_build folders the publishers write): every sample of the
hadith dataset and of the Qur'an datasets is checked for body shape, motion, hands, face, labels and gloss, and a
contact sheet of skeleton frames is drawn per source for an eye check.

  py scripts/eval/qa_sign_datasets.py [--build D:\\islam\\hub_build] [--out results/qa_sign]

Writes <out>/qa.json (every flag, per sample), <out>/summary.md, and <out>/sheet_<dataset>_<source>.png.
Thresholds are human-body ranges in shoulder widths (the Isharati pose unit).
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATASETS = ["hadith-sign", "quran-sign-kfc", "tafsir-mukhtasar-sign", "quran-sign-curriculum", "quran-sign-tebyan",
            "quran-sign-diyanet-tid"]
NOSE, NECK, R_SH, R_EL, R_WR, L_SH, L_EL, L_WR = range(8)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
# human ranges in shoulder widths (median over the clip): adults ~0.7-0.9 upper arm, ~0.6-0.8 forearm
UPPER = (0.5, 1.1)
FORE = (0.4, 1.05)
JUMP = 0.6          # a joint moving more than this between two frames at 25 fps (15 shoulder widths a second)
ISNAD = re.compile(r"(^|\s)(عن|رضي|حدثنا|أخبرنا|سمعت)(\s|$)|حديث\s+(حسن|صحيح)|رواه")


def seg(p, a, b):
    return np.linalg.norm(p[:, a, :2] - p[:, b, :2], axis=-1)


def body_checks(p):
    """Flags for one [T,50,3] pose: proportions, jumps, hand attachment, frozen frames."""
    flags, m = [], {}
    sw = seg(p, R_SH, L_SH)
    med = float(np.nanmedian(sw))
    if not np.isfinite(med) or med <= 0:
        return ["no_shoulders"], m
    ua = np.nanmedian(np.concatenate([seg(p, R_SH, R_EL), seg(p, L_SH, L_EL)])) / med
    fa = np.nanmedian(np.concatenate([seg(p, R_EL, R_WR), seg(p, L_EL, L_WR)])) / med
    m.update(upper_arm=round(float(ua), 2), forearm=round(float(fa), 2))
    if not UPPER[0] <= ua <= UPPER[1]:
        flags.append("upper_arm_out_of_range")
    if not FORE[0] <= fa <= FORE[1]:
        flags.append("forearm_out_of_range")
    flicker = float(np.nanstd(sw) / med)
    m["shoulder_flicker"] = round(flicker, 3)
    if flicker > 0.25:
        flags.append("scale_flicker")
    if len(p) > 1:
        v = np.linalg.norm(np.diff(p[:, :8, :2], axis=0), axis=-1) / med
        jumps = int(np.nansum(v > JUMP))
        m["jumps"] = jumps
        if jumps > max(3, 0.01 * len(p)):
            flags.append("joint_jumps")
        still = np.nanmax(np.abs(np.diff(p[:, :, :2], axis=0)), axis=(1, 2)) < 1e-4
        m["frozen_share"] = round(float(still.mean()), 3)
        if still.mean() > 0.3:
            flags.append("frozen")
    for hand, wrist, name in ((slice(8, 29), L_WR, "left"), (slice(29, 50), R_WR, "right")):
        d = np.linalg.norm(p[:, hand][:, 0, :2] - p[:, wrist, :2], axis=-1) / med
        if np.nanmedian(d) > 0.35:
            flags.append(f"{name}_hand_detached")
        m[f"{name}_hand_to_wrist"] = round(float(np.nanmedian(d)), 2)
    return flags, m


def label_checks(r):
    flags = []
    hs = r.get("hadiths")
    if hs is None:  # Qur'an rows carry ayahs, not hadiths
        if not r.get("ayahs") and r.get("type") == "recitation":
            flags.append("recitation_without_ayah")
        return flags
    if r.get("label_source") == "none" or not hs:
        return ["unlabelled"]
    h = hs[0]
    tref = r.get("title_ref") or {}
    if tref.get("book") == "nawawi" and h["collection"] == "nawawi" and not (
            tref["from"] <= float(h["number"]) <= tref["to"]):
        flags.append("title_number_mismatch")
    if h.get("title_conflict"):
        flags.append("title_conflict")
    if h.get("via") == "speech" and (h.get("matched_words") or 0) < 10:
        flags.append("weak_text_match")
    g = h.get("gloss") or {}
    if g and r["sign_language"] not in g:
        flags.append("gloss_wrong_language")
    for items in g.values():
        if ISNAD.search(" ".join(i["text"] for i in items)):
            flags.append("gloss_has_isnad")
        if items and sum(i["oov"] for i in items) / len(items) > 0.6:
            flags.append("gloss_mostly_oov")
    return flags


def draw_sheet(samples, path, cols=8):
    """One mid-clip skeleton frame per sample, with its id and flags: an eye check of shapes the numbers miss."""
    import cv2
    cell = 220
    rows = (len(samples) + cols - 1) // cols
    img = np.full((rows * cell, cols * cell, 3), 12, np.uint8)
    for k, (sid, p, flags) in enumerate(samples):
        x0, y0 = (k % cols) * cell, (k // cols) * cell
        f = p[len(p) // 2]
        if not np.isfinite(f[:8]).all():
            continue
        xy = lambda q: (int(x0 + cell / 2 + q[0] * 45), int(y0 + 50 + q[1] * 45))  # noqa: E731
        col = (60, 60, 255) if flags else (140, 255, 170)
        for a, b in EDGES:
            cv2.line(img, xy(f[a]), xy(f[b]), col, 2)
        for q in f[8:50]:
            if np.isfinite(q).all():
                cv2.circle(img, xy(q), 1, (255, 200, 120), -1)
        cv2.putText(img, sid[:16], (x0 + 4, y0 + cell - 22), 0, 0.38, (200, 200, 200), 1)
        if flags:
            cv2.putText(img, ",".join(flags)[:30], (x0 + 4, y0 + cell - 8), 0, 0.33, (80, 80, 255), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=r"D:\islam\hub_build")
    ap.add_argument("--out", default="results/qa_sign")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report, lines = {}, ["# Sign dataset QA", ""]
    for ds in DATASETS:
        root = Path(a.build) / ds
        if not (root / "manifest.jsonl").exists():
            continue
        rows = [json.loads(l) for l in (root / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
        packs, per_source, flags_all = {}, defaultdict(list), Counter()
        metrics = defaultdict(lambda: defaultdict(list))
        for r in rows:
            src = r.get("source") or ds
            f = label_checks(r)
            pf = r.get("isharati_file") or (f"isharati/{int(r['surah']):03d}.npz" if "surah" in r else None)
            p = None
            if r.get("isharati") and pf:
                if pf not in packs:
                    packs[pf] = np.load(root / pf)
                if r["id"] in packs[pf].files:
                    p = packs[pf][r["id"]].astype(np.float32)
            if p is not None and len(p):
                bf, m = body_checks(p)
                f += bf
                for k, v in m.items():
                    metrics[src][k].append(v)
                if len(per_source[src]) < 48:
                    per_source[src].append((r["id"], p, bf))
            if "face_file" in r and r.get("isharati") and not r.get("face"):
                f.append("no_face")
            report[f"{ds}/{r['id']}"] = f
            flags_all.update(f)
        lines += [f"## {ds}: {len(rows)} samples", ""]
        for src, ms in sorted(metrics.items()):
            lines.append(f"- {src}: upper arm {np.median(ms['upper_arm']):.2f}, forearm {np.median(ms['forearm']):.2f}, "
                         f"flicker {np.median(ms['shoulder_flicker']):.3f}, samples {len(ms['upper_arm'])}")
        lines.append("")
        lines += [f"- `{k}`: {v}" for k, v in flags_all.most_common()] or ["- no flags"]
        lines.append("")
        for src, samples in per_source.items():
            draw_sheet(samples, out / f"sheet_{ds}_{src}.png")
    (out / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
