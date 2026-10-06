"""Numbers, tables and figures for the chapter on the signed scripture corpora (sections/scripture.tex).

Reads the published builds (D:\\islam\\hub_build: hadith-sign and the five Qur'an datasets), the hadith pipeline state
(gathring data/dataset/hadith_videos) and the local keypoint clips, and writes
  data/scripture_eda.json           every figure quoted in the chapter
  sections/gen_scripture.tex        the generated tables
  figures/hadith_sources.pdf, figures/hadith_collections.pdf, figures/pose_aspect_qa.pdf

  py docs/paper/tools/scripture_eda.py
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BUILD = Path(r"D:\islam\hub_build")
GATHER = Path(r"D:\islam\gathring data")
HADITH = GATHER / "dataset" / "hadith_videos"
sys.path.insert(0, str(ROOT / "scripts" / "hub"))
from publish_quran_sign import despike, frame_aspects, square_units, to_isharati  # noqa: E402

QURAN = [("kfc", "quran-sign-kfc", "kfc_videos", "King Fahd Complex", "ArSL"),
         ("mukhtasar", "tafsir-mukhtasar-sign", "tafsir_videos", "al-Mukhtasar fi Tafsir", "ArSL"),
         ("curriculum", "quran-sign-curriculum", "quran_sign_language_videos", "Al-Kharj curriculum", "ArSL"),
         ("tebyan", "quran-sign-tebyan", "tebyan_videos", "Tebyan", "ArSL"),
         ("diyanet", "quran-sign-diyanet-tid", "diyanet_tid_quran", "Diyanet", "TİD")]
SOURCE_NAMES = {"islamweb_deaf": "Islamweb for Deaf", "insan_nawawi": "Insan, Nawawi's Forty",
                "hafez_nawawi": "M. Hafez, Nawawi's Forty", "deafedu_riyad": "Deaf Educational Ch., Riyad",
                "deafedu_nisa30": "Deaf Educational Ch., 30 for women", "hamza_ahadith": "H. Wadwid, hadith",
                "arsl_misc": "ArSL, single videos", "eray_40hadis": "E. Demir, 40 Hadis (TİD)",
                "tid_misc": "TİD, single videos", "deafedu_umdat": "Deaf Educational Ch., Umdat al-Ahkam",
                "deafedu_nawawi": "Deaf Educational Ch., Nawawi lessons", "hamza_nawawi": "H. Wadwid, Nawawi lessons",
                "amrabbas_nawawi": "A. Abbas, Nawawi lessons", "nusrat_fahid": "Nusrat, hadith lessons"}
COLL = {"nawawi": "Nawawi's Forty", "bukhari": "Bukhari", "muslim": "Muslim", "abudawud": "Abu Dawud",
        "tirmidhi": "Tirmidhi", "nasai": "Nasa'i", "ibnmajah": "Ibn Majah", "malik": "Malik", "qudsi": "Qudsi"}
LABEL = {"speech_match": "voice-over matched", "onscreen_match": "on-screen text matched",
         "translation_match": "translation matched", "title": "title only"}


def rows(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def upper_arm_ratio(p):
    sw = np.linalg.norm(p[:, 2, :2] - p[:, 5, :2], axis=-1)
    ua = np.concatenate([np.linalg.norm(p[:, 2, :2] - p[:, 3, :2], axis=-1), np.linalg.norm(p[:, 5, :2] - p[:, 6, :2], axis=-1)])
    return float(np.nanmedian(ua) / np.nanmedian(sw))


def jumps(p):
    sw = np.nanmedian(np.linalg.norm(p[:, 2, :2] - p[:, 5, :2], axis=-1))
    v = np.linalg.norm(np.diff(p[:, :8, :2], axis=0), axis=-1) / sw
    return int(np.nansum(v > 0.6))


def quran_tables(out):
    tab = []
    for key, repo, folder, name, sl in QURAN:
        m = rows(BUILD / repo / "manifest.jsonl")
        ay = {tuple(a) for r in m for a in r.get("ayahs") or []}
        tab.append({"key": key, "name": name, "sign_language": sl, "segments": len(m),
                    "recitation": sum(r.get("type") == "recitation" for r in m),
                    "tafsir": sum(r.get("type") == "tafsir" for r in m), "ayahs": len(ay),
                    "surahs": len({int(r["surah"]) for r in m}),
                    "hours": round(sum(r["end"] - r["start"] for r in m) / 3600, 2),
                    "with_pose": sum(bool(r.get("isharati")) for r in m)})
    out["quran"] = tab


def hadith_tables(out):
    vids = rows(HADITH / "videos.jsonl")
    states = {}
    for v in vids:
        p = HADITH / "state" / f"{v['video_id']}.json"
        if p.exists():
            states[v["video_id"]] = json.loads(p.read_text(encoding="utf-8"))
    per = defaultdict(lambda: Counter())
    for v in vids:
        s = states.get(v["video_id"], {})
        c = per[v["source"]]
        c["videos"] += 1
        c["seconds"] += v.get("duration") or 0
        c["transcribed"] += bool(s.get("asr_done"))
        c["silent"] += bool(s.get("silent"))
        c["labelled"] += bool(s.get("hadiths"))
        c["with_keypoints"] += bool(s.get("samples"))
    out["hadith_videos"] = {k: dict(c) for k, c in per.items()}
    out["hadith_videos_total"] = {"videos": len(vids), "hours": round(sum(v.get("duration") or 0 for v in vids) / 3600, 1),
                                  "transcribed": sum(bool(states.get(v["video_id"], {}).get("asr_done")) for v in vids),
                                  "silent": sum(bool(states.get(v["video_id"], {}).get("silent")) for v in vids),
                                  "with_keypoints": sum(bool(states.get(v["video_id"], {}).get("samples")) for v in vids)}
    m = rows(BUILD / "hadith-sign" / "manifest.jsonl")
    lab = [r for r in m if r["label_source"] != "none" and r["hadiths"]]
    refs = Counter(r["hadiths"][0]["ref"] for r in lab)
    by_src = defaultdict(lambda: {"samples": 0, "labelled": 0, "seconds": 0.0, "hadith": set(), "sl": ""})
    for r in m:
        b = by_src[r["source"]]
        b["samples"] += 1
        b["seconds"] += r["end"] - r["start"]
        b["sl"] = r["sign_language"]
        if r in lab:
            b["labelled"] += 1
            b["hadith"].add(r["hadiths"][0]["ref"])
    out["hadith_sources"] = {k: {**{x: v[x] for x in ("samples", "labelled", "sl")}, "hours": round(v["seconds"] / 3600, 2),
                                 "hadith": len(v["hadith"])} for k, v in by_src.items()}
    gl = rows(HADITH / "gloss.jsonl")
    oov = defaultdict(list)
    for g in gl:
        if g["gloss"]:
            oov[g["sign_language"]].append(sum(i["oov"] for i in g["gloss"]) / len(g["gloss"]))
    det = {k: round(float(np.mean([r["detection_rate"].get(k, 0) for r in m])), 3)
           for k in ("pose", "left_hand", "right_hand", "face")}
    out["hadith"] = {
        "samples": len(m), "labelled": len(lab), "unlabelled": len(m) - len(lab),
        "distinct_hadith": len(refs), "hours": round(sum(r["end"] - r["start"] for r in m) / 3600, 2),
        "labelled_hours": round(sum(r["end"] - r["start"] for r in lab) / 3600, 2),
        "sign_languages": dict(Counter(r["sign_language"] for r in lab)),
        "label_sources": dict(Counter(r["label_source"].replace("+title", "") for r in lab)),
        "with_title_agreement": sum("+title" in r["label_source"] for r in lab),
        "collections": dict(Counter(r["hadiths"][0]["collection"] for r in lab)),
        "nawawi_covered": len({x for x in refs if x.startswith("nawawi:")}),
        "most_signed": refs.most_common(5),
        "signers_per_hadith": dict(Counter(Counter(r["hadiths"][0]["ref"] for r in lab).values())),
        "detection": det, "with_face": sum(bool(r.get("face")) for r in m),
        "with_expression": sum(bool(r.get("expression")) for r in m),
        "glosses": {k: {"n": len(v), "oov_share_median": round(float(np.median(v)), 3)} for k, v in oov.items()},
    }


def aspect_qa(out, per_dataset=60):
    """Upper arm / shoulder width before and after square units, and joint jumps before and after the spike filter, on
    the local clips (a sample per dataset)."""
    res = {}
    for key, repo, folder, name, sl in QURAN:
        src = GATHER / "dataset" / folder
        m = rows(src / "manifest.jsonl")
        asp = frame_aspects(GATHER / folder) if (GATHER / folder).exists() else {}
        if key == "diyanet":
            asp = frame_aspects(Path(r"F:\tid_quran_videos\diyanet_tid_quran"))
        rng = np.random.default_rng(0)
        pick = [m[i] for i in rng.choice(len(m), min(per_dataset, len(m)), replace=False)]
        res[name] = sample_qa(pick, lambda r: src / "clips" / f"{r['id']}.npz", lambda r, z: asp.get(r.get("video")))
    clips = HADITH / "clips"
    m = [r for r in rows(BUILD / "hadith-sign" / "manifest.jsonl") if (clips / f"{r['id']}.npz").exists()]
    rng = np.random.default_rng(0)
    pick = [m[i] for i in rng.choice(len(m), min(per_dataset * 2, len(m)), replace=False)]
    res["Signed hadith"] = sample_qa(pick, lambda r: clips / f"{r['id']}.npz", lambda r, z: None)
    out["pose_qa"] = res


def sample_qa(pick, path_of, aspect_of):
    before, after, j0, j1 = [], [], [], []
    for r in pick:
        p = path_of(r)
        if not p.exists():
            continue
        z = np.load(p)
        a = aspect_of(r, z)
        if a is None and "meta" in z.files:
            meta = json.loads(str(z["meta"]))
            a = meta["width"] / meta["height"] if meta.get("width") else None
        if a is None:
            continue
        raw = to_isharati(square_units(z, None))
        fixed = to_isharati(square_units(z, a))
        clean = to_isharati(despike(square_units(z, a)))
        if raw is None or fixed is None or clean is None or len(raw) < 10:
            continue
        before.append(upper_arm_ratio(raw.astype(np.float32)))
        after.append(upper_arm_ratio(fixed.astype(np.float32)))
        j0.append(jumps(fixed.astype(np.float32)) / len(fixed) * 100)
        j1.append(jumps(clean.astype(np.float32)) / len(clean) * 100)
    return {"clips": len(before), "upper_arm_before": round(float(np.median(before)), 2),
            "upper_arm_after": round(float(np.median(after)), 2),
            "jumps_per_100_frames_before": round(float(np.mean(j0)), 2),
            "jumps_per_100_frames_after": round(float(np.mean(j1)), 2)}


def avatar_results(out):
    res = ROOT / "results"
    def summary(tag):
        p = res / f"retarget_eval_{tag}.json"
        return json.loads(p.read_text(encoding="utf-8"))["summary"] if p.exists() else None
    out["retarget"] = {"cartoon_ref074": summary("ik_v6_fingers"), "cartoon_ref050": summary("ref050_avatar"),
                       "rocketbox_cap_ref074": summary("elbowfix2_male15"),
                       "rocketbox_ref050": summary("ref050_rocketbox_male_15"),
                       "rocketbox_before_cap": summary("ik_v4_rocketbox_male_15")}
    p = res / "elbow_eval_0381.json"
    out["elbow_wudu"] = json.loads(p.read_text(encoding="utf-8"))["models"] if p.exists() else None


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig_dir = HERE / "figures"
    # 1: samples per source (labelled vs unlabelled)
    src = sorted(out["hadith_sources"].items(), key=lambda kv: kv[1]["samples"])
    names = [SOURCE_NAMES.get(k, k) for k, _ in src]
    lab = [v["labelled"] for _, v in src]
    unl = [v["samples"] - v["labelled"] for _, v in src]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(names, lab, color="#1b7f6b", label="labelled with a hadith")
    ax.barh(names, unl, left=lab, color="#c9d3d1", label="unlabelled")
    ax.set_xlabel("samples")
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "hadith_sources.pdf")
    plt.close(fig)
    # 2: collections of the labels
    col = sorted(out["hadith"]["collections"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.barh([COLL.get(k, k) for k, _ in col], [v for _, v in col], color="#2a5d8f")
    ax.set_xlabel("labelled samples")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "hadith_collections.pdf")
    plt.close(fig)
    # 3: upper arm / shoulder width before and after square units
    qa = out["pose_qa"]
    ds = list(qa)
    x = np.arange(len(ds))
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.bar(x - 0.18, [qa[d]["upper_arm_before"] for d in ds], 0.36, color="#c0504d", label="raw image units")
    ax.bar(x + 0.18, [qa[d]["upper_arm_after"] for d in ds], 0.36, color="#1b7f6b", label="square units")
    ax.axhspan(0.7, 0.9, color="#dddddd", zorder=0, label="adult range")
    ax.set_xticks(x, ds, rotation=18, ha="right")
    ax.set_ylabel("upper arm / shoulder width")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "pose_aspect_qa.pdf")
    plt.close(fig)


def tex(out):
    q = out["quran"]
    L = [r"\begin{table}[ht]", r"\centering", r"\small",
         r"\caption{The five signed Qur'an datasets as published (private Hugging Face datasets). A segment is one ayah's "
         r"recitation or one passage of tafsir; hours are the signed time.}",
         r"\label{tab:quran_sets}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Source & SL & Segments & Recitation & Tafsir & Ayahs & Surahs & Hours \\", r"\midrule"]
    for r in q:
        L.append(f"{r['name']} & {r['sign_language']} & {r['segments']:,} & {r['recitation']:,} & {r['tafsir']:,} & "
                 f"{r['ayahs']:,} & {r['surahs']} & {r['hours']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (HERE / "sections" / "gen_scripture_quran.tex").write_text("\n".join(L), encoding="utf-8")
    L = []
    h = out["hadith_sources"]
    L += [r"\begin{table}[ht]", r"\centering", r"\small",
          r"\caption{Signed hadith samples per source in the published snapshot (6 October 2026). Labelled samples carry a "
          r"hadith verified against its collection; hours are the signed time after trimming.}",
          r"\label{tab:hadith_sources}", r"\begin{tabular}{llrrrr}", r"\toprule",
          r"Source & SL & Samples & Labelled & Distinct hadith & Hours \\", r"\midrule"]
    for k, v in sorted(h.items(), key=lambda kv: -kv[1]["samples"]):
        L.append(f"{SOURCE_NAMES.get(k, k)} & {v['sl']} & {v['samples']} & {v['labelled']} & {v['hadith']} & "
                 f"{v['hours']:.2f} \\\\")
    t = out["hadith"]
    L += [r"\midrule", f"Total & & {t['samples']} & {t['labelled']} & {t['distinct_hadith']} & {t['hours']:.2f} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (HERE / "sections" / "gen_scripture_hadith.tex").write_text("\n".join(L), encoding="utf-8")
    L = []
    qa = out["pose_qa"]
    L += [r"\begin{table}[ht]", r"\centering", r"\small",
          r"\caption{Pose quality before and after the two corrections, on a random sample of clips per dataset. Upper arm "
          r"over shoulder width is about 0.7--0.9 in adults; a jump is a body joint moving more than 0.6 shoulder widths "
          r"between two frames at 25\,fps.}",
          r"\label{tab:pose_qa}", r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"& & \multicolumn{2}{c}{Upper arm / shoulder} & \multicolumn{2}{c}{Jumps per 100 frames} \\",
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
          r"Dataset & Clips & raw units & square units & before filter & after filter \\", r"\midrule"]
    for d, v in qa.items():
        L.append(f"{d} & {v['clips']} & {v['upper_arm_before']:.2f} & {v['upper_arm_after']:.2f} & "
                 f"{v['jumps_per_100_frames_before']:.2f} & {v['jumps_per_100_frames_after']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (HERE / "sections" / "gen_scripture_qa.tex").write_text("\n".join(L), encoding="utf-8")


def main():
    out = {}
    quran_tables(out)
    hadith_tables(out)
    aspect_qa(out)
    avatar_results(out)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "scripture_eda.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                      encoding="utf-8")
    figures(out)
    tex(out)
    print(json.dumps({k: out[k] for k in ("hadith", "hadith_videos_total", "pose_qa")}, ensure_ascii=False, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
