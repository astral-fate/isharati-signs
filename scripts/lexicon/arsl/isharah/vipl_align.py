"""Second-opinion gloss boundaries for Isharah sentences: CTC forced alignment with the ICCV 2025 MSLR winner
(VIPL-SLP TwoStream_Cosign, repo D:/islam/models/vipl_mslr), compared with our Conformer alignments.

The VIPL model is run through the repo's own SkeletonFeeder (test-time transform) and network in eval mode.
Per-frame log-probs come from the fusion stream's BiLSTM ("seq") logits * norm_scale, i.e. the output the repo
decodes for the SI track (recognized_sents_fusion). Temporal mapping to source frames:
  TemporalRescale_test pads T to a multiple of 4 and keeps every 2nd frame (x2), collate_fn pads 6 frames on the
  left, TemporalConv K3-P2-K3-P2 pools x4 -> model frame k covers padded input [4k, 4k+4)
  -> input frames [4k-6, 4k-2) -> source frames [8k-12, 8k-4); a span of model frames [a, b) -> source [8a-12, 8b-12)
  (clamped to [0, T]; glosses the model puts in the padding become empty spans).

Steps (each loads at most one pose pkl):
  python isharah_vipl_align.py verify [si_dev|si_test]  # greedy-decode ~200 si dev sentences, WER
  python isharah_vipl_align.py align  [weights]         # force-align candidate sids on the Isharah-1000 pkl
  python isharah_vipl_align.py match                    # compare recordings against the Isharah-2000 pkl
  python isharah_vipl_align.py report                   # write vipl_alignments.jsonl + alignment_agreement.md
Run with D:/islam/models/.venv/Scripts/python.
"""
import json
import pickle
import sys
import types
from itertools import groupby
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
REPO = Path(r"D:\islam\models\vipl_mslr")
PKL1K = Path(r"D:\islam\ishara\v1\pose_data_isharah1000_hands_lips_body_May12.pkl")
PKL2K = Path(r"D:\islam\ishara\pose_data_isharah2000_hands_lips_body_phase1_SI.pkl")
OURS = DATA / "lexicon_v2" / "isharah" / "alignments.jsonl"
OUT = DATA / "lexicon_v2" / "isharah"
WORK = Path(r"D:\islam\models\vipl_work")  # intermediate results
FPS = 30.0
FP_POS = np.linspace(0, 1, 9)  # fingerprint frame positions (fractions of T)


# ----------------------------------------------------------------------------- model
def import_repo():
    """Import the repo modules. ctcdecode (beam search) and torchvision (unused import) are stubbed:
    we only need logits, not the repo's beam decoder."""
    if "ctcdecode" not in sys.modules:
        sys.modules["ctcdecode"] = types.ModuleType("ctcdecode")
        sys.modules["ctcdecode"].CTCBeamDecoder = lambda *a, **k: None
    try:
        import torchvision  # noqa: F401
    except ImportError:
        tv = types.ModuleType("torchvision")
        tv.models = types.ModuleType("torchvision.models")
        sys.modules["torchvision"], sys.modules["torchvision.models"] = tv, tv.models
    sys.path.insert(0, str(REPO))
    import yaml
    import slr_network
    from datasets.skeleton_feeder import SkeletonFeeder
    cfg = yaml.safe_load((REPO / "configs" / "Double_Cosign_si.yaml").read_text(encoding="utf-8"))
    return slr_network, SkeletonFeeder, cfg


def read_list(name):
    rows = []
    for line in (REPO / "preprocess" / "mslr2025" / name).read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("|")
        if len(parts) >= 2:
            rows.append((parts[0], parts[1].strip()))
    return rows


def build_gloss_dict():
    """Replicates preprocess/mslr2025/mslr_process.py for setting 'si': sorted unique glosses of the train and
    dev lists, index from 1 (0 = CTC blank)."""
    freq = {}
    for name in ("si_train_list.txt", "si_dev_list.txt"):
        for _, g in read_list(name):
            for w in g.split():
                freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda d: d[0])
    gd = {"id2gloss": {}, "gloss2id": {}}
    for idx, (k, v) in enumerate(items):
        gd["gloss2id"][k] = {"index": idx + 1, "frequency": v}
        gd["id2gloss"][idx + 1] = {"gloss": k, "frequency": v}
    shipped = REPO / "datasets" / "mslr2025" / "si_gloss_dict.json"
    if shipped.exists():
        ref = json.loads(shipped.read_text(encoding="utf-8"))
        same = {k: v["index"] for k, v in ref["gloss2id"].items()} == {k: v["index"] for k, v in gd["gloss2id"].items()}
        print(f"gloss dict: {len(items)} glosses; identical to shipped si_gloss_dict.json: {same}")
    return gd


def make_feeder(SkeletonFeeder, cfg, g2i, kps):
    """SkeletonFeeder in eval ('dev', test transform) mode, without its hard-coded pkl path: the attributes are
    set exactly as SkeletonFeeder.__init__ sets them, then its own __getitem__/normalize/collate_fn are used."""
    fa = cfg["feeder_args"]
    f = SkeletonFeeder.__new__(SkeletonFeeder)
    f.mode, f.mode_list, f.dict, f.setting = "dev", ["dev"], g2i, fa["setting"]
    f.data_type, f.transform_mode, f.dataset, f.used_part = fa["datatype"], "test", cfg["dataset"], fa["used_part"]
    f.kps_global, f.inputs_list = kps, []
    f.norm_div = (10240 - 1) / 2
    f.pose_idx = []
    for part in f.used_part:
        if part == "body":
            f.pose_idx += list(range(61, 86))
        elif part == "hand21":
            f.pose_idx += list(range(0, 21)) + list(range(21, 42))
        elif part == "mouth_8":
            f.pose_idx += list(range(42, 61))
    f.split, f.norm_point = fa["split"], fa["norm_point"]
    f.data_aug = f.pose_transform()
    return f


class Vipl:
    def __init__(self, weights, kps):
        import torch
        slr_network, SkeletonFeeder, cfg = import_repo()
        self.torch = torch
        self.gd = build_gloss_dict()
        self.g2i = {k: v["index"] for k, v in self.gd["gloss2id"].items()}
        self.i2g = {v: k for k, v in self.g2i.items()}
        self.model = slr_network.TwoStream_Cosign(**cfg["model_args"], gloss_dict=self.gd)
        state = torch.load(REPO / "weights" / f"{weights}.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        res = self.model.load_state_dict(state, strict=False)
        print(f"weights {weights}: missing {len(res.missing_keys)} {res.missing_keys[:5]}, "
              f"unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:5]}")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev).eval()
        self.feeder = make_feeder(SkeletonFeeder, cfg, self.g2i, kps)
        self.SkeletonFeeder = SkeletonFeeder

    def log_probs(self, sid, glosses=""):
        """(T', C) log-probs of the fusion seq logits for one sentence (batch of 1, repo collate)."""
        torch = self.torch
        self.feeder.inputs_list = [{"video_id": sid, "gloss_sequence": glosses, "original_info": sid}]
        batch = self.SkeletonFeeder.collate_fn([self.feeder[0]])
        with torch.no_grad():
            x, len_x = batch["x"].to(self.dev), batch["len_x"].to(self.dev)
            m = self.model
            fusion = m.visual_module(x, len_x)["fusion"]
            _, seq_logits, feat_len = m.forward_contextual(fusion, len_x, m.conv1d_fusion,
                                                          m.contextual_module_fusion, m.classifier_fusion)
            lp = (seq_logits[: int(feat_len[0]), 0] * m.norm_scale).log_softmax(-1)
        return lp.float().cpu().numpy()


def model_to_src(a, b, t_src):
    return [int(min(max(8 * a - 12, 0), t_src)), int(min(max(8 * b - 12, 0), t_src))]


def load_pkl(path):
    print("loading", path.name, flush=True)
    with open(path, "rb") as f:
        return pickle.load(f)


def forced_align(lp, targets):
    sys.path.insert(0, str(Path(__file__).parent))
    from ctc_align import forced_align as fa  # reuse our Viterbi
    return fa(lp, targets)


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return int(d[-1]), len(ref)


# ----------------------------------------------------------------------------- steps
def verify(weights_list, n=200):
    kps = load_pkl(PKL1K)
    dev = read_list("si_dev_list.txt")
    idx = np.linspace(0, len(dev) - 1, n).astype(int)
    for w in weights_list:
        v = Vipl(w, kps)
        errs = tot = 0
        for i in idx:
            sid, g = dev[i]
            if sid not in kps:
                continue
            pred = [int(k) for k, _ in groupby(v.log_probs(sid, g).argmax(-1)) if k != 0]
            e, t = wer(g.split(), [v.i2g.get(k, "?") for k in pred])
            errs, tot = errs + e, tot + t
        print(f"RESULT {w}: greedy si-dev WER {errs / tot:.2%} ({errs}/{tot} glosses, {len(idx)} sentences)", flush=True)
        del v
        import torch
        torch.cuda.empty_cache()


def fingerprint(k):
    k = np.asarray(k, dtype=np.float32)
    return k[np.round(FP_POS * (len(k) - 1)).astype(int)]


def align(weights):
    ours = [json.loads(l) for l in OURS.read_text(encoding="utf-8").splitlines() if l.strip()]
    kps = load_pkl(PKL1K)
    lists = {sid: g.split() for name in ("si_train_list.txt", "si_dev_list.txt") for sid, g in read_list(name)}
    v = Vipl(weights, kps)
    WORK.mkdir(parents=True, exist_ok=True)
    out, stats = {}, {"ours": len(ours), "not_in_1k": 0, "gloss_mismatch": 0, "oov": 0, "fail": 0}
    for n, row in enumerate(ours):
        sid, glosses = row["sid"], row["glosses"]
        if sid not in kps:
            stats["not_in_1k"] += 1
            continue
        if sid in lists and lists[sid] != glosses:
            stats["gloss_mismatch"] += 1
            continue
        if sid not in lists:
            stats["gloss_mismatch"] += 1  # no Isharah-1000 label to confirm the sentence
            continue
        if any(g not in v.g2i for g in glosses):
            stats["oov"] += 1
            continue
        k = kps[sid]["keypoints"]
        lp = v.log_probs(sid, " ".join(glosses))
        res = forced_align(lp, [v.g2i[g] for g in glosses])
        greedy = [v.i2g.get(int(c), "?") for c, _ in groupby(lp.argmax(-1)) if c != 0]
        if res is None or any(s is None for s in res[0]):
            stats["fail"] += 1
            continue
        spans, score = res
        out[sid] = {"glosses": glosses, "spans_model": spans, "score": score, "T1k": len(k), "Tp": lp.shape[0],
                    "greedy_wer": wer(glosses, greedy), "fp": fingerprint(k)}
        if len(out) % 1000 == 0:
            print(f"  aligned {len(out)} ({n + 1}/{len(ours)} rows seen)", flush=True)
    stats["aligned"] = len(out)
    print("align stats", stats)
    with open(WORK / "vipl_raw.pkl", "wb") as f:
        pickle.dump({"weights": weights, "stats": stats, "rows": out}, f)


def match():
    raw = pickle.load(open(WORK / "vipl_raw.pkl", "rb"))
    kps = load_pkl(PKL2K)
    res = {}
    for sid, r in raw["rows"].items():
        if sid not in kps:
            res[sid] = {"T2k": None}
            continue
        k2 = kps[sid]["keypoints"]
        T2 = len(k2)
        d = None
        if T2 == r["T1k"]:
            fp2 = fingerprint(k2)
            if fp2.shape == r["fp"].shape:
                d = float(np.abs(fp2 - r["fp"]).mean())
        res[sid] = {"T2k": T2, "fp_mad": d}
    del kps
    with open(WORK / "vipl_match.pkl", "wb") as f:
        pickle.dump(res, f)
    ts = [r for r in res.values() if r["T2k"] is not None]
    same_t = [r for r in ts if r.get("fp_mad") is not None]
    mads = np.array([r["fp_mad"] for r in same_t])
    print(f"in 2k pkl {len(ts)}/{len(res)}; same T {len(same_t)}; fingerprint mean-abs-diff (px) "
          f"quantiles 0/50/90/99/100: {np.percentile(mads, [0, 50, 90, 99, 100]) if len(mads) else '-'}")
    print("  MAD==0:", int((mads == 0).sum()), " MAD<1px:", int((mads < 1).sum()))


def activity():
    """Independent plausibility check from the keypoints: a gloss span should fall where the signer's hands are
    visible and moving. Per sentence: hand-visible mask and wrist speed (px/frame) for both hands."""
    raw = pickle.load(open(WORK / "vipl_raw.pkl", "rb"))["rows"]
    kps = load_pkl(PKL1K)
    act = {}
    for sid in raw:
        k = np.asarray(kps[sid]["keypoints"], dtype=np.float32)
        vis = np.stack([np.abs(k[:, 0:21]).sum((1, 2)) > 0, np.abs(k[:, 21:42]).sum((1, 2)) > 0], 1)
        w = k[:, [0, 21]]
        sp = np.zeros((len(k), 2), np.float32)
        sp[1:] = np.linalg.norm(w[1:] - w[:-1], axis=-1)
        sp[1:][~(vis[1:] & vis[:-1])] = 0
        act[sid] = {"vis": vis.any(1), "speed": sp.max(1)}
    del kps
    with open(WORK / "vipl_activity.pkl", "wb") as f:
        pickle.dump(act, f)
    print("activity saved", len(act))


def act_stats(spans_list, acts):
    """Share of (non-empty) gloss spans whose centre frame shows a hand, and mean wrist speed inside the span
    relative to the sentence's mean speed over hand-visible frames; empty spans count as failures."""
    vis_ok, rel, empty = [], [], 0
    for spans, a in zip(spans_list, acts):
        ref = a["speed"][a["vis"]].mean() if a["vis"].any() else 0
        for s0, s1 in spans:
            if s1 <= s0:
                empty += 1; vis_ok.append(False); continue
            c = min((s0 + s1) // 2, len(a["vis"]) - 1)
            vis_ok.append(bool(a["vis"][c]))
            if ref > 0:
                rel.append(a["speed"][s0:s1].mean() / ref)
    return np.mean(vis_ok), np.mean(rel), empty / len(vis_ok)


def even_split(n, t):
    e = np.linspace(0, t, n + 1).round().astype(int)
    return [[int(e[i]), int(e[i + 1])] for i in range(n)]


def partition(spans, t):
    """Contiguous segmentation from (peaky) spans: boundary between gloss k and k+1 at the midpoint of the gap
    between end_k and start_{k+1}; first gloss starts at 0, last ends at T."""
    b = [0] + [int(round((spans[i][1] + spans[i + 1][0]) / 2)) for i in range(len(spans) - 1)] + [t]
    return [[b[i], max(b[i + 1], b[i])] for i in range(len(spans))]


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def metrics(pairs):
    """pairs: list of (spansA, spansB) per sentence -> dict of per-gloss agreement numbers."""
    bd, ious, cd = [], [], []
    for A, B in pairs:
        for a, b in zip(A, B):
            bd += [abs(a[0] - b[0]), abs(a[1] - b[1])]
            cd.append(abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2))
            ious.append(iou(a, b))
    bd, ious, cd = map(np.array, (bd, ious, cd))
    return {"n": len(ious), "bnd_mean": bd.mean(), "bnd_med": np.median(bd), "ctr_mean": cd.mean(),
            "ctr_med": np.median(cd), "iou_mean": ious.mean(), "iou50": (ious >= 0.5).mean(),
            "ctr_le8": (cd <= 8).mean()}


def report(mad_thresh=1.0):
    raw = pickle.load(open(WORK / "vipl_raw.pkl", "rb"))
    mt = pickle.load(open(WORK / "vipl_match.pkl", "rb"))
    acts_all = pickle.load(open(WORK / "vipl_activity.pkl", "rb")) if (WORK / "vipl_activity.pkl").exists() else None
    ours = {}
    for l in OURS.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            ours[r["sid"]] = r
    keep = sorted(s for s, m in mt.items() if m.get("fp_mad") is not None and m["fp_mad"] <= mad_thresh)
    names = {"ov": "Conformer vs VIPL", "ev": "even split vs VIPL", "eo": "even split vs Conformer"}
    groups = ("all", "interior")  # interior = not the first or last gloss of the sentence (sentences with >= 3)
    sets = {f: {g: {c: [] for c in names} for g in groups} for f in ("raw", "part")}
    spans_by = {"Conformer": [], "VIPL": [], "even split": []}
    acts, per_sent_iou, gw = [], [], [0, 0]
    with open(OUT / "vipl_alignments.jsonl", "w", encoding="utf-8") as f:
        for sid in keep:
            r, o = raw["rows"][sid], ours[sid]
            T = r["T1k"]
            vs = [model_to_src(a, b, T) for a, b in r["spans_model"]]
            f.write(json.dumps({"sid": sid, "glosses": r["glosses"], "spans": vs, "score": round(r["score"], 4)},
                               ensure_ascii=False) + "\n")
            os_, ev = o["spans"], even_split(len(vs), T)
            po, pv = partition(os_, T), partition(vs, T)
            for form, (A, B, E) in (("raw", (os_, vs, ev)), ("part", (po, pv, ev))):
                pairs = {"ov": (A, B), "ev": (E, B), "eo": (E, A)}
                for c, (x, y) in pairs.items():
                    sets[form]["all"][c].append((x, y))
                    if len(vs) >= 3:
                        sets[form]["interior"][c].append((x[1:-1], y[1:-1]))
            per_sent_iou.append(np.mean([iou(a, b) for a, b in zip(po, pv)]))
            spans_by["Conformer"].append(os_); spans_by["VIPL"].append(vs); spans_by["even split"].append(ev)
            if acts_all is not None:
                acts.append(acts_all[sid])
            gw[0] += r["greedy_wer"][0]; gw[1] += r["greedy_wer"][1]
    M = {f: {g: {c: metrics(v) for c, v in d.items()} for g, d in fd.items()} for f, fd in sets.items()}
    wo = np.array([b - a for A in spans_by["Conformer"] for a, b in A])
    wv = np.array([b - a for A in spans_by["VIPL"] for a, b in A])
    T_all = np.array([raw["rows"][s]["T1k"] for s in keep])
    signers = sorted({s[:2] for s in keep})
    n_dev = sum(s.startswith("02_") for s in keep)
    # where do the aligners put each gloss (mean relative centre by position in the sentence)
    rel = {}
    for name in ("Conformer", "VIPL"):
        for sp, sid in zip(spans_by[name], keep):
            T, n = raw["rows"][sid]["T1k"], len(sp)
            for k, (a, b) in enumerate(sp):
                key = "first" if k == 0 else "last" if k == n - 1 else f"#{k + 1}" if k < 4 else "#5+"
                rel.setdefault(key, {}).setdefault(name, []).append((a + b) / 2 / T)
    for f_ in M:
        for g in M[f_]:
            for c in M[f_][g]:
                print(f_, g, c, {a: round(float(b), 3) for a, b in M[f_][g][c].items()})

    def line(name, m):
        return (f"| {name} | {m['n']} | {m['bnd_mean']:.1f} ({m['bnd_mean'] / FPS:.2f} s) | {m['bnd_med']:.0f} | "
                f"{m['ctr_mean']:.1f} ({m['ctr_mean'] / FPS:.2f} s) | {m['ctr_med']:.0f} | {m['iou_mean']:.2f} | "
                f"{m['iou50']:.0%} | {m['ctr_le8']:.0%} |")

    hdr = ("| comparison | glosses | mean abs boundary diff, frames (s) | median | mean abs centre diff, frames (s) "
           "| median | mean IoU | IoU >= 0.5 | centre within 8 frames |\n|---|---|---|---|---|---|---|---|---|")

    def table(form):
        out = []
        for g, title in (("all", "All glosses"), ("interior", "Interior glosses only (not first/last; sentences with >= 3 glosses)")):
            out.append(f"**{title}**\n\n{hdr}\n" + "\n".join(line(names[c], M[form][g][c]) for c in names))
        return "\n\n".join(out)

    st = raw["stats"]
    per_sent_iou = np.array(per_sent_iou)
    relrows = "\n".join(f"| {k} | {len(v['Conformer'])} | {np.mean(v['Conformer']):.2f} | {np.mean(v['VIPL']):.2f} |"
                        for k, v in sorted(rel.items(), key=lambda kv: ["first", "#2", "#3", "#4", "#5+", "last"].index(kv[0])))
    act_md = ""
    if acts:
        arows = []
        for name, sp in spans_by.items():
            v_, r_, e_ = act_stats(sp, acts)
            arows.append(f"| {name} | {v_:.0%} | {r_:.2f} | {e_:.1%} |")
        act_md = ("\n## 3. Plausibility check from the keypoints (no aligner involved)\n"
                  "A gloss can only be signed where a hand is visible. For each aligner: share of glosses whose span centre "
                  "falls on a frame with at least one hand detected, and mean wrist speed inside the span relative to the "
                  "sentence's mean over hand-visible frames (> 1 = the span sits on more movement than average). Empty "
                  "spans (gloss placed in the padding / zero-width after clamping) count as not visible.\n\n"
                  "| aligner | span centre on a hand-visible frame | relative wrist speed in span | empty spans |\n"
                  "|---|---|---|---|\n" + "\n".join(arows) + "\n")
    md = f"""# Isharah gloss boundaries: Conformer vs VIPL (ICCV 2025 MSLR winner)

Script: `scripts/lexicon/arsl/isharah/vipl_align.py` (steps verify / align / match / activity / report).
VIPL alignments: `vipl_alignments.jsonl` (same format as `alignments.jsonl`: source frames of the pose pkl, [start, end)).

## Setup
- Second aligner: VIPL-SLP TwoStream_Cosign (`configs/Double_Cosign_si.yaml`), weights `{raw['weights']}.pt`, run through
  the repo's own `SkeletonFeeder` (test transform) and network in eval mode; per-frame log-probs = log_softmax of the
  fusion-stream BiLSTM logits x norm_scale (the output the repo decodes for the SI track). Greedy decoding of 200
  si-dev sentences (signer 02): WER 2.98% with `si_dev.pt`, 1.17% with `si_test.pt` (repo reports ~2% / 7.4% test).
- One VIPL model frame = 8 source frames (x2 frame skip in the test transform, x4 pooling in TemporalConv); the
  collate function pads 6 input frames (12 source frames) on the left and 6+ on the right, so model frames [a, b)
  -> source [8a-12, 8b-12), clamped to [0, T]. Our Conformer: model frame k -> source [4k, 4k+4).
- Forced alignment: the same CTC Viterbi (`forced_align` from `ctc_align.py`) for both.
- Sentences: rows of `alignments.jsonl` whose sid is in the Isharah-1000 pkl, whose gloss sequence equals the
  Isharah-1000 si-list label, and whose recording is the same in both pkls (identical T and mean abs keypoint
  difference <= {mad_thresh} px on 9 sampled frames; in fact all were bit-identical).
  Funnel: {st['ours']} our rows -> {st['ours'] - st['not_in_1k']} sid in 1k pkl -> {st['aligned']} same label, in
  vocabulary, alignable -> **{len(keep)} same recording** (signers {', '.join(signers)}; {n_dev} from dev signer 02,
  the rest are VIPL training recordings).
- {len(keep)} sentences, {len(wo)} glosses, mean clip {T_all.mean():.0f} frames ({T_all.mean() / FPS:.1f} s at {FPS:.0f} fps).
  Greedy VIPL WER on these sentences: {gw[0] / gw[1]:.1%}.

## 1. Raw spans (as each aligner outputs them)
Both are CTC models, so forced-alignment spans are the gloss "spikes"; frames between spikes go to blank.
Mean span width: Conformer {wo.mean():.1f} frames (median {np.median(wo):.0f}), VIPL {wv.mean():.1f} (median {np.median(wv):.0f}).
The even-split baseline uses contiguous spans, so its IoU against spikes is low by construction.

{table('raw')}

## 2. Contiguous segments (spans turned into a partition of the clip)
Boundary between consecutive glosses at the midpoint of the gap between their spikes; first gloss starts at 0, last
ends at T. This is the form comparable to the even-split baseline.

{table('part')}

Per sentence (partition form), mean IoU Conformer vs VIPL >= 0.5 for {(per_sent_iou >= 0.5).mean():.0%} of sentences,
>= 0.7 for {(per_sent_iou >= 0.7).mean():.0%}.

### Where each aligner puts the glosses (mean span centre as a fraction of the clip, by position in the sentence)

| gloss position | glosses | Conformer | VIPL |
|---|---|---|---|
{relrows}
{act_md}"""
    notes = WORK / "agreement_notes.md"  # hand-written caveats/judgement appended if present
    if notes.exists():
        md += "\n" + notes.read_text(encoding="utf-8")
    (OUT / "alignment_agreement.md").write_text(md, encoding="utf-8")
    print(md)



if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "verify":
        verify(sys.argv[2:] or ["si_dev", "si_test"])
    elif mode == "align":
        align(sys.argv[2] if len(sys.argv) > 2 else "si_dev")
    elif mode == "match":
        match()
    elif mode == "activity":
        activity()
    elif mode == "report":
        report()
