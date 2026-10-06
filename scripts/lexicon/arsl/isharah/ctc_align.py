"""Per-gloss boundaries in Isharah sentences by CTC forced alignment with the CSLR Conformer.

The model (FatimahEmadEldin/cslr-conformer-isharah, downloaded to D:/islam/models/) outputs per-frame gloss
log-probabilities (4x subsampled). Forcing the known gloss sequence through them (Viterbi over the CTC
topology) gives each gloss its frames. Run with the models venv (it has PyTorch):

  D:/islam/models/.venv/Scripts/python scripts/lexicon/arsl/isharah/ctc_align.py verify   # pick the gloss<->id map
  D:/islam/models/.venv/Scripts/python scripts/lexicon/arsl/isharah/ctc_align.py align    # all sentences

The HF repo's si_gloss_dict.json is the 683-gloss Isharah-1k vocabulary, but the weights have 1,102 classes
(Isharah-2k). `verify` decodes the dev set under candidate mappings built the way the 1k dictionary was
(sorted glosses from index 1, 0 = blank) and keeps the one with the lowest WER.

Output: data/lexicon_v2/isharah/alignments.jsonl, one row per sentence:
  {"sid", "glosses": [...], "spans": [[start, end), ...] in source frames, "score": mean log-prob per frame}
"""
import importlib.util
import json
import pickle
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
MODEL_DIR = Path(r"D:\islam\models\cslr-conformer-isharah")
ISHARAH = Path(r"D:\islam\ishara")
PKL = ISHARAH / "pose_data_isharah2000_hands_lips_body_phase1_SI.pkl"
OUT = DATA / "lexicon_v2" / "isharah"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_module(name):
    """The HF files start with notebook '%%writefile' lines; drop them and import the rest."""
    src = (MODEL_DIR / f"{name}.py").read_text(encoding="utf-8").splitlines()
    code = "\n".join(l for l in src if not l.startswith("%%"))
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(code, str(MODEL_DIR / f"{name}.py"), "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod


def read_config(path):
    """The HF config.json holds the notebook cell that wrote it (`config = {...}`); read that dict literal."""
    import ast
    import re
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        body = re.search(r"config\s*=\s*(\{.*?\})", text, re.S).group(1)
        return ast.literal_eval(re.sub(r"#.*", "", body))


def load_model():
    model_mod, prep = load_module("model"), load_module("preprocessor")
    cfg = read_config(MODEL_DIR / "config.json")
    model = model_mod.CSLRModel(**cfg)
    state = torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE, weights_only=True)
    # the checkpoint names the embedding layer `linear`; the uploaded model.py calls it `linear_emb`
    state = {("linear_emb" + k[len("linear"):] if k.startswith("linear.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(DEVICE).eval(), prep.preprocess_input, cfg["num_classes"]


def sentences(split):
    rows = {}
    for line in (ISHARAH / split).read_text(encoding="utf-8").splitlines()[1:]:
        if "|" in line:
            sid, g = line.split("|", 1)
            rows[sid] = g.split()
    return rows


def candidate_maps(train_glosses, n_classes):
    vocab = sorted(train_glosses)
    maps = {"sorted_from_1": {g: i + 1 for i, g in enumerate(vocab)}}
    extra = n_classes - 1 - len(vocab)
    if extra > 0:
        maps[f"sorted_from_{1 + extra}"] = {g: i + 1 + extra for i, g in enumerate(vocab)}
    return maps


@torch.no_grad()
def log_probs(model, prep, keypoints):
    x = prep(np.asarray(keypoints)).to(DEVICE)
    logits, _ = model(x, torch.tensor([x.shape[1]], device=DEVICE))
    return F.log_softmax(logits[0], dim=-1).float().cpu().numpy()


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return d[-1], len(ref)


def forced_align(lp, targets):
    """Viterbi over the CTC graph (blank, t1, blank, t2, ..., blank). Returns per-target [start, end)
    in model frames, and the mean log-prob of the best path, or None if the sentence is too short."""
    T, S = lp.shape[0], 2 * len(targets) + 1
    if T < len(targets):
        return None
    labels = [0 if s % 2 == 0 else targets[s // 2] for s in range(S)]
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), np.int8)
    dp[0, 0], dp[0, 1] = lp[0, 0], lp[0, labels[1]]
    for t in range(1, T):
        for s in range(S):
            best, move = dp[t - 1, s], 0
            if s > 0 and dp[t - 1, s - 1] > best:
                best, move = dp[t - 1, s - 1], 1
            if s > 1 and labels[s] != 0 and labels[s] != labels[s - 2] and dp[t - 1, s - 2] > best:
                best, move = dp[t - 1, s - 2], 2
            dp[t, s] = best + lp[t, labels[s]]
            bp[t, s] = move
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    score = dp[T - 1, s] / T
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = []
    for k in range(len(targets)):
        frames = [t for t in range(T) if path[t] == 2 * k + 1]
        spans.append([frames[0], frames[-1] + 1] if frames else None)
    return spans, float(score)


def verify(model, prep, n_classes, data):
    train, dev = sentences("train.csv"), sentences("dev.csv")
    vocab = {g for gs in train.values() for g in gs}
    print(f"train vocabulary {len(vocab)}, model classes {n_classes}")
    for name, g2i in candidate_maps(vocab, n_classes).items():
        i2g = {i: g for g, i in g2i.items()}
        errs = total = 0
        for sid in list(dev)[:300]:
            if sid not in data:
                continue
            pred = [k for k, _ in groupby(log_probs(model, prep, data[sid]["keypoints"]).argmax(-1)) if k != 0]
            e, n = wer(dev[sid], [i2g.get(int(k), "?") for k in pred])
            errs, total = errs + e, total + n
        print(f"  mapping {name}: dev WER {errs / total:.1%} on {total} glosses")


def align(model, prep, n_classes, data, mapping):
    train = sentences("train.csv")
    vocab = {g for gs in train.values() for g in gs}
    g2i = candidate_maps(vocab, n_classes)[mapping]
    rows = {**train, **sentences("dev.csv")}
    OUT.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    with open(OUT / "alignments.jsonl", "w", encoding="utf-8") as f:
        for sid, glosses in rows.items():
            if sid not in data or any(g not in g2i for g in glosses):
                skipped += 1
                continue
            lp = log_probs(model, prep, data[sid]["keypoints"])
            res = forced_align(lp, [g2i[g] for g in glosses])
            if res is None or any(s is None for s in res[0]):
                skipped += 1
                continue
            spans, score = res
            t_src = len(data[sid]["keypoints"])
            # model frame k covers source frames [4k, 4k+4); clamp to the clip
            src = [[min(4 * a, t_src), min(4 * b, t_src)] for a, b in spans]
            f.write(json.dumps({"sid": sid, "glosses": glosses, "spans": src, "score": round(score, 4)},
                               ensure_ascii=False) + "\n")
            done += 1
            if done % 2000 == 0:
                print(f"  aligned {done}", flush=True)
    print(f"aligned {done} sentences, skipped {skipped}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    model, prep, n_classes = load_model()
    print("device", DEVICE, "- loading", PKL.name, flush=True)
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    if mode == "verify":
        verify(model, prep, n_classes, data)
    else:
        align(model, prep, n_classes, data, sys.argv[2] if len(sys.argv) > 2 else "sorted_from_1")


if __name__ == "__main__":
    main()
