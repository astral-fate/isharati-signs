"""Weakly supervised sign spotter (multiple-instance learning over caption words).

A temporal convolutional network gives, for every frame, a score per vocabulary word. A clip's score for a word is
the smooth maximum (log-sum-exp) of its frame scores, trained with binary cross-entropy against the words of its
English caption (a word in the caption = its sign is somewhere in the clip; the network learns where). Lexicon signs
(prepare.py) are short clips with one clean label. Order-free, so English captions work although ASL order differs;
the approach of sign spotting with subtitles (BSL-1K / Momeni et al.).

  train   fits the model on the train split, keeps the best epoch by validation mAP
  spot    per word: the clips with the highest score among those whose caption has the word; the sign is the
          stretch around the frame peak (score > half of the peak, 0.4-1.6 s) -> candidates.json for localize.py

Runs in D:/islam/models/.venv (PyTorch + CUDA):
  D:/islam/models/.venv/Scripts/python scripts/mining/mil_train.py train|spot
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isharati.config import DATA  # noqa: E402
MIL = DATA / "youtube_asl" / "mil"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR, R = 30, 64, 1e-3, 4.0
TOP_CLIPS, MIN_LEN, MAX_LEN = 8, 6, 24        # spotting: clips per word; sign length in frames at 15 fps


class Spotter(nn.Module):
    def __init__(self, dim, vocab, width=192):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, width), nn.GELU())
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * d, dilation=d),
                                                  nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(0.2))
                                    for d in (1, 2, 4, 1, 2))
        self.out = nn.Conv1d(width, vocab, 1)

    def forward(self, x):                       # x [B, T, D] -> per-frame logits [B, V, T]
        v = torch.cat([x, F.pad(x[:, 1:] - x[:, :-1], (0, 0, 1, 0))], -1)
        h = self.inp(v).transpose(1, 2)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def pool(logits, mask):
    """Smooth max over valid frames: log(mean(exp(R x))) / R."""
    x = logits * R
    x = x.masked_fill(~mask[:, None, :], -1e4)
    n = mask.sum(1).clamp(min=1).float()
    return (torch.logsumexp(x, -1) - torch.log(n)[:, None]) / R


class Data:
    def __init__(self):
        self.meta = json.loads((MIL / "meta.json").read_text(encoding="utf-8"))
        self.X = np.load(MIL / "feats.npy", mmap_mode="r")
        self.V = len(self.meta["vocab"])
        self.items = self.meta["items"]
        self.mu = np.asarray(self.X[::50], np.float32).mean(0)
        self.sd = np.asarray(self.X[::50], np.float32).std(0) + 1e-3

    def batch(self, idx):
        its = [self.items[i] for i in idx]
        T = max(i["len"] for i in its)
        x = np.zeros((len(its), T, self.X.shape[1]), np.float32)
        m = np.zeros((len(its), T), bool)
        y = np.zeros((len(its), self.V), np.float32)
        for k, it in enumerate(its):
            x[k, : it["len"]] = (np.asarray(self.X[it["start"]: it["start"] + it["len"]], np.float32) - self.mu) / self.sd
            m[k, : it["len"]] = True
            y[k, it["labels"]] = 1
        return torch.from_numpy(x).to(DEV), torch.from_numpy(m).to(DEV), torch.from_numpy(y).to(DEV)


def mean_ap(scores, y):
    aps = []
    for v in range(y.shape[1]):
        pos = y[:, v] > 0
        if pos.sum() == 0:
            continue
        order = np.argsort(-scores[:, v])
        hits = pos[order]
        prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        aps.append(float((prec * hits).sum() / hits.sum()))
    return float(np.mean(aps)), aps


@torch.no_grad()
def predict(model, d, idx, frames=False):
    model.eval()
    out, per = [], []
    for k in range(0, len(idx), BATCH):
        x, m, _ = d.batch(idx[k:k + BATCH])
        lg = model(x)
        out.append(torch.sigmoid(pool(lg, m)).cpu().numpy())
        if frames:
            per += [torch.sigmoid(lg[j, :, : int(m[j].sum())]).cpu().numpy() for j in range(len(lg))]
    return np.concatenate(out), per


def labels(d, idx):
    y = np.zeros((len(idx), d.V), np.float32)
    for k, i in enumerate(idx):
        y[k, d.items[i]["labels"]] = 1
    return y


def train():
    torch.manual_seed(0); random.seed(0)
    d = Data()
    tr = [i for i, it in enumerate(d.items) if it["split"] == "train"]
    va = [i for i, it in enumerate(d.items) if it["split"] == "val"]
    te = [i for i, it in enumerate(d.items) if it["split"] == "test"]
    freq = labels(d, tr).mean(0)
    pos_weight = torch.tensor(np.clip((1 - freq) / np.maximum(freq, 1e-4), 1, 30) ** 0.5, device=DEV)
    model = Spotter(d.X.shape[1], d.V).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
    steps = EPOCHS * math.ceil(len(tr) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    yv = labels(d, va)
    best, log = -1, []
    tr_sorted = sorted(tr, key=lambda i: d.items[i]["len"])
    for ep in range(EPOCHS):
        model.train()
        chunks = [tr_sorted[k:k + BATCH] for k in range(0, len(tr_sorted), BATCH)]   # similar lengths per batch
        random.shuffle(chunks)
        tot = 0.0
        for idx in chunks:
            x, m, y = d.batch(idx)
            loss = F.binary_cross_entropy_with_logits(pool(model(x), m), y, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        sv, _ = predict(model, d, va)
        mapv, _ = mean_ap(sv, yv)
        log.append({"epoch": ep + 1, "loss": tot / len(tr), "val_map": mapv})
        print(f"epoch {ep + 1:2d} loss {tot / len(tr):.4f} val mAP {mapv:.4f}", flush=True)
        if mapv > best:
            best = mapv
            torch.save({"model": model.state_dict(), "mu": d.mu, "sd": d.sd}, MIL / "spotter.pt")
    ck = torch.load(MIL / "spotter.pt", weights_only=False)
    model.load_state_dict(ck["model"])
    st, _ = predict(model, d, te)
    yt = labels(d, te)
    mapt, aps = mean_ap(st, yt)
    vocab = d.meta["vocab"]
    seen = [v for v in range(d.V) if yt[:, v].sum() > 0]
    ap_of = dict(zip(seen, aps))
    held = [vocab.index(w) for w in d.meta["held_out"] if vocab.index(w) in ap_of]
    miss = [vocab.index(w) for w in d.meta["missing"] if vocab.index(w) in ap_of]
    # chance level: AP of a random ranking = share of positives
    chance = float(np.mean([yt[:, v].mean() for v in seen]))
    res = {"best_val_map": best, "test_map": mapt, "test_map_chance": chance,
           "test_map_held_out_words": float(np.mean([ap_of[v] for v in held])),
           "test_map_missing_words": float(np.mean([ap_of[v] for v in miss])),
           "vocab": d.V, "train_items": len(tr), "log": log}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "spotter_train.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1))


def spot():
    """Candidates for the held-out words (evaluation) and the missing words (new signs), from all YouTube clips."""
    d = Data()
    ck = torch.load(MIL / "spotter.pt", weights_only=False)
    model = Spotter(d.X.shape[1], d.V).to(DEV)
    model.load_state_dict(ck["model"])
    d.mu, d.sd = ck["mu"], ck["sd"]
    vocab = d.meta["vocab"]
    yt_idx = [i for i, it in enumerate(d.items) if it["kind"] == "yt"]
    out = {}
    for group in ("held_out", "missing"):
        for w in d.meta[group]:
            v = vocab.index(w)
            pos = [i for i in yt_idx if v in d.items[i]["labels"]]
            s, per = predict(model, d, pos, frames=True)
            order = np.argsort(-s[:, v])[:TOP_CLIPS]
            cands = []
            for k in order:
                a = per[k][v]
                t = int(a.argmax())
                lo = hi = t
                while lo > 0 and a[lo - 1] > 0.5 * a[t] and hi - lo + 1 < MAX_LEN:
                    lo -= 1
                while hi < len(a) - 1 and a[hi + 1] > 0.5 * a[t] and hi - lo + 1 < MAX_LEN:
                    hi += 1
                while hi - lo + 1 < MIN_LEN:
                    lo, hi = max(0, lo - 1), min(len(a) - 1, hi + 1)
                it = d.items[pos[k]]
                cands.append({"clip": it["id"], "split": it["split"], "clip_score": float(s[k, v]),
                              "peak": float(a[t]), "start": 2 * lo, "end": 2 * (hi + 1)})   # source frames (~30 fps)
            out[w] = {"group": group, "n_clips": len(pos), "candidates": cands}
        print(group, "done", flush=True)
    (MIL / "candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(len(out), "words ->", MIL / "candidates.json")


if __name__ == "__main__":
    train() if (sys.argv[1:2] or ["train"])[0] == "train" else spot()
