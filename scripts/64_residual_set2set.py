#!/usr/bin/env python
"""
64_residual_set2set.py

Why this replaces script 63
---------------------------
In 63 the decoder saw only a hidden vector. To emit a constellation it had to
reconstruct all ~30 mutations from scratch, and under BCE over 25,460 outputs
with ~30 positives the loss is minimised by emitting the most common cells. The
result was occupancy pinned at ~29 for every month and every horizon while the
truth climbed 31 -> 54, and set_jaccard of exactly 0. It was not predicting
badly; it was emitting the training mean.

Three changes:

1. RESIDUAL DECODER. The output logit for a label is

       logit = persist * x_current + correction(h, x_current)

   with `persist` a learned scalar initialised high. At initialisation the model
   reproduces PERSISTENCE exactly, so training starts from the baseline and can
   only learn corrections to it -- which mutations to add, which to drop. It
   never has to rebuild the set.

2. AUTOREGRESSIVE ROLLOUT. After predicting step b, the predicted set becomes
   the current set for step b+1, so occupancy is free to drift across the
   horizon instead of being fixed.

3. SAMPLED OT COUPLING. 63 took argmax of each transport row, which funnelled
   thousands of source rows onto ~57 targets and destroyed the branching that
   forward evolution actually has. Here each chain samples its next state from
   the transport row, so a source with mass spread over several targets produces
   several distinct continuations.

Also fixed: the label space is now built from TRAINING months only. In 63 it was
built from every month including the test period, which is lookahead. Cells that
first appear in the test period are counted and reported as unreachable rather
than quietly included.

Evaluation
----------
SET LEVEL      against the OT-paired target -- inherits the coupling assumption.
POPULATION     against the REAL month: vocabulary Jaccard, constellation
               Jaccard, occupancy. No pairing involved, so this is the honest
               number.
Baseline is persistence throughout. The model is initialised AT persistence, so
any gap is something it learned, and any loss is something it broke.

Outputs
-------
outputs/64_diagnostics.csv
outputs/64_set_level.csv
outputs/64_population.csv

Usage
-----
python scripts/64_residual_set2set.py --min_count 3 --end_month 2024-12 \
    --depth 5000 --n_in 4 --n_out 3 --train_months 30
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("this script needs pytorch: pip install torch")

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_months(data_dir, min_count, start_month=None, end_month=None):
    files = []
    for fn in os.listdir(data_dir):
        m = MONTH_RE.search(fn)
        if m:
            files.append((m.group(1), os.path.join(data_dir, fn)))
    files.sort()
    out = []
    for month, path in files:
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        with open(path, "rb") as f:
            occ = pickle.load(f)
        occ = {k: v for k, v in occ.items() if v >= min_count}
        if occ:
            out.append((month, occ))
    return out


def rarefy_rows(occ, depth, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if counts.sum() < depth:
        return None, None, None
    draws = rng.multinomial(depth, counts / counts.sum())
    nz = np.flatnonzero(draws)
    return ([keys[i] for i in nz], draws[nz].astype(float),
            np.repeat(np.arange(len(nz)), draws[nz]))


# ----------------------------------------------------------------------------
# OT
# ----------------------------------------------------------------------------

def edit_cost(sets_a, sets_b):
    labs = sorted({l for s in sets_a for l in s} | {l for s in sets_b for l in s},
                  key=str)
    idx = {l: i for i, l in enumerate(labs)}
    A = np.zeros((len(sets_a), len(labs)), dtype=np.float32)
    B = np.zeros((len(sets_b), len(labs)), dtype=np.float32)
    for i, s in enumerate(sets_a):
        for l in s:
            A[i, idx[l]] = 1.0
    for i, s in enumerate(sets_b):
        for l in s:
            B[i, idx[l]] = 1.0
    return A.sum(1)[:, None] + B.sum(1)[None, :] - 2.0 * (A @ B.T)


def sinkhorn(C, a, b, reg, n_iter=400, tol=1e-8):
    K = np.clip(np.exp(-C / max(reg, 1e-9)), 1e-300, None)
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(n_iter):
        u_prev = u
        u = a / np.clip(K @ v, 1e-300, None)
        v = b / np.clip(K.T @ u, 1e-300, None)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]


def topk_rows(P, k):
    """Keep the k largest targets per source row, renormalised. Enables
    sampling without storing the full dense plan for every month pair."""
    n, m = P.shape
    k = min(k, m)
    idx = np.argpartition(-P, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(P, idx, axis=1)
    val = val / np.clip(val.sum(axis=1, keepdims=True), 1e-300, None)
    return idx, np.cumsum(val, axis=1)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class ResidualSet2Set(nn.Module):
    """
    Encoder: GRU over pooled label embeddings of the input sets.
    Decoder: at each step the logit for a label is

        persist * x_current  +  correction(h, x_current)

    so at initialisation the model IS persistence and learns only corrections.
    """

    def __init__(self, n_labels, d=64, hidden=128, n_out=3, persist_init=4.0):
        super().__init__()
        self.emb = nn.Embedding(n_labels, d)
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.step = nn.GRUCell(d, hidden)
        self.corr = nn.Linear(hidden, n_labels)
        self.persist = nn.Parameter(torch.tensor(float(persist_init)))
        self.bias = nn.Parameter(torch.full((n_labels,), -float(persist_init) / 2))
        self.n_out = n_out
        nn.init.zeros_(self.corr.weight)
        nn.init.zeros_(self.corr.bias)

    def pool(self, X):
        w = X / X.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return w @ self.emb.weight

    def forward(self, X, hard_rollout=False):
        # X: B x n_in x V
        _, h = self.gru(self.pool(X))
        h = h[0]
        cur = X[:, -1, :]
        outs = []
        for _ in range(self.n_out):
            h = self.step(self.pool(cur), h)
            logit = self.persist * cur + self.corr(h) + self.bias
            outs.append(logit)
            nxt = torch.sigmoid(logit)
            cur = (nxt > 0.5).float() if hard_rollout else nxt
        return torch.stack(outs, dim=1)


def set_metrics(pred, true):
    inter = float(np.logical_and(pred, true).sum())
    union = float(np.logical_or(pred, true).sum())
    return (inter / union if union else 1.0,
            inter / pred.sum() if pred.sum() else 0.0,
            inter / true.sum() if true.sum() else 0.0,
            float(np.array_equal(pred, true)))


def jaccard(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--depth", type=int, default=5000)
    ap.add_argument("--n_in", type=int, default=4)
    ap.add_argument("--n_out", type=int, default=3)
    ap.add_argument("--train_months", type=int, default=30)
    ap.add_argument("--reg", type=float, default=0.02)
    ap.add_argument("--ot_topk", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max_train", type=int, default=150000)
    ap.add_argument("--persist_init", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    L = args.n_in + args.n_out

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    print(f"loaded {len(months)} months: {months[0][0]} .. {months[-1][0]}")

    names, msets, mw, mrows = [], [], [], []
    for month, occ in months:
        s, w, rows = rarefy_rows(occ, args.depth, rng)
        if s is None:
            continue
        names.append(month)
        msets.append(s)
        mw.append(w / w.sum())
        mrows.append(rows)
    T = len(names)
    print(f"months at depth {args.depth}: {T} ({names[0]} .. {names[-1]})")

    # ---- OT, one solve per pair, keep top-k targets per source --------------
    print(f"\nsolving OT for {T-1} month pairs ...")
    tidx, tcum, diag = [], [], []
    for t in range(T - 1):
        C = edit_cost(msets[t], msets[t + 1])
        P = sinkhorn(C / max(C.max(), 1.0), mw[t], mw[t + 1], args.reg)
        idx, cum = topk_rows(P, args.ot_topk)
        tidx.append(idx)
        tcum.append(cum)
        am = P.argmax(axis=1)
        d = np.array([C[i, am[i]] for i in range(len(am))])
        diag.append({
            "from": names[t], "to": names[t + 1],
            "n_from": len(msets[t]), "n_to": len(msets[t + 1]),
            "mean_edit_step": float(d.mean()),
            "frac_self_mapping": float(np.mean(
                [msets[t][i] == msets[t + 1][am[i]] for i in range(len(am))])),
            "argmax_distinct_targets": int(len(np.unique(am))),
            "sampled_distinct_targets": int(len(np.unique(idx))),
        })
    dg = pd.DataFrame(diag)
    dg.to_csv(f"{args.out_dir}/64_diagnostics.csv", index=False)
    print(dg.round(3).to_string(index=False))
    print(f"\nmean edit per step {dg['mean_edit_step'].mean():.2f}   "
          f"self-mapping {dg['frac_self_mapping'].mean():.3f}")
    print(f"distinct targets: argmax {dg['argmax_distinct_targets'].mean():.0f}"
          f" -> sampled {dg['sampled_distinct_targets'].mean():.0f}")
    print("  sampling from the transport row instead of taking argmax is what")
    print("  restores branching; one source can continue several ways.")

    # ---- label space from TRAINING months only -----------------------------
    n_starts = T - L + 1
    train_last = min(args.train_months + L - 1, T)
    all_labels = sorted({l for j in range(train_last)
                         for cs in msets[j] for l in cs}, key=str)
    lab_index = {l: i for i, l in enumerate(all_labels)}
    V = len(all_labels)
    later = {l for j in range(train_last, T) for cs in msets[j] for l in cs}
    print(f"\nlabel space from months 0..{train_last-1}: {V}")
    print(f"cells first appearing after that: {len(later - set(all_labels))} "
          "(unreachable by construction, reported not hidden)")

    def chain(start, row):
        i = mrows[start][row]
        out = [i]
        for step in range(L - 1):
            r = rng.random()
            k = int(np.searchsorted(tcum[start + step][i], r))
            k = min(k, tidx[start + step].shape[1] - 1)
            i = int(tidx[start + step][i, k])
            out.append(i)
        return out

    print(f"chain length {L} = {args.n_in} in + {args.n_out} out; "
          f"{n_starts} starts x {args.depth} chains")

    train_X, train_Y, test_sets = [], [], {}
    for start in range(n_starts):
        X = np.zeros((args.depth, args.n_in, V), dtype=np.float32)
        Y = np.zeros((args.depth, args.n_out, V), dtype=np.float32)
        for r in range(args.depth):
            ch = chain(start, r)
            for a in range(args.n_in):
                for l in msets[start + a][ch[a]]:
                    if l in lab_index:
                        X[r, a, lab_index[l]] = 1.0
            for b in range(args.n_out):
                for l in msets[start + args.n_in + b][ch[args.n_in + b]]:
                    if l in lab_index:
                        Y[r, b, lab_index[l]] = 1.0
        if start < args.train_months:
            train_X.append(X)
            train_Y.append(Y)
        else:
            test_sets[start] = (X, Y)

    Xtr = np.concatenate(train_X)
    Ytr = np.concatenate(train_Y)
    if len(Xtr) > args.max_train:
        sel = rng.choice(len(Xtr), size=args.max_train, replace=False)
        Xtr, Ytr = Xtr[sel], Ytr[sel]
    print(f"training chains: {len(Xtr)}   test starts: "
          f"{[names[s] for s in sorted(test_sets)]}")

    # ---- train --------------------------------------------------------------
    model = ResidualSet2Set(V, d=args.dim, hidden=args.hidden,
                            n_out=args.n_out, persist_init=args.persist_init)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    n = len(Xt)

    with torch.no_grad():
        init_loss = float(lossf(model(Xt[:2048]), Yt[:2048]))
    print(f"\nloss at initialisation (= persistence): {init_loss:.5f}")

    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, args.batch):
            sel = perm[b:b + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xt[sel]), Yt[sel])
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)
        print(f"  epoch {ep+1}/{args.epochs}  loss {tot/n:.5f}  "
              f"persist {float(model.persist):.2f}")

    # ---- test ---------------------------------------------------------------
    srows, prows = [], []
    model.eval()
    for start in sorted(test_sets):
        X, Y = test_sets[start]
        with torch.no_grad():
            logits = model(torch.from_numpy(X), hard_rollout=True).numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        last = X[:, -1, :] > 0.5

        for b in range(args.n_out):
            m = start + args.n_in + b
            pred = prob[:, b, :] > 0.5
            true = Y[:, b, :] > 0.5

            tv = np.zeros(V, dtype=bool)
            for cs in msets[m]:
                for l in cs:
                    if l in lab_index:
                        tv[lab_index[l]] = True
            true_sets = {frozenset(lab_index[l] for l in cs if l in lab_index)
                         for cs in msets[m]}

            for nm, P in [("set2set", pred), ("persistence", last)]:
                mm = np.array([set_metrics(P[r], true[r]) for r in range(len(P))])
                srows.append({
                    "start_month": names[start], "target_month": names[m],
                    "h": b + 1, "model": nm,
                    "jaccard": float(mm[:, 0].mean()),
                    "precision": float(mm[:, 1].mean()),
                    "recall": float(mm[:, 2].mean()),
                    "exact": float(mm[:, 3].mean()),
                })
                pv = P.any(axis=0)
                pred_sets = {frozenset(np.flatnonzero(P[r])) for r in range(len(P))}
                prows.append({
                    "start_month": names[start], "target_month": names[m],
                    "h": b + 1, "model": nm,
                    "vocab_jaccard": jaccard(pv, tv),
                    "pred_vocab": int(pv.sum()), "true_vocab": int(tv.sum()),
                    "set_jaccard": (len(pred_sets & true_sets) /
                                    len(pred_sets | true_sets)
                                    if (pred_sets | true_sets) else np.nan),
                    "n_pred_sets": len(pred_sets), "n_true_sets": len(true_sets),
                    "pred_occupancy": float(P.sum(1).mean()),
                    "true_occupancy": float(np.mean([len(cs) for cs in msets[m]])),
                })

    sdf = pd.DataFrame(srows)
    pdf = pd.DataFrame(prows)
    sdf.to_csv(f"{args.out_dir}/64_set_level.csv", index=False)
    pdf.to_csv(f"{args.out_dir}/64_population.csv", index=False)

    print("\n" + "=" * 74)
    print("SET LEVEL  (vs the OT-paired target)")
    print("=" * 74)
    print(sdf.groupby(["model", "h"])[
        ["jaccard", "precision", "recall", "exact"]].mean().round(4).to_string())

    print("\n" + "=" * 74)
    print("POPULATION LEVEL  (vs the REAL month -- no coupling assumed)")
    print("=" * 74)
    print(pdf.groupby(["model", "h"])[
        ["vocab_jaccard", "set_jaccard", "pred_vocab", "true_vocab",
         "pred_occupancy", "true_occupancy"]].mean().round(4).to_string())

    print("\noccupancy tracking (the failure mode in 63):")
    piv = pdf.pivot_table(index="target_month", columns="model",
                          values="pred_occupancy", aggfunc="mean")
    piv["true"] = pdf.groupby("target_month")["true_occupancy"].mean()
    print(piv.round(2).to_string())
    print("  in 63 set2set sat at ~29 while truth climbed 31 -> 54. If the")
    print("  set2set column tracks the true column here, the residual decoder")
    print("  fixed it.")

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
