#!/usr/bin/env python
"""
63_ot_chains_set2set.py  (v2: 4 sets in -> 3 sets out)

Pipeline
--------
1. Rarefy every month to a fixed number of rows (default 5000).
2. Solve entropic OT ONCE per consecutive month pair. Cost is edit distance
   between constellations, marginals are sequence counts. Each row in month t
   is mapped to a row in month t+1.
   The solve runs on distinct constellations with weights -- mathematically the
   same as solving on all 5000 rows, just far cheaper -- then expands back.
3. Follow the pointers to read chains of length 7 starting at EVERY month.
   Month 0 gives 5000 chains, month 1 gives another 5000, and so on.
4. Set2set: sets 1-4 of a chain are the input, sets 5, 6, 7 are the targets.
   Train on chains starting in the first `train_months`; test on the rest.

The output space is the full reference grid, 1273 positions x 20 residues =
25,460 cells. That is known before any data is collected, so it carries no
lookahead. Cells never observed simply never activate.

Two scores, and they differ in what they assume
-----------------------------------------------
SET LEVEL      compares the predicted set against what OT said that row becomes.
               If the coupling is wrong, this scores against a fabricated
               answer.
POPULATION     pools all 5000 predicted sets and compares against the REAL
               month -- its vocabulary, its occupancy, its constellations.
               No pairing involved, so OT's assumptions do not affect it.
               This is the honest number.

Both report Jaccard. Baseline is persistence: repeat the last input set.
The model predicts its own set size; occupancy is read off what it outputs and
is never given to it.

Outputs
-------
outputs/63_diagnostics.csv   per month pair: transport cost, self-mapping rate
outputs/63_set_level.csv     Jaccard / precision / recall on held-out chains
outputs/63_population.csv    vocabulary Jaccard and occupancy vs the real month

Usage
-----
python scripts/63_ot_chains_set2set.py --min_count 3 --end_month 2024-12 \
    --depth 5000 --n_in 4 --n_out 3 --train_months 30
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

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
    sets = [keys[i] for i in nz]
    w = draws[nz].astype(float)
    rows = np.repeat(np.arange(len(nz)), draws[nz])
    return sets, w, rows


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


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class Set2Set(nn.Module):
    """Pool label embeddings per set, GRU over the input sets, emit n_out sets."""

    def __init__(self, n_labels, d=64, hidden=128, n_out=3):
        super().__init__()
        self.emb = nn.Embedding(n_labels, d)
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.step = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, n_labels)
        self.n_out = n_out

    def forward(self, X):
        w = X / X.sum(dim=2, keepdim=True).clamp(min=1.0)
        _, h = self.gru(w @ self.emb.weight)
        h = h[0]
        outs = []
        for _ in range(self.n_out):
            h = self.step(h, h)
            outs.append(self.out(h))
        return torch.stack(outs, dim=1)          # B x n_out x V


def jaccard(pred, true):
    inter = float(np.logical_and(pred, true).sum())
    union = float(np.logical_or(pred, true).sum())
    return inter / union if union else 1.0


def set_metrics(pred, true):
    inter = float(np.logical_and(pred, true).sum())
    union = float(np.logical_or(pred, true).sum())
    return (inter / union if union else 1.0,
            inter / pred.sum() if pred.sum() else 0.0,
            inter / true.sum() if true.sum() else 0.0,
            float(np.array_equal(pred, true)))


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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max_train", type=int, default=150000)
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
    print(f"distinct constellations per month: "
          f"{min(len(s) for s in msets)}-{max(len(s) for s in msets)}")

    # ---- one OT solve per consecutive pair ---------------------------------
    print(f"\nsolving OT for {T-1} month pairs ...")
    ptr, diag = [], []
    for t in range(T - 1):
        C = edit_cost(msets[t], msets[t + 1])
        P = sinkhorn(C / max(C.max(), 1.0), mw[t], mw[t + 1], args.reg)
        am = P.argmax(axis=1)
        ptr.append(am)
        d = np.array([C[i, am[i]] for i in range(len(am))])
        same = np.mean([msets[t][i] == msets[t + 1][am[i]]
                        for i in range(len(am))])
        diag.append({
            "from": names[t], "to": names[t + 1],
            "n_from": len(msets[t]), "n_to": len(msets[t + 1]),
            "mean_edit_step": float(d.mean()),
            "median_edit_step": float(np.median(d)),
            "frac_self_mapping": float(same),
            "n_distinct_targets": int(len(np.unique(am))),
        })
    dg = pd.DataFrame(diag)
    dg.to_csv(f"{args.out_dir}/63_diagnostics.csv", index=False)
    print(dg.round(3).to_string(index=False))
    print(f"\nmean edit distance per step   : {dg['mean_edit_step'].mean():.2f}")
    print(f"fraction of self-mappings     : {dg['frac_self_mapping'].mean():.3f}")
    print(f"mean distinct targets per pair: {dg['n_distinct_targets'].mean():.0f}")
    print("  high self-mapping means the coupling is near-identity, so the")
    print("  chains mostly repeat the same set and copying wins by default.")

    # ---- label space: the full reference grid ------------------------------
    all_labels = sorted({l for s in msets for cs in s for l in cs}, key=str)
    lab_index = {l: i for i, l in enumerate(all_labels)}
    V = len(all_labels)
    print(f"\nlabel space (cells occupied anywhere in the series): {V}")

    def chain_from(start, row):
        i = mrows[start][row]
        out = [i]
        for step in range(L - 1):
            i = int(ptr[start + step][i])
            out.append(i)
        return out

    n_starts = T - L + 1
    print(f"chain length {L} = {args.n_in} in + {args.n_out} out; "
          f"{n_starts} start months x {args.depth} chains")

    train_X, train_Y, test_sets = [], [], {}
    for start in range(n_starts):
        X = np.zeros((args.depth, args.n_in, V), dtype=np.float32)
        Y = np.zeros((args.depth, args.n_out, V), dtype=np.float32)
        for r in range(args.depth):
            ch = chain_from(start, r)
            for a in range(args.n_in):
                for l in msets[start + a][ch[a]]:
                    X[r, a, lab_index[l]] = 1.0
            for b in range(args.n_out):
                m = start + args.n_in + b
                for l in msets[m][ch[args.n_in + b]]:
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
    model = Set2Set(V, d=args.dim, hidden=args.hidden, n_out=args.n_out)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    n = len(Xt)
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
        print(f"  epoch {ep+1}/{args.epochs}  loss {tot/n:.5f}")

    # ---- test ---------------------------------------------------------------
    srows, prows = [], []
    model.eval()
    for start in sorted(test_sets):
        X, Y = test_sets[start]
        with torch.no_grad():
            logits = model(torch.from_numpy(X)).numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        last = X[:, -1, :] > 0.5

        for b in range(args.n_out):
            m = start + args.n_in + b
            pred = prob[:, b, :] > 0.5          # model chooses its own size
            true = Y[:, b, :] > 0.5

            for nm, P in [("set2set", pred), ("persistence", last)]:
                mm = np.array([set_metrics(P[r], true[r])
                               for r in range(len(P))])
                srows.append({
                    "start_month": names[start], "target_month": names[m],
                    "h": b + 1, "model": nm,
                    "jaccard": float(mm[:, 0].mean()),
                    "precision": float(mm[:, 1].mean()),
                    "recall": float(mm[:, 2].mean()),
                    "exact": float(mm[:, 3].mean()),
                    "mean_pred_size": float(P.sum(1).mean()),
                    "mean_true_size": float(true.sum(1).mean()),
                })

                # population level: pooled prediction vs the REAL month
                pv = np.zeros(V, dtype=bool)
                pv[np.flatnonzero(P.any(axis=0))] = True
                tv = np.zeros(V, dtype=bool)
                for cs in msets[m]:
                    for l in cs:
                        tv[lab_index[l]] = True
                # constellation-level overlap, count-weighted on the truth side
                pred_sets = {frozenset(np.flatnonzero(P[r])) for r in range(len(P))}
                true_sets = {frozenset(lab_index[l] for l in cs) for cs in msets[m]}
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
                    "true_occupancy": float(
                        np.mean([len(cs) for cs in msets[m]])),
                })

    sdf = pd.DataFrame(srows)
    pdf = pd.DataFrame(prows)
    sdf.to_csv(f"{args.out_dir}/63_set_level.csv", index=False)
    pdf.to_csv(f"{args.out_dir}/63_population.csv", index=False)

    print("\n" + "=" * 74)
    print("SET LEVEL  (vs the OT-paired target -- inherits the coupling)")
    print("=" * 74)
    print(sdf.groupby(["model", "h"])[
        ["jaccard", "precision", "recall", "exact",
         "mean_pred_size", "mean_true_size"]].mean().round(4).to_string())

    print("\n" + "=" * 74)
    print("POPULATION LEVEL  (vs the REAL month -- no coupling assumed)")
    print("=" * 74)
    print(pdf.groupby(["model", "h"])[
        ["vocab_jaccard", "pred_vocab", "true_vocab", "set_jaccard",
         "n_pred_sets", "n_true_sets", "pred_occupancy",
         "true_occupancy"]].mean().round(4).to_string())

    print("\nper test month, population level:")
    print(pdf[["start_month", "target_month", "h", "model", "vocab_jaccard",
               "set_jaccard", "pred_occupancy", "true_occupancy"]]
          .round(4).to_string(index=False))

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
