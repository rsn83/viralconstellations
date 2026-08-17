#!/usr/bin/env python
"""
62_ot_set2set.py

Pipeline
--------
1. OT PAIRING. Constellations are not identified across months -- the rows have
   no correspondence. Entropic optimal transport between month t and t+1, with
   cost = edit distance (symmetric difference) and marginals = sequence counts,
   supplies one. This is the standard device for trajectory inference from
   unpaired marginals (Waddington-OT, Schiebinger et al. 2019, for single cells,
   where the cell is destroyed when measured and the same problem arises).

2. TRAJECTORIES. Chain the pairings forward to get, for each source
   constellation, a sequence of sets across months. These are the analogue of a
   user's basket sequence in temporal set prediction.

3. SET2SET. Given k consecutive sets from a trajectory, predict the next j.
   Encoder pools learned label embeddings over the set; a GRU runs over the k
   steps; the decoder emits j multi-label predictions over the mutation space.

4. ROLLING ORIGIN. Train on windows entirely inside months <= t, predict
   t+1..t+j, advance, repeat.

The coupling is an ASSUMPTION, not an observation
-------------------------------------------------
Marginals do not determine the joint, so no pairing is identifiable from the
data. OT resolves it by minimising total edit distance, which encodes "genomes
change as little as possible between months". That is a modelling choice. It is
also consistent with the observed min-edit structure, but it is imposed, and
every trajectory below inherits it.

Evaluation
----------
Set level, against the OT-paired target:
    Jaccard, precision, recall, exact match
Population level, against the real month (no pairing involved, so this part is
assumption-free):
    vocabulary Jaccard, occupancy error
Baseline throughout: persistence, i.e. repeat the last observed set. Given that
levels have been random-walk everywhere in this project, persistence is the
number to beat and is expected to be strong.

Outputs
-------
outputs/62_set_level.csv      per origin, per horizon, set metrics
outputs/62_population.csv     per origin, per horizon, vocabulary and occupancy
outputs/62_summary.csv        pooled

Usage
-----
python scripts/62_ot_set2set.py --min_count 3 --end_month 2024-12 --k 6 --j 3
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


def top_sets(occ, max_sets):
    items = sorted(occ.items(), key=lambda kv: -kv[1])[:max_sets]
    sets = [c for c, _ in items]
    w = np.array([v for _, v in items], dtype=float)
    return sets, w / w.sum()


# ----------------------------------------------------------------------------
# 1. entropic OT
# ----------------------------------------------------------------------------

def edit_cost(sets_a, sets_b):
    """Symmetric-difference distance between every pair, via bit intersection."""
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
    inter = A @ B.T
    sa = A.sum(axis=1)[:, None]
    sb = B.sum(axis=1)[None, :]
    return sa + sb - 2.0 * inter          # |a| + |b| - 2|a n b|


def sinkhorn(C, a, b, reg, n_iter=300, tol=1e-7):
    K = np.exp(-C / max(reg, 1e-9))
    K = np.clip(K, 1e-300, None)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u_prev = u
        u = a / np.clip(K @ v, 1e-300, None)
        v = b / np.clip(K.T @ u, 1e-300, None)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]


def pair_months(sets_t, w_t, sets_n, w_n, reg):
    """Returns, for each source set, the index of its OT-assigned target."""
    C = edit_cost(sets_t, sets_n)
    C = C / max(C.max(), 1.0)
    P = sinkhorn(C, w_t, w_n, reg)
    return P.argmax(axis=1), C


# ----------------------------------------------------------------------------
# 2. trajectories
# ----------------------------------------------------------------------------

def build_trajectories(months_sets, pairings):
    """
    months_sets[j] : list of constellations at month j
    pairings[j]    : array mapping index at j to index at j+1
    Returns a list of trajectories, each a list of constellations, one per month.
    """
    T = len(months_sets)
    trajs = []
    for i0 in range(len(months_sets[0])):
        traj, i = [months_sets[0][i0]], i0
        ok = True
        for j in range(T - 1):
            if i >= len(pairings[j]):
                ok = False
                break
            i = int(pairings[j][i])
            traj.append(months_sets[j + 1][i])
        if ok and len(traj) == T:
            trajs.append(traj)
    return trajs


# ----------------------------------------------------------------------------
# 3. set2set model
# ----------------------------------------------------------------------------

class Set2Set(nn.Module):
    """Pool label embeddings over a set, GRU over k steps, j multi-label heads."""

    def __init__(self, n_labels, d=64, hidden=128, j=3):
        super().__init__()
        self.emb = nn.Embedding(n_labels, d)
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.step = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, n_labels)
        self.j = j

    def encode_sets(self, X):
        # X: B x k x V binary. mean-pool embeddings of present labels.
        w = X / X.sum(dim=2, keepdim=True).clamp(min=1.0)
        return w @ self.emb.weight              # B x k x d

    def forward(self, X):
        z = self.encode_sets(X)
        _, h = self.gru(z)
        h = h[0]
        outs = []
        for _ in range(self.j):
            h = self.step(h, h)
            outs.append(self.out(h))
        return torch.stack(outs, dim=1)         # B x j x V


def windows(trajs, lab_index, t_end, k, j):
    """All (k -> j) windows lying entirely within months 0..t_end."""
    V = len(lab_index)
    Xs, Ys = [], []
    for traj in trajs:
        for s in range(0, t_end + 1 - (k + j) + 1):
            xs = np.zeros((k, V), dtype=np.float32)
            ys = np.zeros((j, V), dtype=np.float32)
            for a in range(k):
                for l in traj[s + a]:
                    if l in lab_index:
                        xs[a, lab_index[l]] = 1.0
            for b in range(j):
                for l in traj[s + k + b]:
                    if l in lab_index:
                        ys[b, lab_index[l]] = 1.0
            Xs.append(xs)
            Ys.append(ys)
    if not Xs:
        return None, None
    return np.stack(Xs), np.stack(Ys)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def set_metrics(pred, true):
    """pred, true: boolean arrays over the label space."""
    inter = float(np.logical_and(pred, true).sum())
    union = float(np.logical_or(pred, true).sum())
    p = inter / pred.sum() if pred.sum() else 0.0
    r = inter / true.sum() if true.sum() else 0.0
    return {
        "jaccard": inter / union if union else 1.0,
        "precision": p, "recall": r,
        "exact": float(np.array_equal(pred, true)),
    }


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
    ap.add_argument("--max_sets", type=int, default=600)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--j", type=int, default=3)
    ap.add_argument("--reg", type=float, default=0.02, help="Sinkhorn entropy")
    ap.add_argument("--min_train", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    # ---- per-month set lists -------------------------------------------------
    msets, mw = [], []
    for m in names:
        s, w = top_sets(occ_by[m], args.max_sets)
        msets.append(s)
        mw.append(w)
    print(f"sets per month (capped at {args.max_sets}): "
          f"{min(len(s) for s in msets)}-{max(len(s) for s in msets)}")

    # ---- 1. OT pairings ------------------------------------------------------
    print("solving OT between consecutive months ...")
    pairings, mean_costs = [], []
    for t in range(T - 1):
        pi, C = pair_months(msets[t], mw[t], msets[t + 1], mw[t + 1], args.reg)
        pairings.append(pi)
        d = np.array([C[i, pi[i]] for i in range(len(pi))])
        mean_costs.append(float(d.mean()))
    print(f"mean normalised transport cost: {np.mean(mean_costs):.4f} "
          f"(0 = identical sets, 1 = maximally distant)")

    # ---- 2. trajectories -----------------------------------------------------
    trajs = build_trajectories(msets, pairings)
    print(f"trajectories spanning all {T} months: {len(trajs)}")
    if not trajs:
        raise SystemExit("no complete trajectories; raise --max_sets")

    # how much do sets actually move along a trajectory?
    steps = [len(traj[a] ^ traj[a + 1]) for traj in trajs for a in range(T - 1)]
    print(f"edit distance per trajectory step: mean {np.mean(steps):.2f}, "
          f"median {np.median(steps):.0f}, "
          f"share unchanged {np.mean(np.array(steps) == 0):.3f}")

    # ---- 3-4. rolling origin -------------------------------------------------
    all_labels = sorted({l for traj in trajs for s in traj for l in s}, key=str)
    lab_index = {l: i for i, l in enumerate(all_labels)}
    V = len(all_labels)
    print(f"label space: {V}\n")

    srows, prows = [], []

    for t in range(args.min_train, T - args.j):
        Xtr, Ytr = windows(trajs, lab_index, t, args.k, args.j)
        if Xtr is None or len(Xtr) < 64:
            continue

        model = Set2Set(V, d=args.dim, hidden=args.hidden, j=args.j)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(Xtr)
        Yt = torch.from_numpy(Ytr)
        n = len(Xt)
        model.train()
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for b in range(0, n, args.batch):
                sel = perm[b:b + args.batch]
                opt.zero_grad()
                loss = lossf(model(Xt[sel]), Yt[sel])
                loss.backward()
                opt.step()

        # test window: months t-k+1..t  ->  t+1..t+j
        Xte = np.zeros((len(trajs), args.k, V), dtype=np.float32)
        Yte = np.zeros((len(trajs), args.j, V), dtype=np.float32)
        for r, traj in enumerate(trajs):
            for a in range(args.k):
                for l in traj[t - args.k + 1 + a]:
                    Xte[r, a, lab_index[l]] = 1.0
            for b in range(args.j):
                for l in traj[t + 1 + b]:
                    Yte[r, b, lab_index[l]] = 1.0

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xte)).numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))

        last = Xte[:, -1, :] > 0.5          # persistence: repeat the last set
        for b in range(args.j):
            true = Yte[:, b, :] > 0.5
            # commit to a set by taking as many labels as the last set had
            pred = np.zeros_like(true)
            for r in range(len(trajs)):
                nkeep = max(int(last[r].sum()), 1)
                top = np.argpartition(-prob[r, b], nkeep - 1)[:nkeep]
                pred[r, top] = True

            for nm, P in [("set2set", pred), ("persistence", last)]:
                mm = [set_metrics(P[r], true[r]) for r in range(len(trajs))]
                srows.append({
                    "origin": names[t], "target": names[t + 1 + b],
                    "h": b + 1, "model": nm,
                    "jaccard": float(np.mean([x["jaccard"] for x in mm])),
                    "precision": float(np.mean([x["precision"] for x in mm])),
                    "recall": float(np.mean([x["recall"] for x in mm])),
                    "exact": float(np.mean([x["exact"] for x in mm])),
                    "n_traj": len(trajs),
                })

            # population level: no pairing involved, so assumption-free
            true_vocab = {l for cs in occ_by[names[t + 1 + b]] for l in cs}
            true_vocab &= set(all_labels)
            true_occ = float(np.mean([len(cs) for cs in msets[t + 1 + b]]))
            for nm, P in [("set2set", pred), ("persistence", last)]:
                pv = {all_labels[i] for i in np.flatnonzero(P.any(axis=0))}
                inter = len(pv & true_vocab)
                union = len(pv | true_vocab)
                prows.append({
                    "origin": names[t], "target": names[t + 1 + b],
                    "h": b + 1, "model": nm,
                    "vocab_jaccard": inter / union if union else np.nan,
                    "pred_vocab": len(pv), "true_vocab": len(true_vocab),
                    "pred_occupancy": float(P.sum(axis=1).mean()),
                    "true_occupancy": true_occ,
                })

        print(f"  origin {names[t]}: trained on {n} windows")

    sdf = pd.DataFrame(srows)
    pdf = pd.DataFrame(prows)
    sdf.to_csv(f"{args.out_dir}/62_set_level.csv", index=False)
    pdf.to_csv(f"{args.out_dir}/62_population.csv", index=False)

    print("\n" + "=" * 74)
    print("SET LEVEL  (against the OT-paired target -- inherits the coupling)")
    print("=" * 74)
    g = sdf.groupby(["model", "h"]).agg(
        jaccard=("jaccard", "mean"), precision=("precision", "mean"),
        recall=("recall", "mean"), exact=("exact", "mean"),
        origins=("jaccard", "count"),
    ).reset_index().sort_values(["h", "jaccard"], ascending=[True, False])
    print(g.round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("POPULATION LEVEL  (against the real month -- no coupling assumed)")
    print("=" * 74)
    g2 = pdf.groupby(["model", "h"]).agg(
        vocab_jaccard=("vocab_jaccard", "mean"),
        pred_vocab=("pred_vocab", "mean"), true_vocab=("true_vocab", "mean"),
        pred_occupancy=("pred_occupancy", "mean"),
        true_occupancy=("true_occupancy", "mean"),
        origins=("vocab_jaccard", "count"),
    ).reset_index().sort_values(["h", "vocab_jaccard"], ascending=[True, False])
    print(g2.round(4).to_string(index=False))

    g.to_csv(f"{args.out_dir}/62_summary.csv", index=False)
    print(f"\nwrote 3 files to {args.out_dir}/")
    print("\nThe population-level table is the one that answers the original")
    print("question: given k months, what does the vocabulary and the occupancy")
    print("look like j months out. It does not depend on the OT coupling.")


if __name__ == "__main__":
    main()
