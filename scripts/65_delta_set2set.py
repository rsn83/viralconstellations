#!/usr/bin/env python
"""
65_delta_set2set.py

What changed from 64, and why
-----------------------------
Sets are NOT mostly identical -- self-mapping is 12% and the mean step is 4.83
edits. But a set carries ~43 mutations, so changing 5 of 43 still leaves
Jaccard at 0.93. Persistence wins because the unchanged 90% dominates the
metric, not because nothing happens. The target is which ~5 change.

Four changes:

1. POPULATION CONTEXT. In 64 each chain was modelled in isolation -- one row's
   own history. But 85-95% of new constellations add a mutation ALREADY
   CIRCULATING ELSEWHERE, and a per-row model cannot see elsewhere. The current
   label-frequency vector rho_t is now an input to the decoder. This is the
   change most directly motivated by what the data says.

2. DELTA HEADS. Two heads instead of one: P(add label | not present) and
   P(drop label | present). Presence prediction is swamped by persistence;
   add/drop is not. pos_weight handles the imbalance so the model spends
   capacity on the labels that move.

3. MIN AND MAX POOLING. Script 58 found pmi_min -- the WORST-matching label in
   the set -- was the strongest co-occurrence feature. Mean pooling averages
   exactly that away. The set encoder now concatenates mean, max and min pools.

4. DELTA SCORING. Jaccard on the whole set is won by copying. The headline
   metrics here are average precision on the additions and on the drops, where
   persistence scores at the base rate by construction and the metric measures
   what is actually being asked.

Set-level Jaccard and population-level vocabulary/occupancy are still reported
so the numbers stay comparable with 63 and 64, but they are not the target.

Evaluation
----------
ADD    among labels absent from the current set, which get added?
DROP   among labels present in the current set, which get dropped?
       AP and lift over base rate. Persistence cannot score above base rate on
       either -- it predicts no change at all.
Baselines: persistence, and marginal (rank candidates by population frequency
rho_t alone, no model). Marginal is the null the co-occurrence term must beat,
the same null used in script 58.

Outputs
-------
outputs/65_delta.csv       AP and lift for add and drop, per test month
outputs/65_set_level.csv   Jaccard etc, for comparison with 63/64
outputs/65_population.csv  vocabulary and occupancy vs the real month

Usage
-----
python scripts/65_delta_set2set.py --min_count 3 --end_month 2024-12 \
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
    n, m = P.shape
    k = min(k, m)
    idx = np.argpartition(-P, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(P, idx, axis=1)
    val = val / np.clip(val.sum(axis=1, keepdims=True), 1e-300, None)
    return idx, np.cumsum(val, axis=1)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class DeltaSet2Set(nn.Module):
    """
    Encoder : GRU over set encodings (mean + max + min pooled embeddings).
    Context : the population label-frequency vector rho_t, projected.
    Decoder : two heads over the label space,
                add_logit  -- masked to labels absent from the current set
                drop_logit -- masked to labels present in the current set
    The next set is current + sampled adds - sampled drops, rolled forward.
    """

    def __init__(self, n_labels, d=64, hidden=128, n_out=3):
        super().__init__()
        self.emb = nn.Embedding(n_labels, d)
        self.enc_proj = nn.Linear(3 * d, d)
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.ctx = nn.Sequential(nn.Linear(n_labels, hidden), nn.ReLU())
        self.step = nn.GRUCell(d, hidden)
        self.mix = nn.Linear(2 * hidden, hidden)
        self.add_head = nn.Linear(hidden, n_labels)
        self.drop_head = nn.Linear(hidden, n_labels)
        self.n_out = n_out

    def pool(self, X):
        """X: (..., V) binary -> (..., d) via mean, max and min over members."""
        E = self.emb.weight                       # V x d
        cnt = X.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (X @ E) / cnt
        big = 1e4
        mask = (1.0 - X).unsqueeze(-1) * big
        Ex = E.unsqueeze(0).unsqueeze(0) if X.dim() == 3 else E.unsqueeze(0)
        mx = (Ex - mask).max(dim=-2).values
        mn = (Ex + mask).min(dim=-2).values
        return self.enc_proj(torch.cat([mean, mx, mn], dim=-1))

    def forward(self, X, rho, hard=False):
        # X: B x n_in x V   rho: B x V
        _, h = self.gru(self.pool(X))
        h = h[0]
        c = self.ctx(rho)
        cur = X[:, -1, :]
        adds, drops, sets = [], [], []
        for _ in range(self.n_out):
            h = self.step(self.pool(cur), h)
            z = torch.relu(self.mix(torch.cat([h, c], dim=-1)))
            a_log = self.add_head(z)
            d_log = self.drop_head(z)
            adds.append(a_log)
            drops.append(d_log)
            pa = torch.sigmoid(a_log) * (1.0 - cur)
            pd_ = torch.sigmoid(d_log) * cur
            nxt = cur + pa - pd_
            cur = (nxt > 0.5).float() if hard else nxt.clamp(0.0, 1.0)
            sets.append(cur)
        return (torch.stack(adds, 1), torch.stack(drops, 1),
                torch.stack(sets, 1))


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, y.size + 1)) * yy).sum() / y.sum())


def set_metrics(pred, true):
    inter = float(np.logical_and(pred, true).sum())
    union = float(np.logical_or(pred, true).sum())
    return (inter / union if union else 1.0,
            inter / pred.sum() if pred.sum() else 0.0,
            inter / true.sum() if true.sum() else 0.0)


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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--pos_weight", type=float, default=20.0)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    L = args.n_in + args.n_out

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    print(f"loaded {len(months)} months")

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

    print(f"solving OT for {T-1} pairs ...")
    tidx, tcum = [], []
    for t in range(T - 1):
        C = edit_cost(msets[t], msets[t + 1])
        P = sinkhorn(C / max(C.max(), 1.0), mw[t], mw[t + 1], args.reg)
        idx, cum = topk_rows(P, args.ot_topk)
        tidx.append(idx)
        tcum.append(cum)

    n_starts = T - L + 1
    train_last = min(args.train_months + L - 1, T)
    all_labels = sorted({l for j in range(train_last)
                         for cs in msets[j] for l in cs}, key=str)
    lab_index = {l: i for i, l in enumerate(all_labels)}
    V = len(all_labels)
    print(f"label space from months 0..{train_last-1}: {V}")

    # population frequency vector per month, causal by construction
    rho = np.zeros((T, V), dtype=np.float32)
    for j in range(T):
        tot = sum(len(cs) for cs in msets[j]) or 1
        cnt = np.zeros(V, dtype=np.float32)
        for cs in msets[j]:
            for l in cs:
                if l in lab_index:
                    cnt[lab_index[l]] += 1.0
        rho[j] = cnt / max(len(msets[j]), 1)

    def chain(start, row):
        i = mrows[start][row]
        out = [i]
        for step in range(L - 1):
            r = rng.random()
            k = min(int(np.searchsorted(tcum[start + step][i], r)),
                    tidx[start + step].shape[1] - 1)
            i = int(tidx[start + step][i, k])
            out.append(i)
        return out

    def encode_window(start, n_rows):
        X = np.zeros((n_rows, args.n_in, V), dtype=np.float32)
        Y = np.zeros((n_rows, args.n_out, V), dtype=np.float32)
        for r in range(n_rows):
            ch = chain(start, r)
            for a in range(args.n_in):
                for l in msets[start + a][ch[a]]:
                    if l in lab_index:
                        X[r, a, lab_index[l]] = 1.0
            for b in range(args.n_out):
                for l in msets[start + args.n_in + b][ch[args.n_in + b]]:
                    if l in lab_index:
                        Y[r, b, lab_index[l]] = 1.0
        R = np.repeat(rho[start + args.n_in - 1][None, :], n_rows, axis=0)
        return X, Y, R

    per_start = max(args.max_train // max(args.train_months, 1), 200)
    train_X, train_Y, train_R, test_sets = [], [], [], {}
    for start in range(n_starts):
        if start < args.train_months:
            X, Y, R = encode_window(start, min(per_start, args.depth))
            train_X.append(X)
            train_Y.append(Y)
            train_R.append(R)
        else:
            test_sets[start] = encode_window(start, args.depth)

    Xtr = np.concatenate(train_X)
    Ytr = np.concatenate(train_Y)
    Rtr = np.concatenate(train_R)
    print(f"training chains: {len(Xtr)}  test starts: {len(test_sets)}")

    # ---- train --------------------------------------------------------------
    model = DeltaSet2Set(V, d=args.dim, hidden=args.hidden, n_out=args.n_out)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pw = torch.tensor(args.pos_weight)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    Xt = torch.from_numpy(Xtr)
    Yt = torch.from_numpy(Ytr)
    Rt = torch.from_numpy(Rtr)
    n = len(Xt)

    # targets: what was added and what was dropped, relative to the input set
    prev = torch.cat([Xt[:, -1:, :], Yt[:, :-1, :]], dim=1)
    add_t = ((Yt > 0.5) & (prev < 0.5)).float()
    drop_t = ((Yt < 0.5) & (prev > 0.5)).float()
    add_mask = (prev < 0.5).float()
    drop_mask = (prev > 0.5).float()
    print(f"mean adds per step {float(add_t.sum(-1).mean()):.2f}, "
          f"drops {float(drop_t.sum(-1).mean()):.2f}, set size "
          f"{float(Yt.sum(-1).mean()):.1f}")

    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, args.batch):
            sel = perm[b:b + args.batch]
            opt.zero_grad()
            a_log, d_log, _ = model(Xt[sel], Rt[sel])
            la = (bce(a_log, add_t[sel]) * add_mask[sel]).sum() / \
                 add_mask[sel].sum().clamp(min=1)
            ld = (bce(d_log, drop_t[sel]) * drop_mask[sel]).sum() / \
                 drop_mask[sel].sum().clamp(min=1)
            loss = la + ld
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)
        print(f"  epoch {ep+1}/{args.epochs}  loss {tot/n:.5f}")

    # ---- test ---------------------------------------------------------------
    drows, srows, prows = [], [], []
    model.eval()
    for start in sorted(test_sets):
        X, Y, R = test_sets[start]
        with torch.no_grad():
            a_log, d_log, sets = model(torch.from_numpy(X),
                                       torch.from_numpy(R), hard=True)
        a_p = torch.sigmoid(a_log).numpy()
        d_p = torch.sigmoid(d_log).numpy()
        pred_sets = sets.numpy() > 0.5
        last = X[:, -1, :] > 0.5
        rho_t = R[0]

        for b in range(args.n_out):
            m = start + args.n_in + b
            prev_s = last if b == 0 else (Y[:, b - 1, :] > 0.5)
            true = Y[:, b, :] > 0.5
            added = true & ~prev_s
            dropped = ~true & prev_s

            # ADD: candidates are labels absent from the current set
            ya, sa_model, sa_marg = [], [], []
            yd, sd_model = [], []
            for r in range(len(X)):
                ai = np.flatnonzero(~prev_s[r])
                ya.append(added[r][ai])
                sa_model.append(a_p[r, b][ai])
                sa_marg.append(rho_t[ai])
                di = np.flatnonzero(prev_s[r])
                yd.append(dropped[r][di])
                sd_model.append(d_p[r, b][di])
            ya = np.concatenate(ya)
            yd = np.concatenate(yd)
            drows.append({
                "start_month": names[start], "target_month": names[m],
                "h": b + 1,
                "add_base": float(ya.mean()),
                "add_ap_model": average_precision(ya, np.concatenate(sa_model)),
                "add_ap_marginal": average_precision(ya, np.concatenate(sa_marg)),
                "drop_base": float(yd.mean()),
                "drop_ap_model": average_precision(yd, np.concatenate(sd_model)),
                "mean_adds": float(added.sum(1).mean()),
                "mean_drops": float(dropped.sum(1).mean()),
            })

            tv = np.zeros(V, dtype=bool)
            for cs in msets[m]:
                for l in cs:
                    if l in lab_index:
                        tv[lab_index[l]] = True
            for nm, P in [("delta_set2set", pred_sets[:, b, :]),
                          ("persistence", last)]:
                mm = np.array([set_metrics(P[r], true[r]) for r in range(len(P))])
                srows.append({
                    "start_month": names[start], "target_month": names[m],
                    "h": b + 1, "model": nm,
                    "jaccard": float(mm[:, 0].mean()),
                    "precision": float(mm[:, 1].mean()),
                    "recall": float(mm[:, 2].mean()),
                })
                prows.append({
                    "start_month": names[start], "target_month": names[m],
                    "h": b + 1, "model": nm,
                    "vocab_jaccard": jaccard(P.any(axis=0), tv),
                    "pred_vocab": int(P.any(axis=0).sum()),
                    "true_vocab": int(tv.sum()),
                    "pred_occupancy": float(P.sum(1).mean()),
                    "true_occupancy": float(np.mean([len(cs) for cs in msets[m]])),
                })

    ddf = pd.DataFrame(drows)
    sdf = pd.DataFrame(srows)
    pdf = pd.DataFrame(prows)
    ddf.to_csv(f"{args.out_dir}/65_delta.csv", index=False)
    sdf.to_csv(f"{args.out_dir}/65_set_level.csv", index=False)
    pdf.to_csv(f"{args.out_dir}/65_population.csv", index=False)

    print("\n" + "=" * 74)
    print("DELTA  (the target: which labels are ADDED and which are DROPPED)")
    print("=" * 74)
    g = ddf.groupby("h")[["add_base", "add_ap_model", "add_ap_marginal",
                          "drop_base", "drop_ap_model",
                          "mean_adds", "mean_drops"]].mean()
    g["add_lift_model"] = g["add_ap_model"] / g["add_base"]
    g["add_lift_marginal"] = g["add_ap_marginal"] / g["add_base"]
    g["drop_lift_model"] = g["drop_ap_model"] / g["drop_base"]
    print(g.round(5).to_string())
    print("\n  persistence scores at the base rate on both by construction --")
    print("  it predicts no change. The model must beat 'marginal' on adds:")
    print("  that is population frequency alone, with no set context.")

    print("\n" + "=" * 74)
    print("SET LEVEL  (kept for comparison with 63 and 64; not the target)")
    print("=" * 74)
    print(sdf.groupby(["model", "h"])[
        ["jaccard", "precision", "recall"]].mean().round(4).to_string())
    print("\nPOPULATION LEVEL")
    print(pdf.groupby(["model", "h"])[
        ["vocab_jaccard", "pred_vocab", "true_vocab",
         "pred_occupancy", "true_occupancy"]].mean().round(4).to_string())

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
