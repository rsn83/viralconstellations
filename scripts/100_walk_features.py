#!/usr/bin/env python3
"""
100_walk_features.py

Structural features for a candidate (parent group j, mutation n): where n sits
in the temporal co-occurrence hypergraph RELATIVE TO j's fingerprint, rather
than n's marginal statistics.

WHY
  Every feature in script 98 is a count about n alone -- how frequent, how
  recently seen, how high it ever got, whether another group carries it. None
  asks whether n is CONNECTED to this particular parent through chains of
  observed co-occurrence. HIT (Liu, Ma & Li, WWW 2022) makes exactly that
  point: when predicting an interaction that has never happened there is no
  history to rely on, so the encoder must represent the candidate's position in
  the temporal structure instead.

HYPERGRAPH
  nodes      = mutations
  hyperedges = observed mutation sets, weighted by how many sequences carry them
  time       = month observed

  Co-occurrence in a backward window [t-L, t] is  C = X^T diag(w) X  with X the
  sparse set-by-mutation matrix. Row-normalising C gives a random-walk matrix P,
  so h-hop reachability from the fingerprint is exactly v @ P^h -- no sampling.
  Only months <= t enter, so nothing from the future is used.

FEATURES per (month, group, mutation)
  reach_1/2/3    reachable from the fingerprint in 1, <=2, <=3 hops
  min_hops       shortest hop distance, scaled
  walk_prob      total random-walk mass arriving within 3 hops
  walk_prob_1    mass arriving in exactly 1 hop
  co_parent      direct co-occurrence mass with the fingerprint, normalised
  recency        how recent the connecting co-occurrences are

Usage:
  python 100_walk_features.py \
      --data-dir data/processed/full_data_graphs_posres \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --npz      results/91_K24.npz --K 24 \
      --months   2020-06:2024-12 --lookback 6 --out results/100_walk.npz
"""
import argparse, pickle, csv, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

EPS = 1e-12
FEATS = ["reach_1", "reach_2", "reach_3", "min_hops",
         "walk_prob", "walk_prob_1", "co_parent", "recency"]


def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def months_in_range(spec):
    a, b = spec.split(":") if ":" in spec else (spec, spec)
    out = [a]
    while out[-1] != b:
        out.append(ym_add(out[-1], 1))
        if len(out) > 400: sys.exit("bad range")
    return out


def load_month(data_dir, ym):
    p = Path(data_dir) / f"{ym}_occupied.pkl"
    if not p.exists(): return None
    obj = pickle.load(open(p, "rb"))
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        return [(frozenset(v), 1) for v in vals]
    items = list(obj)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(frozenset(s), int(c)) for s, c in items]
    return [(frozenset(s), 1) for s in items]


def load_V(path):
    n = 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n = max(n, int(row["node_idx"]) + 1)
    return n


def month_cooc(records, V):
    """C_t = X^T diag(w) X  -- co-occurrence mass between mutations this month."""
    rows, cols, vals = [], [], []
    for i, (s, c) in enumerate(records):
        idx = [n for n in s if 0 <= n < V]
        if len(idx) < 2: continue
        rows += [i] * len(idx); cols += idx
        vals += [float(c)] * len(idx)
    if not rows: return np.zeros((V, V), dtype=np.float32)
    X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(max(rows) + 1, V))
    w = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)),
                      shape=X.shape)
    C = (w.T @ X).toarray().astype(np.float32)
    np.fill_diagonal(C, 0.0)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--months", required=True)
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--fp-thresh", type=float, default=.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    V = load_V(args.vocab)
    months = months_in_range(args.months)
    d = np.load(args.npz)
    if "theta" in d: theta = d["theta"]
    else:
        p = f"K{args.K}_"
        if p + "theta" not in d:
            avail = sorted({k.split('_')[0] for k in d.files if k.startswith('K')})
            sys.exit(f"pass --K; available: {avail}")
        theta = d[p + "theta"]
    K = theta.shape[0]
    print(f"V = {V:,}   K = {K}   months {months[0]}..{months[-1]} "
          f"({len(months)})   lookback = {args.lookback}")

    print("\nper-month co-occurrence ...", flush=True)
    Cs = []
    for ym in months:
        r = load_month(args.data_dir, ym)
        Cs.append(month_cooc(r, V) if r else np.zeros((V, V), np.float32))
        print(f"  {ym}", end="\r", flush=True)
    print(" " * 30, end="\r")

    fps = [np.flatnonzero(theta[k] > args.fp_thresh) for k in range(K)]
    nz = [len(f) for f in fps]
    print(f"fingerprints: min {min(nz)} max {max(nz)}, "
          f"{sum(1 for f in fps if len(f)==0)} empty groups")

    T = len(months) - 1
    out = np.zeros((T, K, V, len(FEATS)), dtype=np.float32)

    for t in range(T):
        lo = max(0, t - args.lookback + 1)
        C = np.zeros((V, V), np.float32)
        # recency: most recent month each pair co-occurred, as a decay weight
        R = np.zeros((V, V), np.float32)
        for s in range(lo, t + 1):
            C += Cs[s]
            R = np.where(Cs[s] > 0, float(s), R)
        deg = C.sum(1, keepdims=True)
        P = C / (deg + EPS)                       # random-walk matrix
        age = np.where(R > 0, (t - R) / max(args.lookback, 1), 1.0)

        for k in range(K):
            if len(fps[k]) == 0: continue
            v0 = np.zeros(V, np.float32); v0[fps[k]] = 1.0 / len(fps[k])
            v1 = v0 @ P
            v2 = v1 @ P
            v3 = v2 @ P
            r1 = (v1 > 0).astype(np.float32)
            r2 = ((v1 + v2) > 0).astype(np.float32)
            r3 = ((v1 + v2 + v3) > 0).astype(np.float32)
            hops = np.where(r1 > 0, 1, np.where(r2 > 0, 2,
                            np.where(r3 > 0, 3, 9))).astype(np.float32)
            co = C[fps[k]].sum(0); co = co / (co.max() + EPS)
            rec = 1.0 - age[fps[k]].min(0)        # 1 = co-occurred this month
            f = out[t, k]
            f[:, 0] = r1; f[:, 1] = r2; f[:, 2] = r3
            f[:, 3] = hops / 9.0
            f[:, 4] = v1 + v2 + v3
            f[:, 5] = v1
            f[:, 6] = co
            f[:, 7] = rec
            # a mutation the group already has is not a candidate
            f[theta[k] > args.fp_thresh, :] = 0.0
        if (t + 1) % 6 == 0 or t == T - 1:
            print(f"  {months[t]} ({t+1}/{T})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, W=out, feats=np.array(FEATS),
                        months=np.array(months[:T]))
    print(f"\nsaved -> {args.out}   shape {out.shape}")

    nzf = (out.reshape(-1, len(FEATS)) != 0).mean(0)
    print(f"\n{'feature':<14}{'nonzero share':>15}{'mean(nonzero)':>15}")
    for i, nm in enumerate(FEATS):
        col = out[..., i].ravel(); m = col[col != 0]
        print(f"{nm:<14}{nzf[i]:>15.1%}{(m.mean() if m.size else 0):>15.4f}")
    print("""
WHAT THESE ADD
  Existing features answer 'is this mutation common / recent / present
  elsewhere' -- marginal facts about the mutation alone. These answer 'is this
  mutation CONNECTED to this particular parent through chains of observed
  co-occurrence, and how recently were those chains formed'.

  For BA.5 the question is whether 452R, which sits in Delta's sets, reaches
  BA.2's fingerprint in two or three hops through sets sharing mutations with
  both. No count-based feature can see that.

  Reachability is computed exactly by propagating the random-walk matrix, not
  by sampling. Only months <= t enter the window.
""")


if __name__ == "__main__":
    main()
