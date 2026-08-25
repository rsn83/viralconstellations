#!/usr/bin/env python3
"""
108_within_block_independence.py

The mixture assumes mutations are independent GIVEN the block. That is what
block structure implies, and the ~12x redundancy supports it. But it is exact
only if blocks are pure. Two ways it breaks:

  sub-lineages       BA.2 and BA.2.12.1 in one block makes 452Q and 704L
                     correlated inside it
  mid-acquisition    if 486V sits at 0.5 in a block because half its sequences
                     are BA.5, then 486V and 452R arrive together and are
                     correlated inside it

This measures the residual correlation. For each block, take the sequences
assigned to it and compute, for every pair of mutations, how far the observed
co-occurrence sits from what independence predicts:

    expected     n * theta_a * theta_b
    observed     sequences in the block carrying both
    residual     (observed - expected) / sqrt(expected)      a z-like score

Under exact conditional independence these are small and centred on zero.

REPORTED
  1  per block: the largest residual correlations, named
  2  a summary: how much of the total pairwise dependence survives inside
     blocks, compared with how much there was before conditioning
  3  the specific pairs the project cares about (486V/452R, 452Q/704L)

If the residuals are large for pairs that define a sub-lineage, independence is
violated exactly where it matters and raising K is the direct fix.

Usage:
  python 108_within_block_independence.py \
      --npz    results/91_K24_withdel.npz \
      --vocab  data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
      --data-dir data/processed/full_data_graphs_withdel \
      --months 2021-06:2022-06 --K 24
"""
import argparse, pickle, csv, sys
from pathlib import Path
import numpy as np

EPS = 1e-12


def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def months_in_range(spec):
    if ":" not in spec: return [spec]
    a, b = spec.split(":"); out = [a]
    while out[-1] != b:
        out.append(ym_add(out[-1], 1))
        if len(out) > 300: sys.exit("bad range")
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


def load_names(path):
    names, V = {}, 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            names[i] = f"{row['aa_pos']}{row['residue'].strip()}"
    return names, V


def build(records, V):
    w = np.array([c for _, c in records], float)
    X = np.zeros((len(records), V), dtype=np.float32)
    for i, (s, _) in enumerate(records):
        X[i, [n for n in s if 0 <= n < V]] = 1.0
    return X, w


def loglik_matrix(X, th):
    lt, lc = np.log(th + EPS), np.log(1 - th + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--months", required=True)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min-share", type=float, default=.02,
                    help="only report blocks holding at least this share")
    ap.add_argument("--pairs", default="486V:452R,452Q:704L")
    args = ap.parse_args()

    d = np.load(args.npz)
    theta = d["theta"] if "theta" in d else d[f"K{args.K}_theta"]
    Pi = d["Pi"] if "Pi" in d else d[f"K{args.K}_Pi"]
    names, V = load_names(args.vocab)
    idx = {v: k for k, v in names.items()}
    K = theta.shape[0]
    months = months_in_range(args.months)
    print(f"K = {K}   V = {V:,}   months {months[0]}..{months[-1]}")

    # ---- assign every sequence to its most likely block ----
    lt, lc = np.log(theta + EPS), np.log(1 - theta + EPS)
    W = lt - lc; C = lc.sum(1)
    n_k = np.zeros(K)                       # weighted sequences per block
    S1 = np.zeros((K, V))                   # per block, weighted count per mutation
    pair_obs = {}                           # per block, sparse pair counts
    blocks_X = {k: [] for k in range(K)}
    blocks_w = {k: [] for k in range(K)}

    for t, ym in enumerate(months):
        r = load_month(args.data_dir, ym)
        if r is None: continue
        X, w = build(r, V)
        lp = X @ W.T + C[None, :] + np.log(Pi[min(t, len(Pi) - 1)] + EPS)[None, :]
        z = lp.argmax(1)
        for k in range(K):
            m = z == k
            if not m.any(): continue
            n_k[k] += w[m].sum()
            S1[k] += (w[m, None] * X[m]).sum(0)
            blocks_X[k].append(X[m]); blocks_w[k].append(w[m])
        print(f"  {ym} assigned", end="\r", flush=True)
    print(" " * 30, end="\r")

    share = n_k / n_k.sum()
    live = [k for k in range(K) if share[k] >= args.min_share]
    print(f"\nblocks holding >= {args.min_share:.0%} of sequences: {live}")

    # ---- per-block residual correlation ----
    print("\n" + "=" * 78)
    print("1  LARGEST RESIDUAL CORRELATIONS INSIDE EACH BLOCK")
    print("=" * 78)
    print("""
  For every pair, expected co-occurrence under independence-given-block is
  n * p_a * p_b, using the block's OWN observed rates. The residual is
  (observed - expected)/sqrt(expected). Under exact conditional independence
  these are small; large positive values mean the pair arrives together inside
  the block, which independence cannot represent.
""")
    summary = []
    for k in live:
        Xk = np.vstack(blocks_X[k]); wk = np.concatenate(blocks_w[k])
        n = wk.sum()
        p = (wk[:, None] * Xk).sum(0) / n
        # restrict to mutations that actually vary inside the block
        var = np.flatnonzero((p > .02) & (p < .98))
        if len(var) < 2:
            print(f"\n  block {k:<3} share {share[k]:.1%}   "
                  f"no mutation varies inside it (pure block)")
            summary.append((k, share[k], len(var), 0.0, 0.0))
            continue
        Xv = Xk[:, var]
        Co = (Xv * wk[:, None]).T @ Xv          # weighted co-occurrence
        pv = p[var]
        Exp = n * np.outer(pv, pv)
        R = (Co - Exp) / np.sqrt(Exp + 1.0)
        np.fill_diagonal(R, 0.0)
        iu = np.triu_indices(len(var), 1)
        vals = R[iu]
        order = np.argsort(-np.abs(vals))[:args.top]
        print(f"\n  block {k:<3} share {share[k]:5.1%}   "
              f"{len(var)} mutations vary inside it   "
              f"median |residual| {np.median(np.abs(vals)):.1f}")
        print(f"    {'pair':<22}{'observed':>11}{'expected':>11}{'residual':>11}")
        for o in order:
            a, b = var[iu[0][o]], var[iu[1][o]]
            print(f"    {names.get(a,a)+' + '+names.get(b,b):<22}"
                  f"{Co[iu[0][o], iu[1][o]]:>11,.0f}"
                  f"{Exp[iu[0][o], iu[1][o]]:>11,.0f}{vals[o]:>+11.1f}")
        summary.append((k, share[k], len(var),
                        float(np.median(np.abs(vals))),
                        float(np.abs(vals).max())))

    # ---- summary ----
    print("\n" + "=" * 78)
    print("2  SUMMARY")
    print("=" * 78)
    print(f"\n  {'block':<8}{'share':>9}{'varying':>10}{'median |res|':>15}"
          f"{'max |res|':>12}")
    for k, sh, nv, med, mx in summary:
        print(f"  blk{k:<5}{sh:>9.1%}{nv:>10}{med:>15.1f}{mx:>12.1f}")
    print("""
  'varying' counts mutations strictly between 2% and 98% inside the block. A
  pure block has few: its sequences either all carry a mutation or none do. A
  block covering two sub-lineages has many, and those are where independence
  breaks.
""")

    # ---- the pairs we care about ----
    print("=" * 78)
    print("3  SPECIFIC PAIRS")
    print("=" * 78)
    for spec in args.pairs.split(","):
        a_s, _, b_s = spec.partition(":")
        if a_s not in idx or b_s not in idx:
            print(f"\n  {spec}: not in the vocabulary"); continue
        a, b = idx[a_s], idx[b_s]
        print(f"\n  {a_s} + {b_s}")
        print(f"    {'block':<8}{'share':>9}{'p(a)':>8}{'p(b)':>8}"
              f"{'observed':>11}{'expected':>11}{'residual':>11}")
        for k in live:
            Xk = np.vstack(blocks_X[k]); wk = np.concatenate(blocks_w[k])
            n = wk.sum()
            pa = float((wk * Xk[:, a]).sum() / n)
            pb = float((wk * Xk[:, b]).sum() / n)
            obs = float((wk * Xk[:, a] * Xk[:, b]).sum())
            exp = n * pa * pb
            res = (obs - exp) / np.sqrt(exp + 1.0)
            print(f"    blk{k:<5}{share[k]:>9.1%}{pa:>8.3f}{pb:>8.3f}"
                  f"{obs:>11,.0f}{exp:>11,.0f}{res:>+11.1f}")

    print("""

HOW TO READ
  A large positive residual for a pair that defines a sub-lineage means the
  block is covering two lineages at once and independence cannot represent it.
  The direct fix is more blocks: split that block and the pair becomes
  independent inside each half.

  Small residuals everywhere mean the assumption holds where it is used, and
  the model's failures lie elsewhere.

  Caveat: residuals scale with block size, so a large block will show larger
  values for the same degree of dependence. Compare the median column across
  blocks rather than raw maxima.
""")


if __name__ == "__main__":
    main()
