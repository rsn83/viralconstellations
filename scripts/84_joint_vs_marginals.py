#!/usr/bin/env python
"""
84_joint_vs_marginals.py

One number per month: how much structure exists in the joint distribution over
mutation sets, beyond what the individual mutation frequencies already say.

The quantity
------------
For a month, let f_m be the frequency of each mutation and let the observed
distribution over sets be p(c). Compare p against the distribution you would get
if every mutation were present independently at its own frequency:

    q(c) = prod over m in c of f_m, times prod over m not in c of (1 - f_m)

The gap is the Kullback-Leibler divergence D(p || q), measured in bits per
sequence. Equivalently, and this is how it is computed here:

    D = sum over mutations of H(f_m)   minus   H(p)

the sum of the individual mutation entropies minus the entropy of the joint.
That is the multi-information, or total correlation: everything the mutations
tell you about each other.

Why it matters
--------------
Every model tested so far asked whether a PARTICULAR kind of structure helps
prediction -- pairwise co-occurrence, inheritance trees, protein language model
scores. This asks how much structure is present at all. It is an upper bound on
what any model exploiting co-occurrence could ever recover, and it needs no
model and no fitting.

Two estimation problems, both handled
-------------------------------------
1. H(p) is estimated from 5,000 sequences over a space of 2^V sets. The plug-in
   estimator is biased downward, which inflates D. The Miller-Madow correction
   is applied and the raw value reported alongside.
2. Even with independent mutations, a finite sample produces a positive D.
   So each month is also run on SHUFFLED data: each mutation's column is
   permuted independently across sequences, which destroys every association
   while keeping the marginals exactly. The shuffled value is the floor. What
   matters is the observed value minus that floor.

Also reported: the sum over all pairs of their mutual information.

This is NOT a component of the total -- it can and usually does exceed it,
because when many mutations share the same underlying association, every pair
counts that same information again. The ratio is therefore a REDUNDANCY measure,
not a share:

  ratio near 1   the associations are essentially pairwise and mostly distinct
  ratio above 1  the same association is visible in many pairs at once, which is
                 what a group of mutations travelling together as a lineage
                 looks like. A ratio of 10 means a typical pairwise association
                 is being counted about ten times over.

Usage
-----
python scripts/84_joint_vs_marginals.py --min_count 3 --end_month 2024-12
python scripts/84_joint_vs_marginals.py --self_test
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


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


def sample_matrix(occ, n_target, rng):
    """Draw n_target sequences and return a binary matrix, sequences x mutations."""
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if counts.sum() < n_target:
        return None, None
    draws = rng.multinomial(n_target, counts / counts.sum())
    labs = sorted({l for k, d in zip(keys, draws) if d > 0 for l in k}, key=str)
    idx = {l: i for i, l in enumerate(labs)}
    X = np.zeros((n_target, len(labs)), dtype=np.int8)
    r = 0
    for k, d in zip(keys, draws):
        if d == 0:
            continue
        cols = [idx[l] for l in k]
        X[r:r + d, cols] = 1
        r += d
    return X, labs


def h_binary(p):
    """Entropy of a Bernoulli, in bits. Vectorised, safe at 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -(p[m] * np.log2(p[m]) + (1 - p[m]) * np.log2(1 - p[m]))
    return out


def joint_entropy(X):
    """
    Plug-in entropy of the distribution over rows, in bits, plus the
    Miller-Madow corrected value. Rows are hashed by their byte representation,
    so identical sets collapse to one outcome.
    """
    n = X.shape[0]
    _, counts = np.unique(np.ascontiguousarray(X).view(
        np.dtype((np.void, X.dtype.itemsize * X.shape[1]))), return_counts=True)
    p = counts / n
    h = float(-(p * np.log2(p)).sum())
    # Miller-Madow: add (number of observed outcomes - 1) / (2 n ln 2)
    h_mm = h + (len(counts) - 1) / (2 * n * np.log(2))
    return h, h_mm, len(counts)


def pairwise_mi(X, max_cols=200):
    """
    Sum of pairwise mutual information, in bits. Restricted to the most variable
    columns when there are many, since the sum grows quadratically and the tail
    columns are nearly constant.
    """
    n, V = X.shape
    p = X.mean(axis=0)
    var = p * (1 - p)
    keep = np.argsort(-var)[:min(max_cols, V)]
    Y = X[:, keep].astype(np.float64)
    py = Y.mean(axis=0)
    # joint counts for each pair
    n11 = (Y.T @ Y) / n
    n10 = py[:, None] - n11
    n01 = py[None, :] - n11
    n00 = 1.0 - n11 - n10 - n01
    tot = 0.0
    for a, pa, pb in ((n11, py[:, None], py[None, :]),
                      (n10, py[:, None], 1 - py[None, :]),
                      (n01, 1 - py[:, None], py[None, :]),
                      (n00, 1 - py[:, None], 1 - py[None, :])):
        q = pa * pb
        m = (a > 0) & (q > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(m, a * np.log2(np.where(m, a / q, 1.0)), 0.0)
        tot += term
    iu = np.triu_indices(len(keep), 1)
    return float(tot[iu].sum()), len(keep)


def shuffle_columns(X, rng):
    """Permute each column independently: marginals kept, associations destroyed."""
    Y = X.copy()
    for j in range(Y.shape[1]):
        rng.shuffle(Y[:, j])
    return Y


def self_test():
    print("checking the information measures")
    rng = np.random.default_rng(0)

    # independent columns: the gap should be near zero once the finite-sample
    # floor is subtracted
    n, V = 4000, 12
    p = rng.uniform(0.2, 0.8, V)
    X = (rng.random((n, V)) < p).astype(np.int8)
    hm = h_binary(X.mean(axis=0)).sum()
    h, h_mm, k = joint_entropy(X)
    d_obs = hm - h_mm
    Xs = shuffle_columns(X, rng)
    hs, hs_mm, _ = joint_entropy(Xs)
    d_null = h_binary(Xs.mean(axis=0)).sum() - hs_mm
    print(f"  independent columns: D {d_obs:.3f}, shuffled floor {d_null:.3f}, "
          f"excess {d_obs - d_null:+.3f}")
    assert abs(d_obs - d_null) < 0.15, (d_obs, d_null)
    print("     -> excess near zero                            ok")

    # perfectly linked columns: the gap must be large
    base = (rng.random(n) < 0.5).astype(np.int8)
    Y = np.tile(base[:, None], (1, V))
    hm2 = h_binary(Y.mean(axis=0)).sum()
    h2, h2_mm, _ = joint_entropy(Y)
    d2 = hm2 - h2_mm
    assert d2 > 10, d2
    print(f"  12 identical columns: D {d2:.2f} bits "
          f"(V-1 = 11 expected)             ok")

    # pairwise mutual information: zero for independent, high for linked
    mi_ind, _ = pairwise_mi(X)
    mi_link, _ = pairwise_mi(Y)
    assert mi_ind < 1.0 and mi_link > 50, (mi_ind, mi_link)
    print(f"  pairwise MI: independent {mi_ind:.2f}, linked "
          f"{mi_link:.1f}          ok")
    # 12 identical columns: total correlation is 11 bits, but the pairwise sum
    # is 66 -- one bit counted once per pair. This is why the ratio is a
    # redundancy measure and not a share.
    assert mi_link / d2 > 5, (mi_link, d2)
    print(f"     ratio {mi_link/d2:.1f} for 12 identical columns, because the")
    print("     same 1 bit appears in all 66 pairs                ok")

    # shuffling must preserve the marginals exactly
    assert np.allclose(X.mean(axis=0), Xs.mean(axis=0))
    print("  shuffling preserves every marginal exactly       ok")

    # entropy of a fair coin is 1 bit
    assert abs(h_binary([0.5])[0] - 1.0) < 1e-12
    assert abs(h_binary([0.0])[0]) < 1e-12
    print("  binary entropy: 0.5 -> 1 bit, 0 -> 0 bits        ok")
    print("all checks passed\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--n_per_month", type=int, default=5000)
    ap.add_argument("--n_shuffle", type=int, default=3)
    ap.add_argument("--max_pair_cols", type=int, default=200)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    rows = []
    for m, occ in months:
        X, labs = sample_matrix(occ, args.n_per_month, rng)
        if X is None:
            continue
        f = X.mean(axis=0)
        sum_marginal_h = float(h_binary(f).sum())
        h, h_mm, n_distinct = joint_entropy(X)
        d_obs = sum_marginal_h - h_mm

        nulls = []
        for _ in range(args.n_shuffle):
            Xs = shuffle_columns(X, rng)
            hs, hs_mm, _ = joint_entropy(Xs)
            nulls.append(float(h_binary(Xs.mean(axis=0)).sum()) - hs_mm)
        d_null = float(np.mean(nulls))

        mi, n_cols = pairwise_mi(X, args.max_pair_cols)
        mi_nulls = []
        for _ in range(args.n_shuffle):
            mi_s, _ = pairwise_mi(shuffle_columns(X, rng), args.max_pair_cols)
            mi_nulls.append(mi_s)
        mi_null = float(np.mean(mi_nulls))

        rows.append({
            "month": m,
            "mutations": len(labs),
            "distinct_sets": n_distinct,
            "sum_of_mutation_entropies": sum_marginal_h,
            "joint_entropy": h_mm,
            "structure_raw": d_obs,
            "structure_shuffled_floor": d_null,
            "structure": d_obs - d_null,
            "share_of_marginal_entropy": ((d_obs - d_null) / sum_marginal_h
                                          if sum_marginal_h > 0 else np.nan),
            "sum_pairwise_mi": mi - mi_null,
            "redundancy_ratio": ((mi - mi_null) / (d_obs - d_null)
                                 if (d_obs - d_null) > 0 else np.nan),
            "pair_columns_used": n_cols,
        })
        print(f"  {m}: {len(labs)} mutations, {n_distinct} distinct sets, "
              f"structure {d_obs - d_null:.2f} bits")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/84_joint_vs_marginals.csv", index=False)

    print("\n" + "=" * 100)
    print("HOW MUCH THE MUTATIONS TELL YOU ABOUT EACH OTHER")
    print("=" * 100)
    print("all values in bits per sequence\n")
    show = df[["month", "mutations", "distinct_sets",
               "sum_of_mutation_entropies", "joint_entropy", "structure_raw",
               "structure_shuffled_floor", "structure",
               "share_of_marginal_entropy", "sum_pairwise_mi",
               "redundancy_ratio"]].copy()
    show.columns = ["month", "mutations", "distinct sets",
                    "sum of mutation entropies", "joint entropy",
                    "structure (raw)", "shuffled floor", "structure",
                    "share of marginal entropy", "sum of pairwise MI",
                    "redundancy ratio"]
    print(show.round(3).to_string(index=False))

    print("\naverages:")
    print(f"  sum of mutation entropies : "
          f"{df['sum_of_mutation_entropies'].mean():.2f} bits")
    print(f"  joint entropy             : {df['joint_entropy'].mean():.2f} bits")
    print(f"  structure                 : {df['structure'].mean():.2f} bits")
    print(f"  as a share of the marginal entropy: "
          f"{df['share_of_marginal_entropy'].mean():.3f}")
    print(f"  sum of pairwise MI        : "
          f"{df['sum_pairwise_mi'].mean():.2f} bits")
    print(f"  redundancy ratio          : "
          f"{df['redundancy_ratio'].mean():.2f}")

    print("\nreading:")
    print("  'sum of mutation entropies' is how many bits you would need per")
    print("  sequence if every mutation were independent.")
    print("  'joint entropy' is how many bits the real sequences actually need.")
    print("  'structure' is the difference, after subtracting the floor that a")
    print("  finite sample produces even with no association at all.")
    print("  A large share means the mutations are highly redundant with each")
    print("  other -- knowing some tells you most of the rest.")
    print("  'sum of pairwise MI' adds up every pair's mutual information.")
    print("  It is NOT a share of the total -- it exceeds it whenever many")
    print("  pairs reflect the same underlying association. The ratio measures")
    print("  that duplication: a ratio of 10 means a typical association is")
    print("  being counted about ten times across different pairs, which is")
    print("  what a block of mutations travelling together looks like.")

    print(f"\nwrote outputs/84_joint_vs_marginals.csv")


if __name__ == "__main__":
    main()
