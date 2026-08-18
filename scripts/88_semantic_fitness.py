#!/usr/bin/env python
"""
88_semantic_fitness.py

The question
------------
Script 87 showed a two-parameter model -- copy-forward plus a size-dependent
fitness term -- beats copy-forward out-of-sample 62% of months. The fitness
term was s(c) = beta * |c|: larger sets gain mass.

This script adds two more terms:

  grammar:   mean llr_ref over mutations in c, from the ESM cache.
             llr_ref(m) = log p(mutated residue | context) / p(reference residue | context)
             High grammar means the set's mutations are tolerable in the
             protein context. Static -- does not change month to month.

  semantic:  cosine distance between c's embedding and the dominant set's
             embedding at month t. The dominant set is the highest-frequency
             set this month. Computed from the dir_pc features in the ESM cache,
             summed over mutations in c and normalised.
             HIGH semantic distance means c looks antigenically different from
             what is currently dominant.
             Dynamic -- changes every month as the dominant changes.

The full fitness model:
  s(c, t) = beta * |c| + gamma * gram(c) + delta * sem_dist(c, d_t)

where d_t is the dominant set at month t.

This is the CSCS hypothesis (Hie et al. Science 2021) implemented dynamically
in a population-level branching process rather than as a static scoring function.
CSCS says high grammar + high semantic change predicts escape. Here: high grammar
+ high semantic distance from the current dominant predicts mass gain.

Two components tested:
  A. Mass redistribution: does the fitness term better predict which existing
     sets gain mass, versus copy-forward and the size-only model?
  B. Composition generation: does the fitness term better rank one-step
     candidate new sets, versus frequency-only ranking?

Both are evaluated out-of-sample: parameters fitted on months 1..train_months,
fixed for months train_months..T.

Usage
-----
python scripts/88_semantic_fitness.py --esm outputs/esm_node_features_ref.pkl
python scripts/88_semantic_fitness.py --self_test
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


# ----------------------------------------------------------------------------
# ESM features
# ----------------------------------------------------------------------------

def load_esm(path):
    """
    Load the ESM cache. Returns:
      llr:   {(pos, res): float}      grammaticality
      pc:    {(pos, res): np.array}   dir_pc embedding vector
      names: list of feature names
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    F = np.asarray(obj["features"])
    names = [str(x) for x in obj["names"]]
    print(f"ESM cache: {F.shape[0]} cells, features: {names}")

    # load the vocabulary to map row index to (pos, res)
    return F, names


def load_vocab(data_dir):
    """
    Load posres_vocab.tsv. Returns {node_idx: (pos, res)} and
    {(pos, res): node_idx}.
    """
    path = os.path.join(data_dir, "posres_vocab.tsv")
    idx2pr, pr2idx = {}, {}
    with open(path) as f:
        header = f.readline().strip().split("\t")
        cols = {c.lower(): i for i, c in enumerate(header)}
        id_col = next(cols[c] for c in ("node_idx", "id", "node")
                      if c in cols)
        pos_col = next(cols[c] for c in ("aa_pos", "pos", "position")
                       if c in cols)
        res_col = next(cols[c] for c in ("residue", "res", "aa")
                       if c in cols)
        for line in f:
            parts = line.strip().split("\t")
            nid = int(parts[id_col])
            pos = int(parts[pos_col])
            res = str(parts[res_col])
            idx2pr[nid] = (pos, res)
            pr2idx[(pos, res)] = nid
    return idx2pr, pr2idx


# ----------------------------------------------------------------------------
# data
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
    w = np.array([v for _, v in items], dtype=float)
    return [c for c, _ in items], w / w.sum()


# ----------------------------------------------------------------------------
# feature computation
# ----------------------------------------------------------------------------

def set_embedding(s, F, pr2idx, pc_cols):
    """
    Embedding of a set as the mean of its mutations' dir_pc vectors.
    Returns a unit-norm vector, or None if no mutations are in the cache.
    """
    vecs = []
    for label in s:
        idx = pr2idx.get(label)
        if idx is not None and idx < F.shape[0]:
            vecs.append(F[idx, pc_cols])
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def set_grammar(s, F, pr2idx, llr_col):
    """Mean llr_ref over mutations in the set."""
    vals = []
    for label in s:
        idx = pr2idx.get(label)
        if idx is not None and idx < F.shape[0]:
            vals.append(float(F[idx, llr_col]))
    return float(np.mean(vals)) if vals else 0.0


def cosine_dist(a, b):
    if a is None or b is None:
        return 0.0
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def model_q(p, sets, features, params):
    """
    p: current distribution (array, sums to 1)
    sets: list of frozensets
    features: array (n_sets, 3): [size, grammar, sem_dist]
    params: [beta, gamma, delta]
    """
    beta, gamma, delta = params
    logits = (beta * features[:, 0] +
              gamma * features[:, 1] +
              delta * features[:, 2])
    q = p * np.exp(logits)
    s = q.sum()
    return q / s if s > 0 else p.copy()


def neg_ll(params, p, q_true, features):
    q = model_q(p, None, features, params)
    return -float(np.sum(q_true * np.log(np.clip(q, 1e-300, None))))


def fit_params(p, q_true, features, n_starts=6):
    best_val, best_p = np.inf, [0.0, 0.0, 0.0]
    for b0, g0, d0 in [(0.1, 0.0, 0.0), (0.2, 0.1, 0.1),
                       (0.0, 0.1, 0.1), (0.1, -0.1, 0.1),
                       (0.0, 0.0, 0.0), (0.3, 0.0, 0.1)]:
        try:
            r = minimize(neg_ll, [b0, g0, d0], args=(p, q_true, features),
                         method="Nelder-Mead",
                         options={"xatol": 1e-9, "fatol": 1e-9,
                                  "maxiter": 10000})
            if r.fun < best_val:
                best_val = r.fun
                best_p = list(r.x)
        except Exception:
            pass
    return best_p


def kl(p, q):
    q = np.clip(q, 1e-300, None)
    ok = p > 0
    return float(np.sum(p[ok] * np.log(p[ok] / q[ok])))


def average_precision(y, s):
    y = np.asarray(y, int)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-np.asarray(s, float))
    yy = y[order]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, y.size + 1)) * yy).sum() / y.sum())


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")
    rng = np.random.default_rng(0)

    # model at all-zero params must return copy-forward
    p = np.array([0.5, 0.3, 0.2])
    F = rng.normal(size=(3, 3))
    q = model_q(p, None, F, [0.0, 0.0, 0.0])
    assert np.allclose(q, p, atol=1e-9), q
    print("  zero params -> copy-forward                      ok")

    # positive beta shifts mass toward the largest-feature set
    F2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    q2 = model_q(p, None, F2, [1.0, 0.0, 0.0])
    assert q2[2] > q2[1] > q2[0], q2
    print("  positive beta shifts mass to larger features     ok")

    # cosine_dist: identical vectors -> 0, orthogonal -> 1
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_dist(a, a)) < 1e-9
    assert abs(cosine_dist(a, b) - 1.0) < 1e-9
    print("  cosine distance: identical 0, orthogonal 1       ok")

    # AP sanity
    y = np.array([1, 0, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
    ap = average_precision(y, s)
    assert ap > 0.8, ap
    print(f"  AP on informative score {ap:.3f}                  ok")

    # KL: zero for identical
    p2 = np.array([0.6, 0.3, 0.1])
    assert abs(kl(p2, p2)) < 1e-12
    print("  KL zero for identical distributions              ok")
    print("all checks passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="data/processed/full_data_graphs_posres")
    ap.add_argument("--esm", default="outputs/esm_node_features_ref.pkl")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_sets", type=int, default=200)
    ap.add_argument("--train_months", type=int, default=30)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # load ESM and vocabulary
    F, feat_names = load_esm(args.esm)
    idx2pr, pr2idx = load_vocab(args.data_dir)

    # identify feature columns
    llr_col = feat_names.index("llr_ref") if "llr_ref" in feat_names else None
    pc_cols = [i for i, n in enumerate(feat_names) if n.startswith("dir_pc")]
    is_sc_col = (feat_names.index("is_scorable")
                 if "is_scorable" in feat_names else None)
    print(f"llr_ref column: {llr_col}")
    print(f"dir_pc columns: {len(pc_cols)} ({feat_names[pc_cols[0]]} .. "
          f"{feat_names[pc_cols[-1]]})")

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")
    print(f"train: months 0..{args.train_months-1}, "
          f"test: months {args.train_months}..{T-2}\n")

    # ---- IN-SAMPLE: fit per transition, collect beta/gamma/delta ----------
    print("=" * 72)
    print("IN-SAMPLE (fits parameters to each transition -- NOT the real test)")
    print("=" * 72)
    in_rows = []
    for i in range(T - 1):
        sets_t, p_t = top_sets(occ_by[names[i]], args.max_sets)
        sets_n, p_n = top_sets(occ_by[names[i + 1]], args.max_sets)
        common = [c for c in sets_t if c in set(sets_n)]
        if len(common) < 10:
            continue
        idx_t = {c: j for j, c in enumerate(sets_t)}
        idx_n = {c: j for j, c in enumerate(sets_n)}
        p_t_r = np.array([p_t[idx_t[c]] for c in common])
        p_n_r = np.array([p_n[idx_n[c]] for c in common])
        p_t_r /= p_t_r.sum(); p_n_r /= p_n_r.sum()

        # dominant set embedding
        dom = sets_t[0]
        dom_emb = set_embedding(dom, F, pr2idx, pc_cols)

        feats = np.zeros((len(common), 3))
        for j, c in enumerate(common):
            feats[j, 0] = len(c)
            feats[j, 1] = (set_grammar(c, F, pr2idx, llr_col)
                           if llr_col is not None else 0.0)
            feats[j, 2] = cosine_dist(
                set_embedding(c, F, pr2idx, pc_cols), dom_emb)

        params = fit_params(p_t_r, p_n_r, feats)
        beta, gamma, delta = params
        q = model_q(p_t_r, common, feats, params)
        in_rows.append({"month": names[i], "beta": beta,
                        "gamma": gamma, "delta": delta,
                        "kl_model": kl(p_n_r, q),
                        "kl_cf": kl(p_n_r, p_t_r)})
        print(f"  {names[i]}: beta={beta:+.3f} gamma={gamma:+.3f} "
              f"delta={delta:+.3f}  "
              f"kl_model={kl(p_n_r,q):.4f} kl_cf={kl(p_n_r,p_t_r):.4f}")

    df_in = pd.DataFrame(in_rows)
    train = df_in.iloc[:args.train_months]
    mean_beta = float(train["beta"].mean())
    mean_gamma = float(train["gamma"].mean())
    mean_delta = float(train["delta"].mean())
    print(f"\nmean parameters from first {args.train_months} months:")
    print(f"  beta  = {mean_beta:+.4f}  (size drift)")
    print(f"  gamma = {mean_gamma:+.4f}  (grammar)")
    print(f"  delta = {mean_delta:+.4f}  (semantic distance from dominant)")

    # ---- OUT-OF-SAMPLE: fixed parameters ----------------------------------
    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE  (fixed parameters from training months)")
    print("=" * 72)
    print(f"  {'month':>8}  {'KL_full':>10}  {'KL_size':>10}  "
          f"{'KL_cf':>10}  {'winner':>12}")

    oos_rows = []
    for i in range(args.train_months, T - 1):
        sets_t, p_t = top_sets(occ_by[names[i]], args.max_sets)
        sets_n, p_n = top_sets(occ_by[names[i + 1]], args.max_sets)
        common = [c for c in sets_t if c in set(sets_n)]
        if len(common) < 10:
            continue
        idx_t = {c: j for j, c in enumerate(sets_t)}
        idx_n = {c: j for j, c in enumerate(sets_n)}
        p_t_r = np.array([p_t[idx_t[c]] for c in common])
        p_n_r = np.array([p_n[idx_n[c]] for c in common])
        p_t_r /= p_t_r.sum(); p_n_r /= p_n_r.sum()

        dom = sets_t[0]
        dom_emb = set_embedding(dom, F, pr2idx, pc_cols)

        feats = np.zeros((len(common), 3))
        for j, c in enumerate(common):
            feats[j, 0] = len(c)
            feats[j, 1] = (set_grammar(c, F, pr2idx, llr_col)
                           if llr_col is not None else 0.0)
            feats[j, 2] = cosine_dist(
                set_embedding(c, F, pr2idx, pc_cols), dom_emb)

        # three models
        q_full = model_q(p_t_r, common, feats,
                         [mean_beta, mean_gamma, mean_delta])
        q_size = model_q(p_t_r, common, feats, [mean_beta, 0.0, 0.0])
        kl_full = kl(p_n_r, q_full)
        kl_size = kl(p_n_r, q_size)
        kl_cf = kl(p_n_r, p_t_r)

        best = min(kl_full, kl_size, kl_cf)
        winner = ("full" if best == kl_full else
                  "size" if best == kl_size else "copy-forward")
        oos_rows.append({"month": names[i], "kl_full": kl_full,
                         "kl_size": kl_size, "kl_cf": kl_cf,
                         "winner": winner})
        print(f"  {names[i]:>8}  {kl_full:>10.5f}  {kl_size:>10.5f}  "
              f"{kl_cf:>10.5f}  {winner:>12}")

    df_oos = pd.DataFrame(oos_rows)
    df_in.to_csv(f"{args.out_dir}/88_insample.csv", index=False)
    df_oos.to_csv(f"{args.out_dir}/88_outofsample.csv", index=False)

    print(f"\nout-of-sample over {len(df_oos)} month pairs:")
    print(f"  full model wins : {(df_oos['winner']=='full').sum()} / {len(df_oos)}")
    print(f"  size model wins : {(df_oos['winner']=='size').sum()} / {len(df_oos)}")
    print(f"  copy-fwd wins   : {(df_oos['winner']=='copy-forward').sum()} / {len(df_oos)}")
    print(f"\n  mean KL full model    : {df_oos['kl_full'].mean():.5f}")
    print(f"  mean KL size only     : {df_oos['kl_size'].mean():.5f}")
    print(f"  mean KL copy-forward  : {df_oos['kl_cf'].mean():.5f}")

    gram_adds = df_oos["kl_size"].mean() - df_oos["kl_full"].mean()
    size_adds = df_oos["kl_cf"].mean() - df_oos["kl_size"].mean()
    print(f"\n  size drift over copy-forward   : {size_adds:+.5f} bits KL reduction")
    print(f"  grammar+semantic over size     : {gram_adds:+.5f} bits KL reduction")

    print("\nreading:")
    print("  full > size > copy-forward -> grammar and semantic distance")
    print("     add signal beyond size drift. CSCS hypothesis confirmed")
    print("     dynamically in a population model.")
    print("  size > copy-forward, full ~ size -> size drift is the only")
    print("     signal. Grammar and semantics add nothing.")
    print("  all similar -> none of the fitness terms generalise.")

    print(f"\nwrote 2 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
