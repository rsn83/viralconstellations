#!/usr/bin/env python
"""
86_capacity_ladder.py

The question this settles
------------------------
Nine model families all landed at what a mutation's frequency and recency give.
Two explanations remain, and they call for different next steps:

  (a) the features are exhausted -- the information simply is not in them, and
      no amount of extra modelling capacity will help
  (b) the models were too weak -- a more flexible model on the same features
      would do better

This distinguishes them by fitting a LADDER of models of increasing capacity on
exactly the same features, and reporting held-out log-loss for each. If the
ladder plateaus, (a). If it keeps improving, (b).

Why log-loss and not AP
-----------------------
Held-out log-loss of any predictor is an upper bound on H(Y | X), the
conditional entropy of the outcome given the features. The lowest value the
ladder reaches is therefore our best estimate of the floor -- the best any model
using these features could achieve. AP measures ranking and does not have this
interpretation.

Reported per origin, in bits:
  H(Y)          entropy of the outcome alone, what you have knowing nothing
  log-loss      for each rung of the ladder, evaluated out of sample
  extracted     H(Y) minus the best log-loss: information the features supply
  remaining     the best log-loss itself: uncertainty no model on these
                features can remove

The ladder
----------
  constant             predict the training base rate. This is the floor of
                       ignorance, and equals H(Y) when the base rate is stable.
  logistic             the model used in scripts 58 and 67
  logistic + squares + pairwise products
  boosted trees, 50 / 200 / 800 trees, increasing depth
A shuffled control is included: the outcome is permuted, so every rung must
return exactly H(Y). If a rung beats H(Y) on shuffled data it is leaking, and
the whole table is void.

What this cannot show
---------------------
It cannot prove no method can beat copy-forward. It bounds only predictors using
THESE features. A plateau is evidence that the feature set is exhausted, not
that the data is.

Usage
-----
python scripts/86_capacity_ladder.py --min_count 3 --end_month 2024-12
python scripts/86_capacity_ladder.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

FEATURES = ["log_rho_m", "m_months_in_support", "c_size", "log_freq_c",
            "log_growth_c", "pmi_mean", "pmi_max", "pmi_min", "cooc_frac"]


# ----------------------------------------------------------------------------
# log-loss in bits
# ----------------------------------------------------------------------------

def logloss_bits(y, p):
    """Mean negative log2 likelihood. Equals H(Y) for a constant base-rate
    predictor, so the numbers are directly comparable to an entropy."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log2(p) + (1 - y) * np.log2(1 - p)).mean())


def entropy_bits(q):
    q = float(np.clip(q, 1e-12, 1 - 1e-12))
    return float(-(q * np.log2(q) + (1 - q) * np.log2(1 - q)))


def self_test():
    print("checking the measures")
    rng = np.random.default_rng(0)

    # a constant predictor at the true rate scores exactly the entropy
    q = 0.002
    y = (rng.random(2_000_000) < q).astype(int)
    ll = logloss_bits(y, np.full(len(y), y.mean()))
    h = entropy_bits(y.mean())
    assert abs(ll - h) < 1e-6, (ll, h)
    print(f"  constant predictor scores exactly H(Y) "
          f"({ll:.6f} bits)      ok")

    # a perfect predictor scores near zero
    assert logloss_bits(y[:1000], y[:1000] * 0.999 + 0.0005) < 0.02
    print("  a perfect predictor scores near zero              ok")

    # a predictor that is confidently wrong scores worse than ignorance
    bad = logloss_bits(np.array([1, 1, 0, 0]), np.array([.01, .01, .99, .99]))
    assert bad > entropy_bits(0.5)
    print(f"  confidently wrong scores worse than ignorance "
          f"({bad:.2f})  ok")

    # on shuffled labels a flexible model must not beat H(Y)
    if HAVE_SK:
        n = 40000
        X = rng.normal(size=(n, 6))
        ysh = (rng.random(n) < 0.02).astype(int)
        tr, te = slice(0, n // 2), slice(n // 2, n)
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        g.fit(X[tr], ysh[tr])
        ll_sh = logloss_bits(ysh[te], g.predict_proba(X[te])[:, 1])
        h_sh = entropy_bits(ysh[tr].mean())
        print(f"  boosted trees on shuffled labels: {ll_sh:.5f} vs "
              f"H(Y) {h_sh:.5f}")
        assert ll_sh >= h_sh - 0.002, (ll_sh, h_sh)
        print("     -> does not beat ignorance, so no leakage       ok")

        # and on a genuinely informative feature it must beat H(Y) clearly
        yin = (rng.random(n) < 1 / (1 + np.exp(-(2 * X[:, 0] - 3)))).astype(int)
        g2 = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        g2.fit(X[tr], yin[tr])
        ll_in = logloss_bits(yin[te], g2.predict_proba(X[te])[:, 1])
        h_in = entropy_bits(yin[tr].mean())
        assert ll_in < h_in - 0.05, (ll_in, h_in)
        print(f"  on an informative feature: {ll_in:.4f} vs H(Y) "
              f"{h_in:.4f}   ok")
    print("all checks passed\n")


# ----------------------------------------------------------------------------
# data and candidate construction (same target as scripts 58 and 67)
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


def label_counts(occ):
    nc = defaultdict(float)
    for cs, w in occ.items():
        for l in cs:
            nc[l] += w
    return nc


def build(occ_t, occ_next, prev_occ, months_in_support, PMI, CO,
          labels_t, lab_index, max_sets):
    sets = sorted(occ_t.items(), key=lambda kv: -kv[1])[:max_sets]
    csets = [c for c, _ in sets]
    cw = np.array([w for _, w in sets], dtype=float)
    tot = float(sum(occ_t.values()))
    freq_c = cw / tot
    growth = np.array([np.log((occ_t[c] + 1.0) / (prev_occ.get(c, 0.0) + 1.0))
                       for c in csets])
    lc = label_counts(occ_t)
    rho = np.array([lc[l] / tot for l in labels_t])
    mis = np.array([months_in_support.get(l, 0) for l in labels_t], dtype=float)

    nlab = len(labels_t)
    B = np.zeros((len(csets), nlab), dtype=bool)
    for i, c in enumerate(csets):
        for l in c:
            j = lab_index.get(l)
            if j is not None:
                B[i, j] = True
    Bf = B.astype(np.float32)
    csize = np.array([len(c) for c in csets], dtype=float)
    npool = np.maximum(Bf.sum(axis=1), 1.0)
    pmi_mean = (Bf @ PMI) / npool[:, None]
    cooc_frac = (Bf @ (CO > 0).astype(np.float32)) / npool[:, None]
    pmi_max = np.full((len(csets), nlab), -10.0, dtype=np.float32)
    pmi_min = np.full((len(csets), nlab), 10.0, dtype=np.float32)
    for i in range(len(csets)):
        idx = np.flatnonzero(B[i])
        if idx.size:
            sub = PMI[idx]
            pmi_max[i] = sub.max(axis=0)
            pmi_min[i] = sub.min(axis=0)

    ii, jj = np.nonzero(~B)
    H_t = set(occ_t.keys())
    y = np.zeros(ii.size, dtype=np.int8)
    for k in range(ii.size):
        x = frozenset(csets[ii[k]] | {labels_t[jj[k]]})
        if x in occ_next and x not in H_t:
            y[k] = 1
    X = np.column_stack([
        np.log(np.clip(rho[jj], 1e-9, None)), mis[jj], csize[ii],
        np.log(np.clip(freq_c[ii], 1e-9, None)), growth[ii],
        pmi_mean[ii, jj], pmi_max[ii, jj], pmi_min[ii, jj], cooc_frac[ii, jj],
    ]).astype(np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_sets", type=int, default=800)
    ap.add_argument("--min_train", type=int, default=12)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return
    if not HAVE_SK:
        raise SystemExit("this script needs scikit-learn")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}\n")

    months_in_support = defaultdict(int)
    cache, rows = {}, []

    for t in range(T - 1):
        occ_t, occ_n = occ[names[t]], occ[names[t + 1]]
        prev = occ[names[t - 1]] if t > 0 else {}
        lc = label_counts(occ_t)
        labels_t = sorted(lc.keys(), key=str)
        lab_index = {l: j for j, l in enumerate(labels_t)}
        for l in labels_t:
            months_in_support[l] += 1

        nlab = len(labels_t)
        CO = np.zeros((nlab, nlab), dtype=np.float32)
        marg = np.zeros(nlab, dtype=np.float32)
        wtot = 0.0
        for j in range(t + 1):
            for cs, w in occ[names[j]].items():
                ix = [lab_index[l] for l in cs if l in lab_index]
                if not ix:
                    continue
                wtot += w
                a = np.array(ix)
                marg[a] += w
                CO[np.ix_(a, a)] += w
        if wtot <= 0:
            continue
        pm = np.clip(marg / wtot, 1e-9, None)
        PMI = np.log(np.clip(CO / wtot, 1e-12, None) / np.outer(pm, pm))
        np.fill_diagonal(PMI, 0.0)

        cache[t] = build(occ_t, occ_n, prev, months_in_support, PMI, CO,
                         labels_t, lab_index, args.max_sets)
        if t < args.min_train:
            continue

        Xs = [cache[j][0] for j in range(max(0, t - args.window), t)
              if j in cache]
        ys = [cache[j][1] for j in range(max(0, t - args.window), t)
              if j in cache]
        if not Xs:
            continue
        Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
        Xte, yte = cache[t]
        if ytr.sum() == 0 or yte.sum() == 0:
            continue

        base = float(ytr.mean())
        h_y = entropy_bits(float(yte.mean()))
        res = {"origin": names[t], "target": names[t + 1],
               "n_test": len(yte), "n_pos": int(yte.sum()),
               "H_Y": h_y, "n_train": len(ytr)}

        res["constant"] = logloss_bits(yte, np.full(len(yte), base))

        lr = LogisticRegression(max_iter=400, C=1.0)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-9] = 1.0
        lr.fit((Xtr - mu) / sd, ytr)
        res["logistic"] = logloss_bits(
            yte, lr.predict_proba((Xte - mu) / sd)[:, 1])

        def expand(A):
            sq = A ** 2
            k = A.shape[1]
            prods = [A[:, i] * A[:, j] for i in range(k) for j in range(i + 1, k)]
            return np.column_stack([A, sq] + prods)
        Etr, Ete = expand((Xtr - mu) / sd), expand((Xte - mu) / sd)
        lr2 = LogisticRegression(max_iter=400, C=1.0)
        lr2.fit(Etr, ytr)
        res["logistic_quadratic"] = logloss_bits(
            yte, lr2.predict_proba(Ete)[:, 1])

        # Early stopping on an internal validation split. Without it the larger
        # models overfit the 0.2% base rate and score WORSE than the logistic,
        # which would make the ladder a test of optimisation rather than of
        # capacity. With it, each rung gets a fair chance to use its capacity.
        for n_iter, depth, tag in ((50, 4, "trees_50"),
                                   (400, 6, "trees_400"),
                                   (2000, None, "trees_2000")):
            g = HistGradientBoostingClassifier(
                max_iter=n_iter, max_depth=depth, learning_rate=0.05,
                early_stopping=True, validation_fraction=0.2, n_iter_no_change=20,
                random_state=0)
            g.fit(Xtr, ytr)
            res[tag] = logloss_bits(yte, g.predict_proba(Xte)[:, 1])
            res[tag + "_used_iters"] = int(g.n_iter_)

        # shuffled control: the outcome permuted, so nothing can beat H(Y)
        ysh = rng.permutation(ytr)
        gsh = HistGradientBoostingClassifier(
            max_iter=400, early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=20, random_state=0)
        gsh.fit(Xtr, ysh)
        res["trees_on_shuffled"] = logloss_bits(
            yte, gsh.predict_proba(Xte)[:, 1])

        rungs = ["constant", "logistic", "logistic_quadratic",
                 "trees_50", "trees_400", "trees_2000"]
        best = min(res[k] for k in rungs)
        res["best"] = best
        res["extracted_bits"] = h_y - best
        res["extracted_share"] = (h_y - best) / h_y if h_y > 0 else np.nan
        res["logistic_gap"] = res["logistic"] - best
        rows.append(res)
        print(f"  {names[t]}: H(Y) {h_y:.5f}  logistic {res['logistic']:.5f}  "
              f"best {best:.5f}  shuffled {res['trees_on_shuffled']:.5f}")

        for j in list(cache):
            if j < t - args.window:
                del cache[j]

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/86_capacity_ladder.csv", index=False)

    rungs = ["constant", "logistic", "logistic_quadratic",
             "trees_50", "trees_400", "trees_2000", "trees_on_shuffled"]
    print("\n" + "=" * 84)
    print("HELD-OUT LOG-LOSS BY MODEL CAPACITY  (bits, lower is better)")
    print("=" * 84)
    summ = pd.DataFrame({
        "rung": rungs,
        "mean_logloss": [df[r].mean() for r in rungs],
        "vs_H_Y": [df[r].mean() - df["H_Y"].mean() for r in rungs],
        "beats_constant_in": [float((df[r] < df["constant"]).mean())
                              for r in rungs],
    })
    print(f"mean H(Y) over origins: {df['H_Y'].mean():.6f} bits")
    print(summ.round(6).to_string(index=False))

    print("\nsummary:")
    print(f"  best rung reaches      : {df['best'].mean():.6f} bits")
    print(f"  information extracted  : {df['extracted_bits'].mean():.6f} bits "
          f"({df['extracted_share'].mean():.3f} of H(Y))")
    print(f"  logistic leaves on the table: "
          f"{df['logistic_gap'].mean():.6f} bits")
    print(f"  shuffled control       : "
          f"{df['trees_on_shuffled'].mean():.6f} bits "
          f"(must be >= H(Y) = {df['H_Y'].mean():.6f})")

    ladder = ["logistic", "logistic_quadratic", "trees_50", "trees_400",
              "trees_2000"]
    vals = {r: df[r].mean() for r in ladder}
    winner = min(vals, key=vals.get)
    gain = vals["logistic"] - vals[winner]
    extracted_by_logistic = df["H_Y"].mean() - vals["logistic"]
    print(f"\n  best rung on the ladder: {winner} at {vals[winner]:.6f} bits")
    print(f"  it improves on the logistic by {gain:.6f} bits")
    if extracted_by_logistic > 0:
        print(f"  the logistic already extracted "
              f"{extracted_by_logistic:.6f} bits, so the extra capacity adds "
              f"{gain / extracted_by_logistic:.1%} on top")
    if winner == "logistic":
        print("  -> no rung beat the logistic: the ladder is flat")
    print("\n  iterations actually used before early stopping:")
    for tag in ("trees_50", "trees_400", "trees_2000"):
        col = tag + "_used_iters"
        if col in df:
            print(f"    {tag}: {df[col].mean():.0f} of "
                  f"{tag.split('_')[1]} available")
    print("    (a model stopping far short of its budget has stopped")
    print("     improving on held-out data, which is itself a plateau)")
    print("\nreading:")
    print("  the ladder flattening -> the FEATURES are exhausted, and more")
    print("     modelling capacity on them will not help")
    print("  the ladder still falling at the largest model -> the earlier")
    print("     models were too weak, and this is worth pursuing")
    print("  the shuffled control below H(Y) -> something leaks, table void")
    print("\nscope: this bounds predictors using THESE features. It says")
    print("nothing about predictors with information from outside them.")

    print(f"\nwrote outputs/86_capacity_ladder.csv")


if __name__ == "__main__":
    main()
