#!/usr/bin/env python
"""
54_backbone_periphery.py

The structure suggested by scripts 51-53
----------------------------------------
Rarefied vocabulary is flat (~110-120 labels, ~18 in and ~18 out per month,
Jaccard ~0.75) while mean labels per sequence rises monotonically
(1.06 -> 55.5) and never falls, not even at Omicron. Meanwhile n_fixed tracks
mean_set_size with a gap of roughly 3.

Reading: every sequence is a BACKBONE of near-fixed labels plus a small
PERIPHERY of variable ones. The backbone ratchets up in steps at sweeps; the
periphery is a turnover equilibrium with no trend. If that holds, the
combinatorial problem is not 890 labels or even 220 -- it is which ~3 labels
from a periphery of ~100 sit on the current backbone.

This script tests that reading and fits the label-level process.

Sections
--------
A. PER-SEQUENCE LOAD. Split each sequence's label count into backbone and
   periphery. Report the weighted distribution of periphery load per month and
   test it for trend. Prediction: backbone rises monotonically, periphery load
   is stationary at ~3.

B. THRESHOLD ROBUSTNESS. The "gap of 3" was measured at a 0.9 fixation cut.
   Sweep 0.50 / 0.75 / 0.90 / 0.95 / 0.99 and check the decomposition is not an
   artefact of where the line was drawn.

C. FREQUENCY-CLASS MARKOV CHAIN. Estimate transitions over
   absent -> rare -> polymorphic -> fixed. This is the label-level process:
   entry rate into rare, escape probability rare->polymorphic (the selection
   quantity), fixation probability, and the reverse flow when a sweep unfixes
   an old backbone. Five numbers, fitted before any learned model.

D. STATIONARITY. Are those transitions constant, or do they differ between
   regimes and at switch months? A non-stationary chain means the label process
   itself changes at sweeps, which would matter for any forecaster built on it.

All frequency classes are assigned on RAREFIED samples so months at different
sequencing depths are comparable (script 51: raw vocabulary size is largely a
readout of sequencing effort).

Outputs
-------
outputs/54_load.csv               per-month backbone / periphery decomposition
outputs/54_threshold_sweep.csv    the same under five fixation thresholds
outputs/54_transitions.csv        pooled class transition matrix
outputs/54_transitions_regime.csv per-regime transition matrices
outputs/54_class_rates.csv        per-month derived rates

Usage
-----
python scripts/54_backbone_periphery.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

SWITCH_MONTHS = ["2021-01", "2021-06", "2022-01", "2022-03", "2022-06", "2023-02"]
SWITCH_NAMES = {
    "2021-01": "Alpha", "2021-06": "Delta", "2022-01": "Omicron BA.1",
    "2022-03": "BA.2", "2022-06": "BA.5", "2023-02": "XBB",
}

CLASSES = ["absent", "rare", "polymorphic", "fixed"]


# ----------------------------------------------------------------------------
# helpers
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


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    draws = rng.multinomial(depth, counts / total)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def node_freqs(occ):
    total = sum(occ.values())
    nc = defaultdict(float)
    for cs, c in occ.items():
        for lab in cs:
            nc[lab] += c
    return {lab: v / total for lab, v in nc.items()}


def classify(f, poly_thr, fixed_thr):
    if f >= fixed_thr:
        return "fixed"
    if f >= poly_thr:
        return "polymorphic"
    return "rare"


def weighted_quantiles(values, weights, qs):
    """Quantiles of a discrete distribution given by (value, weight) pairs."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0 or w.sum() <= 0:
        return [np.nan] * len(qs)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    return [float(v[np.searchsorted(cw, q, side="left").clip(0, v.size - 1)])
            for q in qs]


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 4:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def regime_of(month):
    """Index of the most recent switch at or before this month."""
    k = 0
    for i, sm in enumerate(SWITCH_MONTHS, start=1):
        if month >= sm:
            k = i
    return k


REGIME_NAMES = ["pre-Alpha", "Alpha", "Delta", "BA.1", "BA.2", "BA.5", "XBB+"]


# ----------------------------------------------------------------------------
# A / B: backbone-periphery decomposition
# ----------------------------------------------------------------------------

def decompose(occ, fixed_thr, poly_thr):
    """Split each sequence's labels into backbone and periphery, count-weighted."""
    freqs = node_freqs(occ)
    backbone = {lab for lab, f in freqs.items() if f >= fixed_thr}
    sizes, peri, miss, weights = [], [], [], []
    for cs, w in occ.items():
        nb = len(cs & backbone)
        sizes.append(len(cs))
        peri.append(len(cs) - nb)
        miss.append(len(backbone) - nb)   # backbone labels this sequence lacks
        weights.append(w)
    weights = np.array(weights, dtype=float)
    tot = weights.sum()
    q = weighted_quantiles(peri, weights, [0.25, 0.5, 0.75, 0.95])
    mean_peri = float(np.dot(peri, weights) / tot)
    return {
        "backbone_size": len(backbone),
        "vocab_size": len(freqs),
        "periphery_pool": len(freqs) - len(backbone),
        "mean_set_size": float(np.dot(sizes, weights) / tot),
        "mean_periphery": mean_peri,
        "sd_periphery": float(np.sqrt(
            np.dot((np.array(peri) - mean_peri) ** 2, weights) / tot)),
        "q25_periphery": q[0], "median_periphery": q[1],
        "q75_periphery": q[2], "q95_periphery": q[3],
        "mean_missing_backbone": float(np.dot(miss, weights) / tot),
        "frac_seqs_zero_periphery": float(
            weights[np.array(peri) == 0].sum() / tot),
    }


# ----------------------------------------------------------------------------
# C / D: class transition matrix
# ----------------------------------------------------------------------------

def transition_counts(f_t, f_t1, universe, poly_thr, fixed_thr):
    """Counts over CLASSES x CLASSES for one month pair."""
    M = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    ix = {c: i for i, c in enumerate(CLASSES)}
    for lab in universe:
        a = classify(f_t[lab], poly_thr, fixed_thr) if lab in f_t else "absent"
        b = classify(f_t1[lab], poly_thr, fixed_thr) if lab in f_t1 else "absent"
        M[ix[a], ix[b]] += 1
    return M


def row_normalise(M):
    P = M.copy()
    rs = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(rs > 0, P / rs, np.nan)
    return P


def stationary_dist(P):
    """Left eigenvector for eigenvalue 1, NaN-safe."""
    if np.isnan(P).any():
        return np.full(P.shape[0], np.nan)
    vals, vecs = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(vecs[:, i])
    if v.sum() == 0:
        return np.full(P.shape[0], np.nan)
    v = v / v.sum()
    return np.abs(v)


def chi2_homogeneity(mats):
    """Chi-square across regimes, per source class. Returns (stat, df) per row."""
    out = []
    for r, cname in enumerate(CLASSES):
        rows = np.array([M[r] for M in mats if M[r].sum() > 0])
        if rows.shape[0] < 2:
            out.append((cname, np.nan, np.nan))
            continue
        tot = rows.sum()
        exp = np.outer(rows.sum(axis=1), rows.sum(axis=0)) / tot
        ok = exp > 0
        stat = float((((rows - exp) ** 2)[ok] / exp[ok]).sum())
        dfree = (rows.shape[0] - 1) * (rows.shape[1] - 1)
        out.append((cname, stat, dfree))
    return out


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
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--fixed_thr", type=float, default=0.90)
    ap.add_argument("--poly_thr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    print(f"loaded {len(months)} months: {months[0][0]} .. {months[-1][0]}")

    # pre-draw rarefied replicates once; reused by every section so that A-D
    # describe the same samples
    rare_reps = {}
    for month, occ in months:
        reps = []
        for _ in range(args.reps):
            sub = rarefy(occ, args.depth, args.min_count, rng)
            if sub is not None:
                reps.append(sub)
        if reps:
            rare_reps[month] = reps
    usable = [m for m, _ in months if m in rare_reps]
    print(f"months clearing depth {args.depth}: {len(usable)} / {len(months)}")

    # ========================================================================
    # A. per-sequence load
    # ========================================================================
    print("\n" + "=" * 74)
    print("A. BACKBONE / PERIPHERY DECOMPOSITION")
    print("=" * 74)

    rows = []
    for month, occ in months:
        if month not in rare_reps:
            continue
        ds = [decompose(sub, args.fixed_thr, args.poly_thr)
              for sub in rare_reps[month]]
        row = {"month": month, "n_seqs": int(sum(occ.values())),
               "is_switch": month in SWITCH_MONTHS,
               "regime": REGIME_NAMES[regime_of(month)]}
        for k in ds[0]:
            row[k] = float(np.mean([d[k] for d in ds]))
        rows.append(row)
    load = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    load["gap"] = load["mean_set_size"] - load["backbone_size"]
    load.to_csv(f"{args.out_dir}/54_load.csv", index=False)

    show = ["month", "backbone_size", "periphery_pool", "mean_set_size",
            "mean_periphery", "median_periphery", "q95_periphery",
            "sd_periphery", "mean_missing_backbone", "is_switch"]
    print(load[show].round(2).to_string(index=False))

    idx = np.arange(len(load))
    print(f"\ntrend (Spearman vs time)")
    print(f"  backbone_size  : {spearman(idx, load['backbone_size']):+.3f}"
          "   expected strongly positive (ratchet)")
    print(f"  mean_periphery : {spearman(idx, load['mean_periphery']):+.3f}"
          "   expected near zero (stationary)")
    print(f"  periphery_pool : {spearman(idx, load['periphery_pool']):+.3f}")
    d = load["backbone_size"].diff().dropna()
    print(f"\nbackbone_size: {(d > 0).sum()} up, {(d < 0).sum()} down, "
          f"{(d == 0).sum()} flat")
    print(f"  decreases at switch months: "
          f"{int((load['is_switch'] & (load['backbone_size'].diff() < 0)).sum())} "
          f"of {int((load['backbone_size'].diff() < 0).sum())}")
    late = load[load["month"] >= "2022-06"]
    print(f"\nsteady state (2022-06 on): mean_periphery "
          f"{late['mean_periphery'].mean():.2f} "
          f"(sd across months {late['mean_periphery'].std():.2f}), "
          f"median {late['median_periphery'].median():.1f}, "
          f"q95 {late['q95_periphery'].mean():.1f}")
    print(f"periphery pool available: {late['periphery_pool'].mean():.0f} labels")
    print("\nread: a small stationary periphery on a ratcheting backbone means the")
    print("      assembly problem is 'which few of ~N periphery labels sit on the")
    print("      current backbone', not a search over the whole vocabulary.")

    # ========================================================================
    # B. threshold robustness
    # ========================================================================
    print("\n" + "=" * 74)
    print("B. THRESHOLD ROBUSTNESS")
    print("=" * 74)

    srows = []
    for thr in (0.50, 0.75, 0.90, 0.95, 0.99):
        for month, occ in months:
            if month not in rare_reps or month < "2022-06":
                continue
            ds = [decompose(sub, thr, args.poly_thr) for sub in rare_reps[month]]
            srows.append({
                "fixed_thr": thr, "month": month,
                "backbone_size": float(np.mean([d["backbone_size"] for d in ds])),
                "mean_periphery": float(np.mean([d["mean_periphery"] for d in ds])),
                "periphery_pool": float(np.mean([d["periphery_pool"] for d in ds])),
                "mean_set_size": float(np.mean([d["mean_set_size"] for d in ds])),
            })
    sw = pd.DataFrame(srows)
    sw.to_csv(f"{args.out_dir}/54_threshold_sweep.csv", index=False)
    agg = sw.groupby("fixed_thr").agg(
        backbone=("backbone_size", "mean"),
        periphery_load=("mean_periphery", "mean"),
        periphery_pool=("periphery_pool", "mean"),
        set_size=("mean_set_size", "mean"),
        load_sd_across_months=("mean_periphery", "std"),
    ).reset_index()
    print("(2022-06 onward)")
    print(agg.round(2).to_string(index=False))
    print("\nread: if periphery load stays small and roughly flat across thresholds,")
    print("      the decomposition is real. If it tracks the threshold directly,")
    print("      'backbone' is just wherever the line was drawn.")

    # ========================================================================
    # C. frequency-class Markov chain
    # ========================================================================
    print("\n" + "=" * 74)
    print("C. FREQUENCY-CLASS MARKOV CHAIN")
    print("=" * 74)

    universe = set()
    for month in usable:
        for sub in rare_reps[month]:
            universe |= set(node_freqs(sub).keys())
    universe = sorted(universe, key=str)
    print(f"label universe (ever detected at depth {args.depth}): {len(universe)}")

    pooled = np.zeros((4, 4))
    per_regime = defaultdict(lambda: np.zeros((4, 4)))
    rate_rows = []

    for i in range(len(usable) - 1):
        m_t, m_n = usable[i], usable[i + 1]
        reps_t, reps_n = rare_reps[m_t], rare_reps[m_n]
        n = min(len(reps_t), len(reps_n))
        M = np.zeros((4, 4))
        for r in range(n):
            M += transition_counts(node_freqs(reps_t[r]), node_freqs(reps_n[r]),
                                   universe, args.poly_thr, args.fixed_thr)
        M /= n
        pooled += M
        per_regime[REGIME_NAMES[regime_of(m_t)]] += M

        P = row_normalise(M)
        ix = {c: k for k, c in enumerate(CLASSES)}
        rate_rows.append({
            "month_t": m_t, "month_t1": m_n,
            "is_pre_switch": m_n in SWITCH_MONTHS,
            "p_absent_to_rare": P[ix["absent"], ix["rare"]],
            "p_rare_to_poly": P[ix["rare"], ix["polymorphic"]],
            "p_rare_to_absent": P[ix["rare"], ix["absent"]],
            "p_poly_to_fixed": P[ix["polymorphic"], ix["fixed"]],
            "p_poly_to_rare": P[ix["polymorphic"], ix["rare"]],
            "p_fixed_to_lower": 1 - P[ix["fixed"], ix["fixed"]],
            "n_rare": M[ix["rare"]].sum(),
            "n_poly": M[ix["polymorphic"]].sum(),
            "n_fixed": M[ix["fixed"]].sum(),
        })

    rates = pd.DataFrame(rate_rows)
    rates.to_csv(f"{args.out_dir}/54_class_rates.csv", index=False)

    P = row_normalise(pooled)
    tm = pd.DataFrame(P, index=[f"from {c}" for c in CLASSES],
                      columns=[f"to {c}" for c in CLASSES])
    tm.to_csv(f"{args.out_dir}/54_transitions.csv")
    print("\npooled transition matrix (row = class at t, entries = P(class at t+1))")
    print(tm.round(4).to_string())

    ix = {c: k for k, c in enumerate(CLASSES)}
    print("\nthe five numbers:")
    print(f"  entry     P(absent -> rare)        = {P[ix['absent'], ix['rare']]:.5f}")
    print(f"  escape    P(rare -> polymorphic)   = {P[ix['rare'], ix['polymorphic']]:.4f}"
          "   <- the selection quantity")
    print(f"  loss      P(rare -> absent)        = {P[ix['rare'], ix['absent']]:.4f}")
    print(f"  fixation  P(polymorphic -> fixed)  = {P[ix['polymorphic'], ix['fixed']]:.4f}")
    print(f"  unfixing  P(fixed -> not fixed)    = {1 - P[ix['fixed'], ix['fixed']]:.4f}")
    exp_rare_life = (1 / (1 - P[ix["rare"], ix["rare"]])
                     if P[ix["rare"], ix["rare"]] < 1 else np.inf)
    print(f"\n  expected months spent in 'rare' before leaving: {exp_rare_life:.2f}")
    sd = stationary_dist(P)
    print("  implied stationary distribution: " +
          ", ".join(f"{c} {v:.4f}" for c, v in zip(CLASSES, sd)))
    print("\n  compare stationary 'rare' mass x universe against the observed")
    print(f"  periphery pool ({late['periphery_pool'].mean():.0f}); a large mismatch")
    print("  means the chain is not at equilibrium over this window.")

    # ========================================================================
    # D. stationarity
    # ========================================================================
    print("\n" + "=" * 74)
    print("D. STATIONARITY")
    print("=" * 74)

    reg_rows = []
    mats = []
    for name in REGIME_NAMES:
        if name not in per_regime:
            continue
        M = per_regime[name]
        if M.sum() == 0:
            continue
        mats.append(M)
        Pr = row_normalise(M)
        reg_rows.append({
            "regime": name,
            "n_month_pairs": int(round(M.sum() / len(universe))),
            "p_absent_to_rare": Pr[ix["absent"], ix["rare"]],
            "p_rare_to_poly": Pr[ix["rare"], ix["polymorphic"]],
            "p_rare_to_absent": Pr[ix["rare"], ix["absent"]],
            "p_poly_to_fixed": Pr[ix["polymorphic"], ix["fixed"]],
            "p_fixed_to_lower": 1 - Pr[ix["fixed"], ix["fixed"]],
        })
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(f"{args.out_dir}/54_transitions_regime.csv", index=False)
    print(reg.round(5).to_string(index=False))

    print("\nchi-square homogeneity across regimes, by source class:")
    try:
        from scipy.stats import chi2 as _chi2
        have_scipy = True
    except Exception:
        have_scipy = False
    for cname, stat, dfree in chi2_homogeneity(mats):
        if np.isnan(stat):
            print(f"  {cname:12s} insufficient data")
        elif have_scipy:
            p = float(_chi2.sf(stat, dfree))
            print(f"  {cname:12s} chi2 {stat:10.1f}  df {dfree:3d}  p {p:.2e}")
        else:
            print(f"  {cname:12s} chi2 {stat:10.1f}  df {dfree:3d}  "
                  f"(ratio {stat / dfree:.2f}; >>1 means non-stationary)")

    print("\nswitch vs non-switch month pairs:")
    for col in ["p_absent_to_rare", "p_rare_to_poly", "p_poly_to_fixed",
                "p_fixed_to_lower"]:
        a = rates.loc[rates["is_pre_switch"], col].mean()
        b = rates.loc[~rates["is_pre_switch"], col].mean()
        print(f"  {col:20s} pre-switch {a:.5f}   other {b:.5f}   "
              f"ratio {a / b if b else float('nan'):.2f}")

    print("\nread: homogeneous rows -> one stationary label process, and a single")
    print("      chain is a legitimate null model for the vocabulary. Rows that")
    print("      differ by regime -> the process itself changes at sweeps, and any")
    print("      forecaster must condition on regime rather than pool across it.")

    print(f"\nwrote 5 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
