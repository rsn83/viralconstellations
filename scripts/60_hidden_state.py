#!/usr/bin/env python
"""
60_hidden_state.py

Question
--------
Script 54 found the label frequency-class transition matrix is strongly
non-stationary across regimes (chi-square p = 1e-25, 1e-12, 1e-8 on the absent,
rare and polymorphic rows). Something unobserved is switching.

The hypothesis: that something is LINEAGE COMPOSITION -- which genomic
backgrounds are circulating. A label's entry rate depends on whether a genome
exists that can carry it, so the vocabulary process is non-stationary only
because it is being marginalised over a hidden state.

The test: condition on a causal proxy for composition and re-estimate. If the
non-stationarity disappears, ONE low-dimensional hidden state explains it and a
mixture model over lineages is justified. If it survives, the hidden state is
higher-dimensional or is not composition, and building a mixture on that premise
would be building on a false one.

Leakage discipline
------------------
Script 54 failed twice and both failures are fixed here:
  - its label universe was built from all 60 months, contaminating every
    'absent' transition rate. Here the universe at month t is labels seen in
    months <= t only.
  - its regime labels came from SWITCH_MONTHS, which descends from a cluster
    partition fitted on pooled months. Here the composition state is segmented
    ONLINE from each month's own modal constellation, with no clustering, no
    pooled partition, and no lookahead.

Known variant months appear in the output for READING ONLY. They are never used
to fit or segment anything.

Composition proxies (each computed from a single month's occupancy dict)
------------------------------------------------------------------------
  modal constellation and its frequency, modal set size,
  effective number of sets covering half the sequences,
  mean pairwise distance, mean set size

Sections
--------
A. PROXIES. The per-month series, plus the online segmentation.
B. LIKELIHOOD TEST. Transition matrices fitted globally vs per composition
   segment. Reported as log-likelihood gain and BIC, against a permutation null
   of random contiguous segmentations with the same number and sizes of
   segments -- because more segments always fit better, and the question is
   whether THESE segments fit better than arbitrary ones.
C. CONTINUOUS VERSION. Rather than segmenting, regress each transition rate on
   the composition covariates directly. R^2 says how much month-to-month
   variation in the label process is explained by composition, and the
   coefficients say which aspect of composition carries it.

Outputs
-------
outputs/60_proxies.csv        per-month composition proxies and segment id
outputs/60_likelihood.csv     global vs segmented fit, with the null
outputs/60_regression.csv     per-rate regression on composition covariates

Usage
-----
python scripts/60_hidden_state.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

CLASSES = ["absent", "rare", "polymorphic", "fixed"]

# for reading the output only -- never used in fitting or segmentation
KNOWN_VARIANTS = {
    "2021-01": "Alpha", "2021-06": "Delta", "2022-01": "BA.1",
    "2022-03": "BA.2", "2022-06": "BA.5", "2023-02": "XBB",
    "2023-12": "JN.1",
}


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


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    draws = rng.multinomial(depth, counts / total)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def node_freqs(occ):
    total = float(sum(occ.values()))
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


# ----------------------------------------------------------------------------
# A. composition proxies, each from one month alone
# ----------------------------------------------------------------------------

def proxies(occ, rng, n_pairs=4000):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    p = counts / total
    order = np.argsort(-counts)

    modal = keys[order[0]]
    modal_freq = float(p[order[0]])
    cum = np.cumsum(p[order])
    n_half = int(np.searchsorted(cum, 0.5) + 1)

    if len(keys) > 1:
        i = rng.choice(len(keys), size=n_pairs, p=p)
        j = rng.choice(len(keys), size=n_pairs, p=p)
        d = np.fromiter((len(keys[a] ^ keys[b]) for a, b in zip(i, j)),
                        dtype=float, count=n_pairs)
        mpd = float(d.mean())
    else:
        mpd = 0.0

    sizes = np.array([len(k) for k in keys], dtype=float)
    return {
        "modal": modal,
        "modal_freq": modal_freq,
        "modal_size": float(len(modal)),
        "n_sets_half": float(n_half),
        "mpd": mpd,
        "mean_set_size": float((sizes * p).sum()),
        "n_sets": float(len(keys)),
    }


def online_segment(modals, thresh):
    """
    Assign a composition state id to each month, causally.

    A new state opens when the month's modal constellation is more than
    `thresh` edits from the modal constellation that opened the current state.
    Only past information is used, so this can be run in real time.
    """
    seg = np.zeros(len(modals), dtype=int)
    ref = modals[0]
    sid = 0
    for i, m in enumerate(modals):
        if len(m ^ ref) > thresh:
            sid += 1
            ref = m
        seg[i] = sid
    return seg


# ----------------------------------------------------------------------------
# B. transition matrices and likelihood
# ----------------------------------------------------------------------------

def transition_counts(f_t, f_t1, universe, poly_thr, fixed_thr):
    M = np.zeros((4, 4))
    ix = {c: i for i, c in enumerate(CLASSES)}
    for lab in universe:
        a = classify(f_t[lab], poly_thr, fixed_thr) if lab in f_t else "absent"
        b = classify(f_t1[lab], poly_thr, fixed_thr) if lab in f_t1 else "absent"
        M[ix[a], ix[b]] += 1
    return M


def multinomial_ll(counts, probs):
    """Sum of n_ij log p_ij over rows, ignoring the constant."""
    ll = 0.0
    for i in range(counts.shape[0]):
        n = counts[i]
        if n.sum() == 0:
            continue
        p = np.clip(probs[i], 1e-12, None)
        ll += float((n * np.log(p)).sum())
    return ll


def row_normalise(M, smooth=0.5):
    P = M + smooth
    return P / P.sum(axis=1, keepdims=True)


def fit_and_score(Ms, groups):
    """Log-likelihood of the transition counts under a per-group model."""
    ll = 0.0
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        agg = sum(Ms[i] for i in idx)
        P = row_normalise(agg)
        for i in idx:
            ll += multinomial_ll(Ms[i], P)
    return ll


def random_contiguous(n, n_seg, rng):
    """A random contiguous segmentation of n items into n_seg pieces."""
    if n_seg <= 1:
        return np.zeros(n, dtype=int)
    cuts = np.sort(rng.choice(np.arange(1, n), size=n_seg - 1, replace=False))
    seg = np.zeros(n, dtype=int)
    for c in cuts:
        seg[c:] += 1
    return seg


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
    ap.add_argument("--seg_thresh", type=int, default=5,
                    help="edit distance at which a new composition state opens")
    ap.add_argument("--poly_thr", type=float, default=0.10)
    ap.add_argument("--fixed_thr", type=float, default=0.90)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    print(f"loaded {len(months)} months: {names[0]} .. {names[-1]}")

    # rarefied replicates, shared by every section
    reps = {}
    for month, occ in months:
        r = []
        for _ in range(args.reps):
            sub = rarefy(occ, args.depth, args.min_count, rng)
            if sub is not None:
                r.append(sub)
        if r:
            reps[month] = r
    names = [m for m in names if m in reps]
    T = len(names)
    print(f"months clearing depth {args.depth}: {T}")

    # ========================================================================
    # A. proxies and online segmentation
    # ========================================================================
    print("\n" + "=" * 74)
    print("A. COMPOSITION PROXIES  (each from one month alone)")
    print("=" * 74)

    occ_by_month = {m: o for m, o in months}
    prox = [proxies(occ_by_month[m], rng) for m in names]

    modals = [p["modal"] for p in prox]
    seg = online_segment(modals, args.seg_thresh)

    dfp = pd.DataFrame({
        "month": names,
        "segment": seg,
        "modal_freq": [p["modal_freq"] for p in prox],
        "modal_size": [p["modal_size"] for p in prox],
        "n_sets_half": [p["n_sets_half"] for p in prox],
        "mpd": [p["mpd"] for p in prox],
        "mean_set_size": [p["mean_set_size"] for p in prox],
        "variant_note": [KNOWN_VARIANTS.get(m, "") for m in names],
    })
    dfp.to_csv(f"{args.out_dir}/60_proxies.csv", index=False)
    print(dfp.round(3).to_string(index=False))

    n_seg = len(set(seg))
    print(f"\nonline segmentation at edit threshold {args.seg_thresh}: "
          f"{n_seg} composition states")
    starts = [names[i] for i in range(T) if i == 0 or seg[i] != seg[i - 1]]
    print(f"state changes at: {starts}")
    print("(variant_note is printed for reading only; it was not used to fit"
          " or segment anything)")

    # ========================================================================
    # B. likelihood test
    # ========================================================================
    print("\n" + "=" * 74)
    print("B. DOES CONDITIONING ON COMPOSITION EXPLAIN THE NON-STATIONARITY?")
    print("=" * 74)

    # causal universe: labels in support at or before month t
    universe_at = []
    seen = set()
    for m in names:
        for sub in reps[m]:
            seen |= set(node_freqs(sub).keys())
        universe_at.append(frozenset(seen))

    Ms = []
    for t in range(T - 1):
        rt, rn = reps[names[t]], reps[names[t + 1]]
        k = min(len(rt), len(rn))
        M = np.zeros((4, 4))
        for r in range(k):
            M += transition_counts(node_freqs(rt[r]), node_freqs(rn[r]),
                                   universe_at[t], args.poly_thr, args.fixed_thr)
        Ms.append(M / k)
    seg_pairs = seg[:-1]
    n = len(Ms)

    ll_global = fit_and_score(Ms, np.zeros(n, dtype=int))
    ll_seg = fit_and_score(Ms, seg_pairs)
    k_seg = len(set(seg_pairs))
    gain = ll_seg - ll_global
    n_obs = sum(M.sum() for M in Ms)
    dparams = (k_seg - 1) * 12
    bic_gain = 2 * gain - dparams * np.log(max(n_obs, 2))

    print(f"log-likelihood, one global matrix     : {ll_global:,.1f}")
    print(f"log-likelihood, per composition state : {ll_seg:,.1f}")
    print(f"gain                                   : {gain:,.1f} "
          f"({k_seg} states, {dparams} extra parameters)")
    print(f"BIC improvement                        : {bic_gain:,.1f} "
          f"({'favours' if bic_gain > 0 else 'does NOT favour'} the "
          f"conditioned model)")

    # permutation null: random contiguous segmentations, same count
    null = np.empty(args.n_perm)
    for b in range(args.n_perm):
        g = random_contiguous(n, k_seg, rng)
        null[b] = fit_and_score(Ms, g) - ll_global
    p_emp = float((null >= gain).mean())
    print(f"\nrandom contiguous segmentations with {k_seg} states:")
    print(f"  null gain  mean {null.mean():,.1f}  sd {null.std():,.1f}  "
          f"max {null.max():,.1f}")
    print(f"  observed gain {gain:,.1f}   empirical p = {p_emp:.4f}")
    print(f"  z = {(gain - null.mean()) / (null.std() + 1e-9):.2f}")

    pd.DataFrame([{
        "ll_global": ll_global, "ll_segmented": ll_seg, "gain": gain,
        "n_states": k_seg, "extra_params": dparams, "bic_gain": bic_gain,
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
        "p_empirical": p_emp,
    }]).to_csv(f"{args.out_dir}/60_likelihood.csv", index=False)

    print("\nread:")
    print("  observed gain far above the null -> composition IS the hidden state;")
    print("     the label process is stationary once conditioned on it, and a")
    print("     mixture over lineages is the right generative structure.")
    print("  observed gain inside the null -> ANY segmentation of the months")
    print("     fits this well, so composition explains nothing specific and the")
    print("     hidden state is higher-dimensional or is something else.")

    # ========================================================================
    # C. continuous version
    # ========================================================================
    print("\n" + "=" * 74)
    print("C. RATES REGRESSED ON COMPOSITION DIRECTLY  (no segmentation)")
    print("=" * 74)

    ix = {c: i for i, c in enumerate(CLASSES)}
    targets = {
        "p_absent_to_rare": (ix["absent"], ix["rare"]),
        "p_rare_to_poly": (ix["rare"], ix["polymorphic"]),
        "p_rare_to_absent": (ix["rare"], ix["absent"]),
        "p_poly_to_fixed": (ix["polymorphic"], ix["fixed"]),
        "p_fixed_to_lower": None,
    }
    covs = ["modal_freq", "modal_size", "n_sets_half", "mpd", "mean_set_size",
            "time_index"]
    dfp["time_index"] = np.arange(len(dfp), dtype=float)
    Xc = dfp[covs].to_numpy()[:-1]
    Xc = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-9)
    Xd = np.column_stack([np.ones(len(Xc)), Xc])

    rrows = []
    for name, cell in targets.items():
        y = []
        for M in Ms:
            P = row_normalise(M)
            if cell is None:
                y.append(1 - P[ix["fixed"], ix["fixed"]])
            else:
                y.append(P[cell[0], cell[1]])
        y = np.log(np.clip(np.array(y), 1e-6, 1 - 1e-6) /
                   (1 - np.clip(np.array(y), 1e-6, 1 - 1e-6)))
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        pred = Xd @ beta
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        row = {"rate": name, "r2": r2}
        for c, b in zip(covs, beta[1:]):
            row[f"beta_{c}"] = float(b)
        rrows.append(row)

    reg = pd.DataFrame(rrows)
    reg.to_csv(f"{args.out_dir}/60_regression.csv", index=False)
    print(reg.round(4).to_string(index=False))
    print("\nread: high R^2 means the rate is a deterministic function of")
    print("      composition rather than an independent process. The largest")
    print("      coefficient says which aspect of composition drives it --")
    print("      modal_freq is 'how dominant is the incumbent', mpd is 'are two")
    print("      populations coexisting', mean_set_size is 'how derived is the")
    print("      population'.")

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
