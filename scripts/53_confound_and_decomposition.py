#!/usr/bin/env python
"""
53_confound_and_decomposition.py

Four tests, in order of how much they can change the project.

A. DEPTH CONFOUND (can kill the script 52 result)
   d1_rare_mpd predicts switches at AUC 0.887, p=0.0025 -- but n_seqs alone
   predicts them at AUC 0.789, p=0.016. mpd is measured at fixed depth so
   n_seqs cannot mechanically inflate it, but switches happen during case
   surges and surges get sequenced hard. Residualise mpd on sequencing effort
   and re-test. Baseline: residual AUC should stay near 0.85 with p < 0.05.
   If AUC falls toward 0.65, this is surveillance intensity, not diversity.

B. MIXTURE TEST (checks the proposed mechanism)
   Claim: mpd spikes because two dissimilar populations coexist in the sample,
   not because one population became more variable. Those differ in the shape
   of the pairwise-distance distribution: a mixture is bimodal, a diffuse
   population is unimodal-and-wide. Tested by 1-D 2-means separation and
   Sarle's bimodality coefficient.

C. SPIKE MAGNITUDE vs INCUMBENT-CHALLENGER DISTANCE (explains the BA.5 miss)
   BA.5 was the only switch script 52 missed (extremity 0.520). If mpd is a
   mixture detector, spike size should scale with how far apart incumbent and
   challenger are. BA.2 -> BA.5 is a short hop; Delta -> Omicron is enormous.
   Measured clustering-free using the modal constellation before and after.

D. VOCABULARY DECOMPOSITION (serves the actual forecasting problem)
   Vocabulary size is an average of two quantities moving in opposite
   directions: labels near fixation ratchet up at sweeps and never fall,
   while polymorphic labels collapse at sweeps and rebuild within a regime.
   Only the non-fixed labels can form new constellations, so this defines the
   real candidate pool. Section D also asks which frequency class the added
   label of a new constellation comes from.

Outputs
-------
outputs/53_confound.csv        AUC before/after residualisation
outputs/53_mixture.csv         per-month distance-distribution shape
outputs/53_spike_scaling.csv   spike size vs incumbent-challenger distance
outputs/53_vocab_classes.csv   fixed / polymorphic / rare counts per month
outputs/53_added_label_class.csv  frequency class of labels entering new sets

Usage
-----
python scripts/53_confound_and_decomposition.py --min_count 3 --end_month 2024-12
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


# ----------------------------------------------------------------------------
# shared helpers
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


def pair_distances(occ, n_pairs, rng):
    """Sequence-weighted sample of pairwise symmetric-difference sizes."""
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if len(keys) < 2 or counts.sum() <= 0:
        return np.zeros(0)
    p = counts / counts.sum()
    i = rng.choice(len(keys), size=n_pairs, p=p)
    j = rng.choice(len(keys), size=n_pairs, p=p)
    return np.fromiter((len(keys[a] ^ keys[b]) for a, b in zip(i, j)),
                       dtype=float, count=n_pairs)


def auc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def perm_p(y, s, observed, n_perm, rng):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or (1 - y).sum() == 0 or np.isnan(observed):
        return np.nan
    target = abs(observed - 0.5)
    hits = 0
    for _ in range(n_perm):
        a = auc(rng.permutation(y), s)
        if not np.isnan(a) and abs(a - 0.5) >= target:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def ols_residual(y, X):
    """Residual of y on design X (constant added), NaN-safe."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    out = np.full_like(y, np.nan)
    if ok.sum() < X.shape[1] + 2:
        return out
    Xd = np.column_stack([np.ones(ok.sum()), X[ok]])
    beta, *_ = np.linalg.lstsq(Xd, y[ok], rcond=None)
    out[ok] = y[ok] - Xd @ beta
    return out


# ----------------------------------------------------------------------------
# B: 1-D two-means and bimodality
# ----------------------------------------------------------------------------

def two_means_1d(x, n_iter=50):
    """Returns (frac_upper, separation) where separation = |mu1-mu0| / pooled sd."""
    x = np.asarray(x, dtype=float)
    if x.size < 10 or np.allclose(x, x[0]):
        return np.nan, np.nan
    lo, hi = np.percentile(x, 25), np.percentile(x, 75)
    if lo == hi:
        lo, hi = x.min(), x.max()
    c0, c1 = lo, hi
    for _ in range(n_iter):
        upper = np.abs(x - c1) < np.abs(x - c0)
        if upper.all() or (~upper).all():
            break
        nc0, nc1 = x[~upper].mean(), x[upper].mean()
        if np.isclose(nc0, c0) and np.isclose(nc1, c1):
            break
        c0, c1 = nc0, nc1
    upper = np.abs(x - c1) < np.abs(x - c0)
    if upper.all() or (~upper).all():
        return np.nan, np.nan
    s0, s1 = x[~upper].std(), x[upper].std()
    n0, n1 = (~upper).sum(), upper.sum()
    pooled = np.sqrt((n0 * s0 ** 2 + n1 * s1 ** 2) / (n0 + n1)) + 1e-9
    return float(upper.mean()), float(abs(c1 - c0) / pooled)


def bimodality_coefficient(x):
    """Sarle's BC. > 0.555 is conventionally taken as evidence of bimodality."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return np.nan
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return np.nan
    g1 = ((x - m) ** 3).mean() / s ** 3
    g2 = ((x - m) ** 4).mean() / s ** 4 - 3.0
    denom = g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0:
        return np.nan
    return float((g1 ** 2 + 1) / denom)


# ----------------------------------------------------------------------------
# D: frequency classes
# ----------------------------------------------------------------------------

def node_freqs(occ):
    """Within-month sequence-weighted frequency of each label."""
    total = sum(occ.values())
    nc = defaultdict(float)
    for cs, c in occ.items():
        for lab in cs:
            nc[lab] += c
    return {lab: v / total for lab, v in nc.items()}, total


def classify(freq, fixed_thr, poly_thr):
    if freq >= fixed_thr:
        return "fixed"
    if freq >= poly_thr:
        return "polymorphic"
    return "rare"


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
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=8000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--fixed_thr", type=float, default=0.90)
    ap.add_argument("--poly_thr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    print(f"loaded {len(months)} months: {months[0][0]} .. {months[-1][0]}")

    # ---- build per-month series (rarefied mpd + distance shape) -------------
    rows = []
    dist_store = {}
    for month, occ in months:
        n_seqs = int(sum(occ.values()))
        mpds, fracs, seps, bcs = [], [], [], []
        pooled_d = []
        for _ in range(args.reps):
            sub = rarefy(occ, args.depth, args.min_count, rng)
            if sub is None:
                continue
            d = pair_distances(sub, args.pairs, rng)
            if d.size == 0:
                continue
            mpds.append(d.mean())
            f, s = two_means_1d(d)
            fracs.append(f)
            seps.append(s)
            bcs.append(bimodality_coefficient(d))
            pooled_d.append(d)
        if pooled_d:
            dist_store[month] = np.concatenate(pooled_d)
        rows.append({
            "month": month,
            "n_seqs": n_seqs,
            "rare_mpd": float(np.mean(mpds)) if mpds else np.nan,
            "upper_mode_frac": float(np.nanmean(fracs)) if fracs else np.nan,
            "mode_separation": float(np.nanmean(seps)) if seps else np.nan,
            "bimodality_coef": float(np.nanmean(bcs)) if bcs else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    df["d1_rare_mpd"] = df["rare_mpd"].diff()
    df["log_n_seqs"] = np.log(df["n_seqs"].clip(lower=1))
    df["d1_log_n_seqs"] = df["log_n_seqs"].diff()
    df["switch_next"] = df["month"].shift(-1).isin(SWITCH_MONTHS).astype(int)

    # ========================================================================
    # A. depth confound
    # ========================================================================
    print("\n" + "=" * 72)
    print("A. DEPTH CONFOUND")
    print("=" * 72)

    ev = df.iloc[:-1].copy()  # last month has no t+1
    y = ev["switch_next"].to_numpy()

    tests = {}
    tests["d1_rare_mpd (raw)"] = ev["d1_rare_mpd"].to_numpy()
    tests["n_seqs"] = ev["n_seqs"].to_numpy().astype(float)
    tests["d1_log_n_seqs"] = ev["d1_log_n_seqs"].to_numpy()
    tests["d1_rare_mpd | log_n_seqs"] = ols_residual(
        ev["d1_rare_mpd"].to_numpy(), ev[["log_n_seqs"]].to_numpy())
    tests["d1_rare_mpd | log_n_seqs + d1_log_n_seqs"] = ols_residual(
        ev["d1_rare_mpd"].to_numpy(),
        ev[["log_n_seqs", "d1_log_n_seqs"]].to_numpy())
    # reverse direction: does sequencing effort survive controlling for mpd?
    tests["n_seqs | d1_rare_mpd"] = ols_residual(
        ev["n_seqs"].to_numpy().astype(float), ev[["d1_rare_mpd"]].to_numpy())

    crows = []
    for name, s in tests.items():
        a = auc(y, s)
        if np.isnan(a):
            continue
        oriented = a if a >= 0.5 else 1 - a
        p = perm_p(y, s, a, args.n_perm, rng)
        crows.append({"test": name, "auc": oriented,
                      "direction": "+" if a >= 0.5 else "-", "perm_p": p})
    conf = pd.DataFrame(crows)
    conf.to_csv(f"{args.out_dir}/53_confound.csv", index=False)
    print(conf.round(4).to_string(index=False))

    r = np.corrcoef(ev["d1_rare_mpd"].fillna(0), ev["d1_log_n_seqs"].fillna(0))[0, 1]
    print(f"\ncorr(d1_rare_mpd, d1_log_n_seqs) = {r:+.3f}")
    print("\nread: residual AUC near 0.85 with p<0.05 -> mpd survives, real result.")
    print("      residual AUC toward 0.65 or p>0.05 -> surveillance intensity.")

    # ========================================================================
    # B. mixture test
    # ========================================================================
    print("\n" + "=" * 72)
    print("B. MIXTURE TEST: is the distance distribution bimodal at spikes?")
    print("=" * 72)

    df["is_pre_switch"] = df["month"].shift(-1).isin(SWITCH_MONTHS)
    mix = df[["month", "n_seqs", "rare_mpd", "d1_rare_mpd", "upper_mode_frac",
              "mode_separation", "bimodality_coef", "is_pre_switch"]]
    mix.to_csv(f"{args.out_dir}/53_mixture.csv", index=False)

    pre = mix[mix["is_pre_switch"]]
    other = mix[~mix["is_pre_switch"] & mix["mode_separation"].notna()]
    print("\npre-switch months:")
    print(pre.round(3).to_string(index=False))
    print(f"\nmode_separation   pre-switch {pre['mode_separation'].mean():.3f}  "
          f"vs other {other['mode_separation'].mean():.3f}")
    print(f"bimodality_coef   pre-switch {pre['bimodality_coef'].mean():.3f}  "
          f"vs other {other['bimodality_coef'].mean():.3f}  (>0.555 = bimodal)")
    print(f"upper_mode_frac   pre-switch {pre['upper_mode_frac'].mean():.3f}  "
          f"vs other {other['upper_mode_frac'].mean():.3f}")
    a_sep = auc(mix["is_pre_switch"].astype(int).to_numpy()[:-1],
                mix["mode_separation"].to_numpy()[:-1])
    if not np.isnan(a_sep):
        print(f"\nAUC of mode_separation alone: "
              f"{a_sep if a_sep >= 0.5 else 1 - a_sep:.3f}")
    print("\nread: separation and bimodality elevated pre-switch -> two populations")
    print("      coexisting (mixture). Elevated mpd without them -> one population")
    print("      simply became more variable, which is a different claim.")

    # ========================================================================
    # C. spike magnitude vs incumbent-challenger distance
    # ========================================================================
    print("\n" + "=" * 72)
    print("C. SPIKE MAGNITUDE vs INCUMBENT-CHALLENGER DISTANCE")
    print("=" * 72)

    mo = {m: occ for m, occ in months}
    order = [m for m, _ in months]
    pos = {m: i for i, m in enumerate(order)}

    def modal(occ):
        return max(occ.items(), key=lambda kv: kv[1])[0]

    srows = []
    for sm in SWITCH_MONTHS:
        if sm not in pos:
            continue
        i = pos[sm]
        if i - 2 < 0 or i + 2 >= len(order):
            continue
        before = modal(mo[order[i - 2]])
        after = modal(mo[order[i + 2]])
        d1 = df.loc[df["month"] == order[i - 1], "d1_rare_mpd"]
        srows.append({
            "switch_month": sm,
            "variant": SWITCH_NAMES.get(sm, "?"),
            "incumbent_challenger_dist": len(before ^ after),
            "modal_size_before": len(before),
            "modal_size_after": len(after),
            "d1_rare_mpd_at_t": float(d1.iloc[0]) if len(d1) else np.nan,
        })
    spike = pd.DataFrame(srows)
    spike.to_csv(f"{args.out_dir}/53_spike_scaling.csv", index=False)
    print(spike.round(3).to_string(index=False))
    if len(spike) >= 3 and spike["d1_rare_mpd_at_t"].notna().sum() >= 3:
        v = spike.dropna(subset=["d1_rare_mpd_at_t"])
        rr = np.corrcoef(v["incumbent_challenger_dist"], v["d1_rare_mpd_at_t"])[0, 1]
        print(f"\ncorr(incumbent-challenger distance, spike size) = {rr:+.3f}")
        print("read: strong positive -> the BA.5 miss is predicted by the")
        print("      mechanism (short hop, small spike), not a random failure.")

    # ========================================================================
    # D. vocabulary decomposition
    # ========================================================================
    print("\n" + "=" * 72)
    print("D. VOCABULARY DECOMPOSITION: fixed vs polymorphic vs rare")
    print("=" * 72)

    vrows = []
    freq_by_month = {}
    for month, occ in months:
        freqs, total = node_freqs(occ)
        freq_by_month[month] = freqs
        cls = defaultdict(int)
        for lab, f in freqs.items():
            cls[classify(f, args.fixed_thr, args.poly_thr)] += 1
        vrows.append({
            "month": month,
            "n_seqs": total,
            "n_fixed": cls["fixed"],
            "n_polymorphic": cls["polymorphic"],
            "n_rare": cls["rare"],
            "vocab_size": len(freqs),
            "candidate_pool": cls["polymorphic"] + cls["rare"],
        })
    vc = pd.DataFrame(vrows).sort_values("month").reset_index(drop=True)
    vc["d1_n_fixed"] = vc["n_fixed"].diff()
    vc["is_switch"] = vc["month"].isin(SWITCH_MONTHS)
    vc.to_csv(f"{args.out_dir}/53_vocab_classes.csv", index=False)
    print(vc[["month", "n_seqs", "n_fixed", "n_polymorphic", "n_rare",
              "candidate_pool", "is_switch"]].to_string(index=False))

    dec = vc["d1_n_fixed"].dropna()
    print(f"\nn_fixed: {(dec > 0).sum()} increases, {(dec < 0).sum()} decreases, "
          f"{(dec == 0).sum()} flat")
    print("  (monotone increase = ratchet; decreases mean fixation is reversible)")
    sw_jump = vc.loc[vc["is_switch"], "d1_n_fixed"].mean()
    ns_jump = vc.loc[~vc["is_switch"], "d1_n_fixed"].mean()
    print(f"mean d1_n_fixed at switch months {sw_jump:.2f} vs elsewhere {ns_jump:.2f}")
    print(f"\ncandidate pool size: median {vc['candidate_pool'].median():.0f}, "
          f"range {vc['candidate_pool'].min()}-{vc['candidate_pool'].max()}")
    print("  ^ this, not 890, is the number of labels that can form new sets")

    # ---- which class does the added label of a new constellation come from? -
    print("\n--- frequency class of labels entering new constellations ---")
    arows = []
    for i in range(len(months) - 1):
        m_t, occ_t = months[i]
        m_n, occ_n = months[i + 1]
        H_t = set(occ_t.keys())
        freqs = freq_by_month[m_t]
        counts = defaultdict(int)
        wcounts = defaultdict(int)
        n_checked = 0
        for c, w in occ_n.items():
            if c in H_t:
                continue
            added = [lab for lab in c if frozenset(c - {lab}) in H_t]
            if not added:
                continue
            n_checked += 1
            # if several routes exist, take the most common label (easiest route)
            lab = max(added, key=lambda l: freqs.get(l, 0.0))
            k = (classify(freqs[lab], args.fixed_thr, args.poly_thr)
                 if lab in freqs else "absent")
            counts[k] += 1
            wcounts[k] += w
        if n_checked == 0:
            continue
        arows.append({
            "month_t": m_t,
            "n_new_with_source": n_checked,
            "frac_fixed": counts["fixed"] / n_checked,
            "frac_polymorphic": counts["polymorphic"] / n_checked,
            "frac_rare": counts["rare"] / n_checked,
            "frac_absent": counts["absent"] / n_checked,
        })
    ac = pd.DataFrame(arows)
    ac.to_csv(f"{args.out_dir}/53_added_label_class.csv", index=False)
    print(ac.round(3).tail(24).to_string(index=False))
    print("\nmeans over all months:")
    for col in ["frac_fixed", "frac_polymorphic", "frac_rare", "frac_absent"]:
        print(f"  {col:20s} {ac[col].mean():.3f}")
    print("\nread: mass on 'rare' means new constellations are built from labels")
    print("      that are already circulating but at low frequency -- the")
    print("      candidate pool is small and enumerable, which makes the")
    print("      co-occurrence forecasting problem tractable.")

    print(f"\nwrote 5 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
