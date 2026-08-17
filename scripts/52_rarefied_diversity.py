#!/usr/bin/env python
"""
52_rarefied_diversity.py

Question
--------
Does node-level diversity, measured at FIXED SEQUENCING DEPTH, predict regime
switches -- and does it match the pooled-cluster entropy result (AUC 0.776
in-sample) that it would replace?

Why this matters
----------------
The entropy result in script 40/41 uses a cluster partition built on all 78
months pooled, so it carries temporal leakage. Every statistic in this script is
computed from a SINGLE month's sequences with no clustering, no partition, and
no reference to any other month. There is nothing to leak.

Script 51 showed corr(n_seqs, vocab_size) = +0.815 raw and -0.496 after
rarefaction: raw vocabulary size is a readout of sequencing effort. So diversity
must be measured at fixed depth or not at all.

Prediction target
-----------------
switch at t+1, where switch months are the six variant transitions recovered by
unsupervised clustering in script 38. Features are evaluated at month t.

Baseline to beat
----------------
- pooled-cluster entropy: AUC 0.776 in-sample (the leaky result being replaced)
- n_seqs at month t: the confound. If sequencing depth alone predicts switches
  at comparable AUC, no diversity claim survives.
- permutation null: switch labels shuffled, 2000 reps. With only 6 positives,
  AUC alone is not evidence; the permutation p-value is the actual test.

Outputs
-------
outputs/52_rarefied_series.csv   per-month diversity statistics
outputs/52_auc.csv               AUC + permutation p per feature
outputs/52_walkforward.csv       prospective scores at each switch

Usage
-----
python scripts/52_rarefied_diversity.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

# regime switch months from script 38 (unsupervised edit-distance clustering)
SWITCH_MONTHS = ["2021-01", "2021-06", "2022-01", "2022-03", "2022-06", "2023-02"]
SWITCH_NAMES = {
    "2021-01": "Alpha",
    "2021-06": "Delta",
    "2022-01": "Omicron BA.1",
    "2022-03": "BA.2",
    "2022-06": "BA.5",
    "2023-02": "XBB",
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


# ----------------------------------------------------------------------------
# per-month statistics at a given depth
# ----------------------------------------------------------------------------

def shannon(counts):
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return np.nan
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def month_stats(occ, min_count, n_mpd_pairs, rng):
    """All statistics computed from one month's occupancy dict alone."""
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total <= 0 or not keys:
        return None

    # node-level frequency distribution (sequence-weighted)
    node_counts = defaultdict(float)
    for k, c in zip(keys, counts):
        for lab in k:
            node_counts[lab] += c
    node_vec = np.array(list(node_counts.values()), dtype=float)

    # mean pairwise distance, sequence-weighted, via pair subsampling
    probs = counts / total
    if len(keys) > 1:
        i = rng.choice(len(keys), size=n_mpd_pairs, p=probs)
        j = rng.choice(len(keys), size=n_mpd_pairs, p=probs)
        d = np.fromiter(
            (len(keys[a] ^ keys[b]) for a, b in zip(i, j)),
            dtype=float, count=n_mpd_pairs,
        )
        mpd = float(d.mean())
    else:
        mpd = 0.0

    sizes = np.array([len(k) for k in keys], dtype=float)

    return {
        "vocab_size": len(node_counts),
        "H_node": shannon(node_vec),
        "H_set": shannon(counts),
        "mean_set_size": float((sizes * probs).sum()),
        "mpd": mpd,
        "n_constellations": len(keys),
    }


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    draws = rng.multinomial(depth, counts / total)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def build_series(months, depth, min_count, n_reps, n_mpd_pairs, seed):
    """Rarefied statistics (mean over reps) plus raw statistics, per month."""
    rng = np.random.default_rng(seed)
    stat_names = ["vocab_size", "H_node", "H_set", "mean_set_size",
                  "mpd", "n_constellations"]
    rows = []

    for month, occ in months:
        raw = month_stats(occ, min_count, n_mpd_pairs, rng)
        row = {"month": month, "n_seqs": int(sum(occ.values()))}
        for s in stat_names:
            row[f"raw_{s}"] = raw[s] if raw else np.nan

        reps = []
        for _ in range(n_reps):
            sub = rarefy(occ, depth, min_count, rng)
            if sub is None:
                continue
            st = month_stats(sub, min_count, n_mpd_pairs, rng)
            if st:
                reps.append(st)
        if reps:
            for s in stat_names:
                vals = [r[s] for r in reps]
                row[f"rare_{s}"] = float(np.mean(vals))
                row[f"rare_{s}_sd"] = float(np.std(vals))
        else:
            for s in stat_names:
                row[f"rare_{s}"] = np.nan
                row[f"rare_{s}_sd"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    # first differences: level may matter less than direction of change
    for pref in ("raw", "rare"):
        for s in stat_names:
            col = f"{pref}_{s}"
            if col in df:
                df[f"d1_{col}"] = df[col].diff()
                df[f"d2_{col}"] = df[col].diff(2)
    return df


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

def auc(y, s):
    """Rank AUC with tie handling. Returns nan if either class is empty."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, n1, n0
    r = pd.Series(s).rank().to_numpy()
    a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(a), n1, n0


def perm_p(y, s, observed, n_perm, rng):
    """Two-sided-ish: fraction of label shuffles reaching |AUC-0.5| >= observed."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or (1 - y).sum() == 0 or np.isnan(observed):
        return np.nan
    target = abs(observed - 0.5)
    hits = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        a, _, _ = auc(yp, s)
        if not np.isnan(a) and abs(a - 0.5) >= target:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def evaluate(df, feature_cols, n_perm, seed):
    """AUC of each feature at month t for a switch at month t+1."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["switch_next"] = d["month"].shift(-1).isin(SWITCH_MONTHS).astype(int)
    d = d.iloc[:-1]  # last month has no t+1

    rows = []
    for col in feature_cols:
        if col not in d:
            continue
        # a rise and a fall are both signals; score both orientations
        a_pos, n1, n0 = auc(d["switch_next"], d[col])
        if np.isnan(a_pos):
            continue
        oriented = a_pos if a_pos >= 0.5 else 1 - a_pos
        sign = "+" if a_pos >= 0.5 else "-"
        p = perm_p(d["switch_next"], d[col], a_pos, n_perm, rng)
        rows.append({
            "feature": col,
            "auc_raw_orientation": a_pos,
            "auc": oriented,
            "direction": sign,
            "perm_p": p,
            "n_switch": n1,
            "n_nonswitch": n0,
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False), d


def walk_forward(d, col, direction):
    """
    Prospective check: at each switch, was the feature extreme relative only to
    months that had already been observed? Uses percentile rank among prior
    months, so nothing after the switch informs the score.
    """
    rows = []
    vals = d[col].to_numpy()
    months = d["month"].to_numpy()
    nxt = d["month"].shift(-1).to_numpy()
    for i in range(6, len(d)):  # need some history
        if not isinstance(nxt[i], str) or nxt[i] not in SWITCH_MONTHS:
            continue
        prior = vals[:i]
        prior = prior[~np.isnan(prior)]
        if len(prior) < 6 or np.isnan(vals[i]):
            continue
        pct = float((prior < vals[i]).mean())
        rows.append({
            "switch_month": nxt[i],
            "variant": SWITCH_NAMES.get(nxt[i], "?"),
            "feature_month": months[i],
            "value": float(vals[i]),
            "pct_rank_vs_prior": pct,
            "extremity": (1 - pct) if direction == "-" else pct,
            "n_prior_months": len(prior),
        })
    return pd.DataFrame(rows)


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
    ap.add_argument("--mpd_pairs", type=int, default=4000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    print(f"loaded {len(months)} months: {months[0][0]} .. {months[-1][0]}")

    df = build_series(months, args.depth, args.min_count,
                      args.reps, args.mpd_pairs, args.seed)
    df.to_csv(f"{args.out_dir}/52_rarefied_series.csv", index=False)

    usable = df["rare_H_node"].notna().sum()
    print(f"months clearing depth {args.depth}: {usable} / {len(df)}")
    present_switches = [m for m in SWITCH_MONTHS if m in set(df["month"])]
    print(f"switch months in range: {present_switches}")

    print("\n=== per-month series (rarefied) ===")
    show = ["month", "n_seqs", "raw_vocab_size", "rare_vocab_size",
            "rare_H_node", "rare_H_set", "rare_mpd", "rare_mean_set_size"]
    print(df[[c for c in show if c in df]].round(3).to_string(index=False))

    feature_cols = []
    for pref in ("rare", "raw"):
        for s in ("vocab_size", "H_node", "H_set", "mpd", "mean_set_size"):
            for lag in ("", "d1_", "d2_"):
                feature_cols.append(f"{lag}{pref}_{s}")
    feature_cols += ["n_seqs"]  # the confound, evaluated as a predictor

    res, d = evaluate(df, feature_cols, args.n_perm, args.seed)
    res.to_csv(f"{args.out_dir}/52_auc.csv", index=False)

    print("\n=== AUC for switch at t+1, feature at t ===")
    print("(auc is orientation-corrected; 'direction' says whether high or low "
          "values flag a switch)")
    print(res.head(20).round(4).to_string(index=False))

    print("\n--- benchmarks ---")
    ns = res[res["feature"] == "n_seqs"]
    if len(ns):
        print(f"n_seqs (confound)          : AUC {ns['auc'].iloc[0]:.3f}  "
              f"p {ns['perm_p'].iloc[0]:.4f}")
    print("pooled-cluster entropy      : AUC 0.776 (leaky, being replaced)")
    print("chance                      : AUC 0.500")
    print(f"\nwith {res['n_switch'].max()} positives, AUC alone is weak evidence.")
    print("perm_p is the test. Treat perm_p > 0.05 as no signal regardless of AUC.")

    # walk-forward on the single best rarefied feature
    rare_res = res[res["feature"].str.contains("rare")]
    if len(rare_res):
        best = rare_res.iloc[0]
        print(f"\n=== walk-forward on best rarefied feature: {best['feature']} "
              f"(direction {best['direction']}) ===")
        wf = walk_forward(d, best["feature"], best["direction"])
        if len(wf):
            wf.to_csv(f"{args.out_dir}/52_walkforward.csv", index=False)
            print(wf.round(3).to_string(index=False))
            print(f"\nmean extremity vs prior months: {wf['extremity'].mean():.3f} "
                  f"(0.5 = no signal, 1.0 = most extreme month so far)")
            print(f"switches in top decile of prior months: "
                  f"{(wf['extremity'] >= 0.9).sum()} / {len(wf)}")
        else:
            print("no switch had enough prior history for a walk-forward score")

    print(f"\nwrote 3 files to {args.out_dir}/")
    print("\nInterpretation guide:")
    print("  perm_p < 0.05 AND AUC near/above 0.776  -> rarefied diversity is a")
    print("     leakage-free replacement for the cluster entropy result")
    print("  perm_p < 0.05 but n_seqs also significant -> depth confound, not a")
    print("     diversity result; rerun with a lower depth and more usable months")
    print("  perm_p > 0.05 -> the dips seen by eye in script 51 are noise, and")
    print("     the entropy finding does not survive removal of the partition")


if __name__ == "__main__":
    main()
