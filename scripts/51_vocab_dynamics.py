#!/usr/bin/env python
"""
51_vocab_dynamics.py

Month-to-month dynamics of the MUTATION VOCABULARY (node level), as opposed to
the constellation level analysed in scripts 22-50.

Definitions
-----------
V_t   : set of (position, residue) labels observed in >= min_count sequences in month t
A_t   : additions,  V_{t+1} \ V_t
R_t   : removals,   V_t \ V_{t+1}
H_t   : set of constellations (frozensets) observed at month t with count >= min_count

Sections
--------
1. Vocabulary size and churn per month, with copy-forward Jaccard baseline.
2. Rarefaction control: recompute (1) after subsampling every month to a fixed
   number of sequences, to separate real turnover from sequencing-depth effects.
3. Node lifetimes: per-label occupancy runs, return-after-removal rates.
4. New-constellation decomposition: for each new constellation at t+1, find its
   source at t (symmetric difference 1) and classify the added label as
   ALREADY IN V_t (recombination) vs NOT IN V_t (vocabulary expansion).

Outputs
-------
outputs/51_vocab.csv          per-month churn statistics
outputs/51_rarefied.csv       same under fixed-depth subsampling
outputs/51_node_runs.csv      per-label occupancy runs and returns
outputs/51_newset_decomp.csv  per-month recombination vs expansion split

Usage
-----
python scripts/51_vocab_dynamics.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_months(data_dir, min_count, start_month=None, end_month=None):
    """Return ordered list of (month, {frozenset: count}) filtered by min_count."""
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


def vocab_of(occ):
    """Union of labels across all constellations in a month."""
    v = set()
    for cs in occ:
        v |= cs
    return v


def n_seqs_of(occ):
    return int(sum(occ.values()))


# ----------------------------------------------------------------------------
# section 1: raw churn
# ----------------------------------------------------------------------------

def churn_table(months):
    rows = []
    seen_ever = set()
    prev_vocab = None
    prev_month = None

    for month, occ in months:
        V = vocab_of(occ)
        row = {
            "month": month,
            "n_seqs": n_seqs_of(occ),
            "n_constellations": len(occ),
            "vocab_size": len(V),
            "n_first_ever": len(V - seen_ever),
        }
        if prev_vocab is not None:
            A = V - prev_vocab
            R = prev_vocab - V
            union = V | prev_vocab
            row.update({
                "prev_month": prev_month,
                "n_added": len(A),
                "n_removed": len(R),
                "n_added_novel": len(A - seen_ever),      # never seen in any prior month
                "n_added_returning": len(A & seen_ever),  # seen before, dropped out, came back
                "jaccard_copyforward": len(V & prev_vocab) / len(union) if union else np.nan,
                "frac_vocab_turnover": (len(A) + len(R)) / len(union) if union else np.nan,
            })
        else:
            row.update({
                "prev_month": None, "n_added": np.nan, "n_removed": np.nan,
                "n_added_novel": np.nan, "n_added_returning": np.nan,
                "jaccard_copyforward": np.nan, "frac_vocab_turnover": np.nan,
            })
        rows.append(row)
        seen_ever |= V
        prev_vocab, prev_month = V, month

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# section 2: rarefaction control
# ----------------------------------------------------------------------------

def rarefy(occ, depth, min_count, rng):
    """Subsample `depth` sequences from the month, return filtered occupancy dict."""
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    probs = counts / total
    draws = rng.multinomial(depth, probs)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def rarefaction_table(months, depth, min_count, n_reps, seed):
    rng = np.random.default_rng(seed)
    reps = []
    for rep in range(n_reps):
        sub = []
        for month, occ in months:
            r = rarefy(occ, depth, min_count, rng)
            if r is None:
                continue
            sub.append((month, r))
        if len(sub) < 2:
            continue
        df = churn_table(sub)
        df["rep"] = rep
        reps.append(df)
    if not reps:
        return pd.DataFrame()
    allr = pd.concat(reps, ignore_index=True)
    agg = allr.groupby("month").agg(
        n_months_used=("rep", "count"),
        vocab_size_mean=("vocab_size", "mean"),
        vocab_size_sd=("vocab_size", "std"),
        n_added_mean=("n_added", "mean"),
        n_removed_mean=("n_removed", "mean"),
        jaccard_mean=("jaccard_copyforward", "mean"),
        jaccard_sd=("jaccard_copyforward", "std"),
    ).reset_index()
    return agg


# ----------------------------------------------------------------------------
# section 3: node lifetimes
# ----------------------------------------------------------------------------

def node_runs(months):
    """For each label: occupancy pattern over months -> runs, gaps, return rate."""
    month_names = [m for m, _ in months]
    idx = {m: i for i, m in enumerate(month_names)}
    T = len(month_names)

    presence = defaultdict(lambda: np.zeros(T, dtype=bool))
    for month, occ in months:
        for lab in vocab_of(occ):
            presence[lab][idx[month]] = True

    rows = []
    for lab, vec in presence.items():
        on = np.flatnonzero(vec)
        first, last = int(on[0]), int(on[-1])
        span = last - first + 1
        n_present = int(vec.sum())
        # runs of consecutive presence
        breaks = np.flatnonzero(np.diff(on) > 1)
        n_runs = len(breaks) + 1
        # gaps: absences strictly inside [first, last] -> these are re-entries
        n_gap_months = span - n_present
        rows.append({
            "label": str(lab),
            "first_month": month_names[first],
            "last_month": month_names[last],
            "span_months": span,
            "months_present": n_present,
            "n_runs": n_runs,
            "n_gap_months": n_gap_months,
            "occupancy_frac": n_present / span,
            "right_censored": last == T - 1,
        })
    return pd.DataFrame(rows).sort_values("first_month").reset_index(drop=True)


def return_rate(months, horizons=(1, 2, 3, 6)):
    """Of labels removed at t, what fraction reappear within h months?"""
    month_names = [m for m, _ in months]
    vocabs = [vocab_of(occ) for _, occ in months]
    T = len(vocabs)
    rows = []
    for t in range(T - 1):
        removed = vocabs[t] - vocabs[t + 1]
        if not removed:
            continue
        row = {"month": month_names[t], "n_removed": len(removed)}
        for h in horizons:
            end = min(t + 1 + h, T)
            if end <= t + 1:
                row[f"ret_{h}"] = np.nan
                continue
            future = set().union(*vocabs[t + 1:end]) if end > t + 1 else set()
            # right-censoring: only report if the full horizon is observed
            row[f"ret_{h}"] = (len(removed & future) / len(removed)
                               if t + 1 + h <= T else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# section 4: new-constellation decomposition
# ----------------------------------------------------------------------------

def newset_decomposition(months):
    """
    For each new constellation c at t+1, look for a source c' at t with
    symmetric difference exactly 1 (c = c' u {m}, or c = c' \ {m}).
    Classify the differing label m by whether it was in V_t.
    """
    rows = []
    for i in range(len(months) - 1):
        m_t, occ_t = months[i]
        m_n, occ_n = months[i + 1]
        H_t = set(occ_t.keys())
        V_t = vocab_of(occ_t)

        new_sets = [c for c in occ_n if c not in H_t]
        if not new_sets:
            continue

        n_add_src = n_del_src = n_no_src = 0
        n_recomb = n_expand = 0
        n_multi_src = 0
        add_seqs = recomb_seqs = expand_seqs = 0

        for c in new_sets:
            w = occ_n[c]
            sources_added = []   # c' subset of c, |c \ c'| == 1
            for lab in c:
                cand = frozenset(c - {lab})
                if cand in H_t:
                    sources_added.append(lab)
            if sources_added:
                n_add_src += 1
                add_seqs += w
                if len(sources_added) > 1:
                    n_multi_src += 1
                # a new constellation counts as "expansion" only if EVERY route
                # into it requires a label absent from V_t
                if any(lab in V_t for lab in sources_added):
                    n_recomb += 1
                    recomb_seqs += w
                else:
                    n_expand += 1
                    expand_seqs += w
                continue
            # deletion route: c' = c u {lab} present at t
            found_del = False
            for lab in V_t:
                if lab in c:
                    continue
                if frozenset(c | {lab}) in H_t:
                    found_del = True
                    break
            if found_del:
                n_del_src += 1
            else:
                n_no_src += 1

        n_new = len(new_sets)
        tot_seqs_new = sum(occ_n[c] for c in new_sets)
        rows.append({
            "month_t": m_t,
            "month_t1": m_n,
            "n_new_sets": n_new,
            "frac_new_of_H": n_new / len(occ_n),
            "n_src_addition": n_add_src,
            "n_src_deletion": n_del_src,
            "n_src_none": n_no_src,
            "frac_min_edit_1": (n_add_src + n_del_src) / n_new,
            "n_multi_source": n_multi_src,
            # the key split, among addition-route new sets
            "n_recombination": n_recomb,
            "n_expansion": n_expand,
            "frac_recombination": n_recomb / n_add_src if n_add_src else np.nan,
            # sequence-weighted version (does expansion carry more sequences?)
            "seqfrac_recombination": recomb_seqs / add_seqs if add_seqs else np.nan,
            "seqs_in_new_sets": tot_seqs_new,
            "seqfrac_new": tot_seqs_new / n_seqs_of(occ_n),
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
    ap.add_argument("--rarefy_depth", type=int, default=5000)
    ap.add_argument("--rarefy_reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    print(f"loaded {len(months)} months: {months[0][0]} .. {months[-1][0]}")

    # 1. churn
    churn = churn_table(months)
    churn.to_csv(f"{args.out_dir}/51_vocab.csv", index=False)
    print("\n=== 1. vocabulary churn (raw) ===")
    print(churn[["month", "n_seqs", "vocab_size", "n_added", "n_removed",
                 "n_added_novel", "jaccard_copyforward"]].to_string(index=False))
    j = churn["jaccard_copyforward"].dropna()
    print(f"\ncopy-forward Jaccard: mean {j.mean():.3f}  median {j.median():.3f} "
          f"min {j.min():.3f} max {j.max():.3f}")
    print("  ^ this is the baseline any vocabulary model must beat")

    # 2. rarefaction
    print(f"\n=== 2. rarefaction control (depth={args.rarefy_depth}, "
          f"reps={args.rarefy_reps}) ===")
    rar = rarefaction_table(months, args.rarefy_depth, args.min_count,
                            args.rarefy_reps, args.seed)
    if len(rar):
        rar.to_csv(f"{args.out_dir}/51_rarefied.csv", index=False)
        print(rar.to_string(index=False))
        merged = churn.merge(rar, on="month", how="inner")
        if len(merged) > 2:
            r_raw = np.corrcoef(merged["n_seqs"], merged["vocab_size"])[0, 1]
            r_rar = np.corrcoef(merged["n_seqs"], merged["vocab_size_mean"])[0, 1]
            print(f"\ncorr(n_seqs, vocab_size)        raw      = {r_raw:+.3f}")
            print(f"corr(n_seqs, vocab_size) rarefied         = {r_rar:+.3f}")
            print("  large raw correlation collapsing after rarefaction means")
            print("  vocabulary growth is a sequencing-depth artefact, not biology")
    else:
        print("no months met the rarefaction depth; lower --rarefy_depth")

    # 3. node lifetimes
    print("\n=== 3. node occupancy ===")
    runs = node_runs(months)
    runs.to_csv(f"{args.out_dir}/51_node_runs.csv", index=False)
    print(f"labels ever observed: {len(runs)}")
    print(f"median span (months): {runs['span_months'].median():.1f}")
    print(f"median occupancy frac within span: {runs['occupancy_frac'].median():.3f}")
    print(f"labels with >1 run (dropped out and returned): "
          f"{(runs['n_runs'] > 1).sum()} / {len(runs)} "
          f"({(runs['n_runs'] > 1).mean():.1%})")

    ret = return_rate(months)
    if len(ret):
        print("\nreturn rate after removal (mean over months):")
        for h in (1, 2, 3, 6):
            col = f"ret_{h}"
            if col in ret:
                print(f"  within {h} month(s): {ret[col].mean():.3f}")
        print("  high return rates => 'removal' is sampling dropout, not loss")

    # 4. new-set decomposition
    print("\n=== 4. new constellations: recombination vs vocabulary expansion ===")
    dec = newset_decomposition(months)
    dec.to_csv(f"{args.out_dir}/51_newset_decomp.csv", index=False)
    print(dec[["month_t1", "n_new_sets", "frac_new_of_H", "frac_min_edit_1",
               "frac_recombination", "seqfrac_recombination"]].to_string(index=False))
    print(f"\nmean frac_new_of_H        : {dec['frac_new_of_H'].mean():.3f} "
          f"(compare to script 22 value ~0.38)")
    print(f"mean frac_min_edit_1      : {dec['frac_min_edit_1'].mean():.3f} "
          f"(script: claimed 1.000 universally)")
    print(f"mean frac_recombination   : {dec['frac_recombination'].mean():.3f}")
    print(f"mean seqfrac_recombination: {dec['seqfrac_recombination'].mean():.3f}")
    print("\n  frac_recombination near 1.0 => new constellations are new")
    print("  COMBINATIONS of already-circulating labels; the forecasting problem")
    print("  is co-occurrence, not mutation generation.")
    print("  frac_recombination well below 1.0 => vocabulary expansion is a real")
    print("  channel and a proposal model over labels is warranted.")

    print(f"\nwrote 4 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
