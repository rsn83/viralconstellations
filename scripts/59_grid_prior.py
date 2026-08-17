#!/usr/bin/env python
"""
59_grid_prior.py

Question
--------
The candidate space is (position, residue) cells. How much of it is ever used,
and does a hierarchy over positions narrow what is possible at time t?

Three candidate priors, and they narrow the space differently:

  1. STATIC OCCUPANCY  -- which cells are ever occupied at all. Time-invariant,
     prunes the grid globally, costs nothing.
  2. POSITION HIERARCHY -- positions differ in how often they mutate and in how
     many residues they admit. Learnable, still time-invariant.
  3. LINEAGE-CONDITIONAL -- which mutations are possible given the background
     already present. Only this one is conditioned on t. Not measured here;
     script 58's pmi_min is the evidence it exists.

Caution carried in from earlier work: the frontier result (positional score vs
offspring count, r = -0.406) says positional structure alone does NOT predict
where evolution goes. So the expectation is that 1 and 2 shrink the denominator
without ranking what remains. This script measures how much shrinking there is,
which sets how much room the conditional term has to work in.

Everything time-dependent here is causal: a prior at month t is built from
months <= t and scored on t+1. Nothing is pooled across the full series except
the descriptive totals in section A, which are labelled as such.

Sections
--------
A. GRID OCCUPANCY (descriptive, pooled -- not a prediction)
   How many cells of the full grid are ever occupied. Positions that ever
   mutate. Residues admitted per mutating position.

B. CAUSAL COVERAGE OF THE STATIC PRIOR
   For each t: of the cells active at t+1, what share was already seen by t?
   That is the prior's recall. And of the cells seen by t, what share is active
   at t+1? That is its precision. Recall bounds any model restricted to
   previously-seen cells; precision says how much ranking work is left.

C. POSITION HIERARCHY
   Does a position's own history predict whether it carries a NEW residue at
   t+1? Compared against a uniform-over-positions baseline. Reported as lift,
   since the positive rate is small.

D. RESIDUE GIVEN POSITION
   Given that a position produces a new residue at t+1, is which residue
   predictable from the residues that position has admitted before? This is the
   second level of the hierarchy and is scored separately, because a prior that
   only ranks positions leaves 19 choices untouched.

Outputs
-------
outputs/59_grid_summary.csv     section A totals
outputs/59_coverage.csv         per-month recall/precision of the static prior
outputs/59_position_hier.csv    per-month position-level lift
outputs/59_residue_hier.csv     per-month residue-level lift

Usage
-----
python scripts/59_grid_prior.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

AAS = list("ACDEFGHIKLMNPQRSTVWY")   # 20 standard residues
SPIKE_LEN = 1273                      # reference length, for context only


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


def split_label(lab):
    """
    Labels are (position, residue) pairs. Tolerate tuple/list or a string form
    like '501Y' / '501_Y' / 'N501Y' so this runs without reformatting the data.
    """
    if isinstance(lab, (tuple, list)) and len(lab) == 2:
        return int(lab[0]), str(lab[1])
    s = str(lab)
    m = re.match(r"^[A-Za-z]?(\d+)[_\-]?([A-Za-z\*\-]+)$", s)
    if m:
        return int(m.group(1)), m.group(2)
    raise ValueError(f"cannot parse label: {lab!r}")


def month_cells(occ):
    """Set of (pos, res) cells present in a month."""
    out = set()
    for cs in occ:
        for lab in cs:
            out.add(split_label(lab))
    return out


def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, y.size + 1)
    return float((prec * yy).sum() / y.sum())


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
    ap.add_argument("--min_train", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    cells = {m: month_cells(o) for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    # ========================================================================
    # A. grid occupancy  (descriptive, pooled across all months)
    # ========================================================================
    print("\n" + "=" * 72)
    print("A. GRID OCCUPANCY  (descriptive; uses all months, not a forecast)")
    print("=" * 72)

    all_cells = set()
    for m in names:
        all_cells |= cells[m]
    positions = sorted({p for p, _ in all_cells})
    residues = sorted({r for _, r in all_cells})
    max_pos = max(positions)
    n_pos_grid = max(max_pos, SPIKE_LEN)
    grid = n_pos_grid * len(AAS)

    res_per_pos = defaultdict(set)
    for p, r in all_cells:
        res_per_pos[p].add(r)
    rpp = np.array([len(v) for v in res_per_pos.values()])

    print(f"positions observed          : {len(positions)} "
          f"(max index {max_pos}; reference length {SPIKE_LEN})")
    print(f"distinct residue symbols     : {len(residues)}  {residues}")
    print(f"full grid (positions x 20)   : {grid}")
    print(f"cells ever occupied          : {len(all_cells)}")
    print(f"reduction from the full grid : "
          f"{100 * (1 - len(all_cells) / grid):.2f}%  "
          f"({grid / max(len(all_cells), 1):.1f}x smaller)")
    print(f"positions that ever mutate   : {len(positions)} / {n_pos_grid} "
          f"({100 * len(positions) / n_pos_grid:.1f}%)")
    print(f"residues per mutating position: mean {rpp.mean():.2f}, "
          f"median {np.median(rpp):.0f}, max {rpp.max()}")
    print("\ndecomposition of the reduction:")
    print(f"  by position filter alone   : {len(positions) * len(AAS)} cells")
    print(f"  plus residue filter        : {len(all_cells)} cells")
    print(f"  the position filter does {100 * (1 - len(positions) * len(AAS) / grid):.1f}%"
          " of the work,")
    print(f"  the residue filter a further "
          f"{100 * (1 - len(all_cells) / max(len(positions) * len(AAS), 1)):.1f}%")

    pd.DataFrame([{
        "n_positions_observed": len(positions), "max_position": max_pos,
        "grid_size": grid, "cells_ever_occupied": len(all_cells),
        "positions_x_20": len(positions) * len(AAS),
        "mean_residues_per_position": float(rpp.mean()),
        "median_residues_per_position": float(np.median(rpp)),
    }]).to_csv(f"{args.out_dir}/59_grid_summary.csv", index=False)

    # ========================================================================
    # B. causal coverage of the static prior
    # ========================================================================
    print("\n" + "=" * 72)
    print("B. CAUSAL COVERAGE  (prior = cells seen by t, scored on t+1)")
    print("=" * 72)

    rows = []
    seen = set()
    for t in range(T - 1):
        seen |= cells[names[t]]
        nxt = cells[names[t + 1]]
        if not nxt:
            continue
        hit = len(nxt & seen)
        rows.append({
            "month_t": names[t], "month_t1": names[t + 1],
            "n_seen_by_t": len(seen),
            "n_active_t1": len(nxt),
            "recall": hit / len(nxt),
            "precision": hit / len(seen) if seen else np.nan,
            "n_novel_cells": len(nxt - seen),
            "search_space_reduction": 1 - len(seen) / grid,
        })
    cov = pd.DataFrame(rows)
    cov.to_csv(f"{args.out_dir}/59_coverage.csv", index=False)
    print(cov.round(4).tail(24).to_string(index=False))
    print(f"\nmean recall    {cov['recall'].mean():.4f}  "
          f"(share of next month's cells already seen)")
    print(f"mean precision {cov['precision'].mean():.4f}  "
          f"(share of seen cells that are active next month)")
    print(f"mean novel cells per month: {cov['n_novel_cells'].mean():.1f}")
    print(f"prior keeps {cov['n_seen_by_t'].mean():.0f} of {grid} cells on "
          f"average ({100 * cov['search_space_reduction'].mean():.2f}% removed)")
    print("\nread: high recall with low precision is the expected shape. It means")
    print("      the static prior prunes hard but does not rank -- exactly what")
    print("      the frontier null result (r = -0.406) predicted.")

    # ========================================================================
    # C. position hierarchy
    # ========================================================================
    print("\n" + "=" * 72)
    print("C. POSITION HIERARCHY  (does a position's history predict new")
    print("   residues appearing there at t+1?)")
    print("=" * 72)

    pos_hist_months = defaultdict(int)     # months in which the position mutated
    pos_hist_res = defaultdict(set)        # residues the position has admitted
    pos_last_seen = {}
    prows = []

    for t in range(T - 1):
        for p, r in cells[names[t]]:
            pos_hist_months[p] += 1
            pos_hist_res[p].add(r)
            pos_last_seen[p] = t

        if t < args.min_train:
            continue

        seen_by_t = set()
        for j in range(t + 1):
            seen_by_t |= cells[names[j]]
        seen_pos = sorted({p for p, _ in seen_by_t})
        if len(seen_pos) < 10:
            continue

        # target: does this position carry a residue at t+1 that it has NOT
        # carried before? that is the position producing something new
        nxt = cells[names[t + 1]]
        new_by_pos = defaultdict(set)
        for p, r in nxt:
            if r not in pos_hist_res.get(p, set()):
                new_by_pos[p].add(r)
        y = np.array([1 if p in new_by_pos else 0 for p in seen_pos])
        if y.sum() == 0:
            continue

        n_months = np.array([pos_hist_months[p] for p in seen_pos], dtype=float)
        n_res = np.array([len(pos_hist_res[p]) for p in seen_pos], dtype=float)
        recency = np.array([1.0 / (1.0 + (t - pos_last_seen.get(p, -99)))
                            for p in seen_pos])

        base = float(y.mean())
        for sname, s in [("uniform", rng.random(len(y))),
                         ("n_months_active", n_months),
                         ("n_residues_seen", n_res),
                         ("recency", recency)]:
            apv = average_precision(y, s)
            prows.append({
                "month_t": names[t], "score": sname,
                "ap": apv, "base_rate": base,
                "lift": apv / base if base > 0 else np.nan,
                "n_positions": len(seen_pos), "n_pos_new": int(y.sum()),
            })

    ph = pd.DataFrame(prows)
    ph.to_csv(f"{args.out_dir}/59_position_hier.csv", index=False)
    if len(ph):
        g = ph.groupby("score").agg(
            ap=("ap", "mean"), lift=("lift", "mean"),
            base=("base_rate", "mean"), n_positions=("n_positions", "mean"),
            n_new=("n_pos_new", "mean"), origins=("ap", "count"),
        ).reset_index().sort_values("ap", ascending=False)
        print(g.round(4).to_string(index=False))
        print("\nread: lift near 1 for every score means position history does not")
        print("      say where novelty appears, and the hierarchy prunes only.")
        print("      lift well above 1 means positions carry a usable rate.")

    # ========================================================================
    # D. residue given position
    # ========================================================================
    print("\n" + "=" * 72)
    print("D. RESIDUE GIVEN POSITION  (given a position produces something new,")
    print("   is WHICH residue predictable?)")
    print("=" * 72)

    res_global = defaultdict(int)          # how often each residue is used anywhere
    pos_res_hist = defaultdict(set)
    rrows = []

    for t in range(T - 1):
        for p, r in cells[names[t]]:
            pos_res_hist[p].add(r)
            res_global[r] += 1

        if t < args.min_train:
            continue

        nxt = cells[names[t + 1]]
        ys, s_uniform, s_global = [], [], []
        gtot = sum(res_global.values()) or 1
        n_events = 0
        for p in sorted({p for p, _ in nxt}):
            known = pos_res_hist.get(p, set())
            new_res = {r for q, r in nxt if q == p} - known
            if not new_res:
                continue
            n_events += 1
            cand = [a for a in AAS if a not in known]
            if len(cand) < 2:
                continue
            for a in cand:
                ys.append(1 if a in new_res else 0)
                s_uniform.append(rng.random())
                s_global.append(res_global.get(a, 0) / gtot)
        if not ys or sum(ys) == 0:
            continue
        y = np.array(ys)
        base = float(y.mean())
        for sname, s in [("uniform", np.array(s_uniform)),
                         ("global_residue_freq", np.array(s_global))]:
            apv = average_precision(y, s)
            rrows.append({
                "month_t": names[t], "score": sname, "ap": apv,
                "base_rate": base, "lift": apv / base if base > 0 else np.nan,
                "n_candidates": len(y), "n_events": n_events,
            })

    rh = pd.DataFrame(rrows)
    rh.to_csv(f"{args.out_dir}/59_residue_hier.csv", index=False)
    if len(rh):
        g2 = rh.groupby("score").agg(
            ap=("ap", "mean"), lift=("lift", "mean"),
            base=("base_rate", "mean"), n_candidates=("n_candidates", "mean"),
            origins=("ap", "count"),
        ).reset_index().sort_values("ap", ascending=False)
        print(g2.round(4).to_string(index=False))
        print("\nread: if global residue frequency has lift ~1, the second level of")
        print("      the hierarchy carries nothing and knowing the position still")
        print("      leaves ~19 equally likely choices. That would put the whole")
        print("      burden of ranking on the background-conditional term.")

    print(f"\nwrote 4 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
