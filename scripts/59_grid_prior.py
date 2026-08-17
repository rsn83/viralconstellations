#!/usr/bin/env python
"""
59_grid_prior.py  (v2 -- resolves integer node IDs through index.tsv)

Question
--------
The candidate space is (position, residue) cells. How much of it is ever used,
and does a hierarchy over positions narrow what is possible at time t?

Three candidate priors, narrowing the space differently:

  1. STATIC OCCUPANCY   -- which cells are ever occupied at all. Time-invariant,
     prunes globally, costs nothing.
  2. POSITION HIERARCHY -- positions differ in how often they mutate and how
     many residues they admit. Learnable, still time-invariant.
  3. LINEAGE-CONDITIONAL -- which mutations are possible given the background
     already present. Only this one is conditioned on t. Not measured here;
     script 58's pmi_min is the evidence it exists.

Caution carried in: the frontier result (positional score vs offspring count,
r = -0.406) says positional structure alone does NOT predict where evolution
goes. So the expectation is that 1 and 2 shrink the denominator without ranking
what remains. This measures how much shrinking there is, which sets how much
room the conditional term has to work in.

Everything time-dependent is causal: a prior at month t is built from months
<= t and scored on t+1. Section A is pooled and labelled descriptive.

Sections
--------
A. GRID OCCUPANCY (descriptive, pooled -- not a prediction)
B. CAUSAL COVERAGE OF THE STATIC PRIOR  (recall and precision on t+1)
C. POSITION HIERARCHY   (does a position's history predict new residues there?)
D. RESIDUE GIVEN POSITION (given a position produces novelty, which residue?)

Outputs
-------
outputs/59_grid_summary.csv
outputs/59_coverage.csv
outputs/59_position_hier.csv
outputs/59_residue_hier.csv

Usage
-----
python scripts/59_grid_prior.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

AAS = list("ACDEFGHIKLMNPQRSTVWY")
SPIKE_LEN = 1273

# 'N501Y', '501Y', 'S:N501Y', '501_Y', 'del69', ...
MUT_RE = re.compile(r"^(?:[A-Za-z]+:)?[A-Za-z\*\-]?(\d+)[_\-]?([A-Za-z\*\-]+)$")


# ----------------------------------------------------------------------------
# node id -> (position, residue), via index.tsv
# ----------------------------------------------------------------------------

def _parse_mut_string(s):
    m = MUT_RE.match(str(s).strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def build_id_map(index_path, verbose=True):
    """
    Reads index.tsv and returns {node_id: (pos, res)}.

    Handles the layouts these index files usually come in:
      - explicit position and residue columns
      - a single mutation-label column such as N501Y / 501Y / S:N501Y
      - no header, in which case columns are positional
    The id is taken from an id-like column when present, otherwise from the
    row order, which is how these indices are normally written.
    """
    if not os.path.exists(index_path):
        sys.exit(f"index file not found: {index_path}\n"
                 "pass its location with --index_path")

    df = pd.read_csv(index_path, sep="\t", dtype=str, keep_default_na=False)
    # a headerless file shows up as a first row that is itself data
    looks_headerless = all(
        _parse_mut_string(c) is not None or str(c).isdigit() for c in df.columns
    )
    if looks_headerless:
        df = pd.read_csv(index_path, sep="\t", dtype=str, header=None,
                         keep_default_na=False)
        df.columns = [f"c{i}" for i in range(df.shape[1])]

    cols = {c.lower().strip(): c for c in df.columns}
    if verbose:
        print(f"index.tsv columns: {list(df.columns)}  ({len(df)} rows)")
        print(df.head(3).to_string(index=False))

    # id column
    id_col = None
    for cand in ("node_idx", "node", "node_id", "id", "index", "idx", "i"):
        if cand in cols:
            id_col = cols[cand]
            break

    # explicit position / residue columns
    pos_col = next((cols[c] for c in ("pos", "position", "site", "aa_pos",
                                      "aapos", "amino_acid_position")
                    if c in cols), None)
    res_col = next((cols[c] for c in ("res", "residue", "aa", "alt", "mut_aa",
                                      "aa_res", "alt_aa")
                    if c in cols), None)

    id_map = {}
    if pos_col is not None and res_col is not None:
        for i, row in enumerate(df.itertuples(index=False)):
            d = dict(zip(df.columns, row))
            key = int(d[id_col]) if id_col else i
            id_map[key] = (int(str(d[pos_col]).strip()), str(d[res_col]).strip())
        how = f"columns '{pos_col}' + '{res_col}'"
    else:
        # find the column whose values parse as mutation labels
        best_col, best_hits = None, 0
        for c in df.columns:
            if c == id_col:
                continue
            sample = df[c].head(200)
            hits = sum(_parse_mut_string(v) is not None for v in sample)
            if hits > best_hits:
                best_col, best_hits = c, hits
        if best_col is None or best_hits < max(5, 0.5 * min(200, len(df))):
            sys.exit("could not identify a (position, residue) column in "
                     "index.tsv; print a few rows and I will adjust the parser")
        for i, row in enumerate(df.itertuples(index=False)):
            d = dict(zip(df.columns, row))
            key = int(d[id_col]) if id_col else i
            pr = _parse_mut_string(d[best_col])
            if pr:
                id_map[key] = pr
        how = f"column '{best_col}'"

    if verbose:
        print(f"resolved {len(id_map)} node ids from {how}"
              f"{' with id column ' + id_col if id_col else ' by row order'}")
        ex = list(id_map.items())[:5]
        print("examples: " + ", ".join(f"{k} -> {v}" for k, v in ex))
    return id_map


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


def month_cells(occ, id_map, missing):
    out = set()
    for cs in occ:
        for lab in cs:
            cell = id_map.get(int(lab)) if not isinstance(lab, tuple) else lab
            if cell is None:
                missing.add(lab)
                continue
            out.add(cell)
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
    ap.add_argument("--index_path", default=None)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--min_train", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    index_path = args.index_path or os.path.join(args.data_dir, "index.tsv")
    id_map = build_id_map(index_path)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    missing = set()
    cells = {m: month_cells(o, id_map, missing) for m, o in months}
    names = [m for m, _ in months]
    T = len(names)
    print(f"\nloaded {T} months: {names[0]} .. {names[-1]}")
    if missing:
        print(f"WARNING: {len(missing)} node ids absent from index.tsv, dropped "
              f"(e.g. {sorted(missing, key=str)[:5]})")

    # ========================================================================
    # A. grid occupancy (descriptive, pooled)
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

    print(f"positions observed           : {len(positions)} "
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
    print(f"  position filter alone      : {len(positions) * len(AAS)} cells "
          f"({100 * (1 - len(positions) * len(AAS) / grid):.1f}% removed)")
    print(f"  plus residue filter        : {len(all_cells)} cells "
          f"(a further "
          f"{100 * (1 - len(all_cells) / max(len(positions) * len(AAS), 1)):.1f}%)")

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
            "n_seen_by_t": len(seen), "n_active_t1": len(nxt),
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
          f"(share of seen cells active next month)")
    print(f"mean novel cells per month: {cov['n_novel_cells'].mean():.1f}")
    print(f"prior keeps {cov['n_seen_by_t'].mean():.0f} of {grid} cells "
          f"({100 * cov['search_space_reduction'].mean():.2f}% removed)")
    print("\nread: high recall with low precision is the expected shape -- the")
    print("      static prior prunes hard but does not rank, which is what the")
    print("      frontier null result (r = -0.406) predicted.")

    # ========================================================================
    # C. position hierarchy
    # ========================================================================
    print("\n" + "=" * 72)
    print("C. POSITION HIERARCHY  (does a position's history predict new")
    print("   residues appearing there at t+1?)")
    print("=" * 72)

    pos_hist_months = defaultdict(int)
    pos_hist_res = defaultdict(set)
    pos_last_seen = {}
    prows = []
    seen_by_t = set()

    for t in range(T - 1):
        for p, r in cells[names[t]]:
            pos_hist_months[p] += 1
            pos_hist_res[p].add(r)
            pos_last_seen[p] = t
        seen_by_t |= cells[names[t]]

        if t < args.min_train:
            continue
        seen_pos = sorted({p for p, _ in seen_by_t})
        if len(seen_pos) < 10:
            continue

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
                "month_t": names[t], "score": sname, "ap": apv,
                "base_rate": base, "lift": apv / base if base > 0 else np.nan,
                "n_positions": len(seen_pos), "n_pos_new": int(y.sum()),
            })

    ph = pd.DataFrame(prows)
    ph.to_csv(f"{args.out_dir}/59_position_hier.csv", index=False)
    if len(ph):
        g = ph.groupby("score").agg(
            ap=("ap", "mean"), lift=("lift", "mean"), base=("base_rate", "mean"),
            n_positions=("n_positions", "mean"), n_new=("n_pos_new", "mean"),
            origins=("ap", "count"),
        ).reset_index().sort_values("ap", ascending=False)
        print(g.round(4).to_string(index=False))
        print("\nread: lift near 1 for every score means position history does not")
        print("      say where novelty appears -- the hierarchy prunes only.")

    # ========================================================================
    # D. residue given position
    # ========================================================================
    print("\n" + "=" * 72)
    print("D. RESIDUE GIVEN POSITION  (given a position produces novelty,")
    print("   is WHICH residue predictable?)")
    print("=" * 72)

    res_global = defaultdict(int)
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
        by_pos = defaultdict(set)
        for p, r in nxt:
            by_pos[p].add(r)
        for p in sorted(by_pos):
            known = pos_res_hist.get(p, set())
            new_res = by_pos[p] - known
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
            ap=("ap", "mean"), lift=("lift", "mean"), base=("base_rate", "mean"),
            n_candidates=("n_candidates", "mean"), origins=("ap", "count"),
        ).reset_index().sort_values("ap", ascending=False)
        print(g2.round(4).to_string(index=False))
        print("\nread: lift ~1 for global residue frequency means the second level")
        print("      of the hierarchy carries nothing -- knowing the position still")
        print("      leaves ~19 near-equal choices, putting the whole ranking")
        print("      burden on the background-conditional term.")

    print(f"\nwrote 4 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
