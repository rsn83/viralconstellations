#!/usr/bin/env python
"""
22_set_dynamics_stats.py

Pure bookkeeping. No model, no GPU, no training. Counts only.

Question
--------
When a constellation appears at month t+h that was not present at month t,
how far is it from the sets that WERE present?

This tests -- rather than assumes -- the claim that "a new constellation is an
existing one plus a mutation". If most new sets sit at addition-distance 1,
a one-mutation frontier is a defensible candidate generator. If they don't,
it isn't, and that is itself worth reporting.

Definitions (all exact set operations -- NO ancestry is inferred)
----------------------------------------------------------------
  H_t      = {frozenset(mutations): count} at month t
  persist  = c in H_t and c in H_{t+h}
  extinct  = c in H_t and c not in H_{t+h}
  new      = c not in H_t and c in H_{t+h}

For each NEW set c, over parents c' in H_t:
  add_dist(c) = min |c \\ c'| over c' in H_t with c' SUBSET of c
              = the fewest mutations you must ADD to some circulating set
                to reach c.
  If no c' in H_t is a subset of c, add_dist is undefined ("no_subset"):
  reaching c requires losing a mutation, recombination, or an unsampled
  intermediate. Those cases can never be produced by additive expansion.

IMPORTANT: add_dist=1 means c is REACHABLE from a circulating set by one
addition. It does NOT mean c descended from that set. Reachability is
geometric; ancestry is latent and not measured here.

Usage
-----
  python scripts/22_set_dynamics_stats.py
  python scripts/22_set_dynamics_stats.py --horizons 1 3 6 --min_count 1
  python scripts/22_set_dynamics_stats.py --weight_by_count
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def constellations_of(occ: dict) -> dict:
    """Same normalisation as script 19."""
    out = {}
    for c, v in occ.items():
        count = v if isinstance(v, (int, float)) else 1
        out[frozenset(c)] = count
    return out


def load_month_sets(graphs_dir: Path, month: str) -> dict:
    with open(graphs_dir / f"{month}_occupied.pkl", "rb") as fh:
        occ = pickle.load(fh)
    return constellations_of(occ)


# ---------------------------------------------------------------------------
# addition distance
# ---------------------------------------------------------------------------

def addition_distance(c: frozenset, by_size: dict, max_report: int = 4):
    """Smallest k such that some circulating set c' with c' SUBSET c has
    |c| - |c'| == k. Returns (k, n_parents_at_k) or (None, 0) if no
    circulating set is a subset of c.

    Searched smallest-k-first, so it stops at the closest parent.
    """
    n = len(c)
    for k in range(1, max_report + 1):
        psize = n - k
        if psize < 1:
            break
        bucket = by_size.get(psize)
        if not bucket:
            continue
        hits = 0
        for c2 in bucket:
            if c2 <= c:          # frozenset subset test
                hits += 1
        if hits:
            return k, hits
    # anything beyond max_report, or genuinely unreachable by addition
    for psize in range(1, n - max_report):
        bucket = by_size.get(psize)
        if not bucket:
            continue
        for c2 in bucket:
            if c2 <= c:
                return -1, 1     # -1 == "far" (reachable but > max_report)
    return None, 0               # no subset at all


def index_by_size(H: dict) -> dict:
    by_size = {}
    for c in H.keys():
        by_size.setdefault(len(c), []).append(c)
    return by_size


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--min_count", type=int, default=1,
                    help="ignore constellations seen fewer than this many times "
                         "in a month (singletons are often sequencing noise)")
    ap.add_argument("--max_set_size", type=int, default=64)
    ap.add_argument("--max_add_dist", type=int, default=4)
    ap.add_argument("--weight_by_count", action="store_true",
                    help="weight each new set by its abundance at t+h, so the "
                         "table reflects sequences rather than distinct sets")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "22_set_dynamics.csv"))
    args = ap.parse_args()

    graphs_dir = Path(args.graphs_dir)
    index_df = pd.read_csv(graphs_dir / "index.tsv", sep="\t")
    months = sorted(index_df["month"].tolist())
    log(f"{len(months)} months: {months[0]} .. {months[-1]}")
    log(f"min_count={args.min_count}  max_set_size={args.max_set_size}  "
        f"weight_by_count={args.weight_by_count}\n")

    cache = {}

    def H(month):
        if month not in cache:
            raw = load_month_sets(graphs_dir, month)
            cache[month] = {c: v for c, v in raw.items()
                            if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
        return cache[month]

    rows = []
    for h in args.horizons:
        log("=" * 78)
        log(f"HORIZON h={h}")
        log("=" * 78)
        for i in range(len(months) - h):
            m_t, m_th = months[i], months[i + h]
            Ht, Hth = H(m_t), H(m_th)
            if not Ht or not Hth:
                continue

            keys_t, keys_th = set(Ht), set(Hth)
            persist = keys_t & keys_th
            extinct = keys_t - keys_th
            new = keys_th - keys_t

            by_size = index_by_size(Ht)
            dist_counter = Counter()
            weight_counter = Counter()
            parent_counts = []

            for c in new:
                k, npar = addition_distance(c, by_size, args.max_add_dist)
                key = ("no_subset" if k is None else ("far" if k == -1 else k))
                dist_counter[key] += 1
                weight_counter[key] += Hth[c]
                if k == 1:
                    parent_counts.append(npar)

            src = weight_counter if args.weight_by_count else dist_counter
            tot = sum(src.values())

            def frac(key):
                return (src.get(key, 0) / tot) if tot else float("nan")

            row = {
                "horizon": h,
                "month_t": m_t,
                "month_th": m_th,
                "n_H_t": len(keys_t),
                "n_H_th": len(keys_th),
                "n_persist": len(persist),
                "n_extinct": len(extinct),
                "n_new": len(new),
                "frac_persist_of_Ht": len(persist) / len(keys_t) if keys_t else np.nan,
                "frac_new_of_Hth": len(new) / len(keys_th) if keys_th else np.nan,
                "new_d1": frac(1),
                "new_d2": frac(2),
                "new_d3": frac(3),
                "new_d4": frac(4),
                "new_far": frac("far"),
                "new_no_subset": frac("no_subset"),
                "n_new_d1_abs": dist_counter.get(1, 0),
                "median_parents_at_d1": float(np.median(parent_counts)) if parent_counts else np.nan,
            }
            rows.append(row)

            log(f"  {m_t} -> {m_th} | H_t={row['n_H_t']:6d} H_th={row['n_H_th']:6d} "
                f"| persist={row['n_persist']:6d} extinct={row['n_extinct']:6d} "
                f"new={row['n_new']:6d} "
                f"|| d1={row['new_d1']:.3f} d2={row['new_d2']:.3f} d3={row['new_d3']:.3f} "
                f"d4={row['new_d4']:.3f} far={row['new_far']:.3f} "
                f"no_subset={row['new_no_subset']:.3f}")

    if not rows:
        log("No usable month pairs.")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---------------- summary ----------------
    log("\n" + "=" * 78)
    log("SUMMARY (mean over month pairs)")
    log("=" * 78)
    log(f"  {'h':>2} {'pairs':>6} {'persist%':>9} {'new%':>7} || "
        f"{'d1':>6} {'d2':>6} {'d3':>6} {'d4':>6} {'far':>6} {'no_sub':>7} {'par@d1':>7}")
    for h in args.horizons:
        s = df[df["horizon"] == h]
        if not len(s):
            continue
        log(f"  {h:>2} {len(s):>6} "
            f"{s['frac_persist_of_Ht'].mean():>9.3f} {s['frac_new_of_Hth'].mean():>7.3f} || "
            f"{s['new_d1'].mean():>6.3f} {s['new_d2'].mean():>6.3f} {s['new_d3'].mean():>6.3f} "
            f"{s['new_d4'].mean():>6.3f} {s['new_far'].mean():>6.3f} "
            f"{s['new_no_subset'].mean():>7.3f} {s['median_parents_at_d1'].median():>7.1f}")

    log("\n" + "-" * 78)
    log("HOW TO READ")
    log("-" * 78)
    d1 = df[df["horizon"] == min(args.horizons)]["new_d1"].mean()
    log(f"  d1 at h={min(args.horizons)} is {d1:.3f}.")
    log("  d1  = new sets reachable by adding ONE mutation to a circulating set.")
    log("        High -> a one-mutation frontier is a defensible candidate pool.")
    log("        Low  -> it is not, and the pool must be built some other way.")
    log("  no_subset = new sets containing NO circulating set as a subset. These")
    log("        cannot be produced by additive expansion at all (mutation loss,")
    log("        recombination, or an unsampled intermediate). Hard ceiling.")
    log("  par@d1 = median number of circulating sets a d1 set is reachable from.")
    log("        Large -> parentage is genuinely ambiguous, which is why this")
    log("        script measures REACHABILITY and not ancestry.")
    log("  Re-run with --min_count 2 or 3: singleton constellations are often")
    log("  sequencing artefacts and can inflate the 'new' bucket substantially.")
    log("  Re-run with --weight_by_count to see the same table by sequence")
    log("  abundance rather than by distinct set.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
