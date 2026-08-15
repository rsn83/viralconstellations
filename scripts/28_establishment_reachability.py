#!/usr/bin/env python
"""
28_establishment_reachability.py

Follow-up to script 27. Two questions about the ~281 establishment events.

Q1  How far is each establishing constellation from the sets circulating at t?
    Script 27 found from_0 = 0.79-1.00: establishing sets are ABSENT at t, not
    merely rare. So there is no trajectory to extrapolate and they must be
    NAMED. The frontier generator names sets one mutation from a circulating
    one. Does that reach them?

      most at distance 1  -> the frontier is the right generator, ~280 targets
      most far away       -> saltational arrivals. The frontier is the wrong
                             generator, and script 24's +0.02 was measured on
                             one-mutation additions that are NOT these events.

Q2  How many events survive dropping the sparse era?
    2023-06 has 27 sets from 174 sequences; 2024-12 has 4 sets from 20. At 174
    sequences a "1% frequency" event is two sequences. Events after mid-2022
    are probably noise and the usable count may be far below 281.

BUG FIX FROM SCRIPT 27
----------------------
Script 27's lead-time loop walked back from month t+k-1, which at h=1 IS
month t -- the month the set was required to be absent in. So lead=0 was
guaranteed by construction at h=1, not a finding. Here lead is measured only
over months strictly before t, and h=1 lead is reported as undefined rather
than as a spurious zero.

Usage
-----
  python scripts/28_establishment_reachability.py
  python scripts/28_establishment_reachability.py --high 0.01 --horizons 3
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


def constellations_of(occ):
    out = {}
    for c, v in occ.items():
        out[frozenset(c)] = v if isinstance(v, (int, float)) else 1
    return out


def min_add_distance(c, by_size, max_k=6):
    """Fewest mutations that must be ADDED to some circulating set to reach c.
    Returns (k, n_sources) or (None, 0) if no circulating set is a subset of c.

    GEOMETRIC reachability, not ancestry. A source is a set within edit
    distance k; no descent is claimed.
    """
    n = len(c)
    for k in range(1, max_k + 1):
        ps = n - k
        if ps < 1:
            break
        hits = sum(1 for c2 in by_size.get(ps, ()) if c2 <= c)
        if hits:
            return k, hits
    return None, 0


def nearest_jaccard(c, sets_t, cap=4000):
    """Closest circulating set by Jaccard, regardless of subset relation.
    Catches events reachable only by swap or loss, which additive distance
    cannot see at all."""
    best, arg = 0.0, None
    for i, c2 in enumerate(sets_t):
        if i >= cap:
            break
        inter = len(c & c2)
        if not inter:
            continue
        j = inter / len(c | c2)
        if j > best:
            best, arg = j, c2
    return best, (len(arg) if arg else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", type=str, default=None)
    ap.add_argument("--end_month", type=str, default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=30)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--low", type=float, default=0.001)
    ap.add_argument("--high", type=float, default=0.01)
    ap.add_argument("--min_seqs", type=int, default=1000,
                    help="a month pair is USABLE only if both months have at "
                         "least this many sequences. At 174 sequences a 1%% "
                         "event is two reads.")
    ap.add_argument("--max_k", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "28_establishment_reach.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    if args.start_month:
        months = [m for m in months if m >= args.start_month]
    if args.end_month:
        months = [m for m in months if m <= args.end_month]

    cache = {}

    def H(mo):
        if mo not in cache:
            with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
                raw = constellations_of(pickle.load(fh))
            filt = {c: v for c, v in raw.items()
                    if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
            tot = sum(filt.values())
            cache[mo] = (filt, {c: v / max(tot, 1) for c, v in filt.items()}, tot)
        return cache[mo]

    log(f"{len(months)} months  low={args.low} high={args.high} "
        f"min_seqs={args.min_seqs}\n")

    rows = []
    for k in args.horizons:
        for i in range(len(months) - k):
            mt, mtk = months[i], months[i + k]
            Ht, ft, tot_t = H(mt)
            Htk, ftk, tot_tk = H(mtk)
            if not Ht or not Htk:
                continue
            usable = (tot_t >= args.min_seqs) and (tot_tk >= args.min_seqs)

            by_size = {}
            for c in Ht:
                by_size.setdefault(len(c), []).append(c)
            sets_t = sorted(Ht, key=lambda c: -Ht[c])

            for c, f_after in ftk.items():
                if f_after <= args.high:
                    continue
                if ft.get(c, 0.0) > args.low:
                    continue

                kk, nsrc = min_add_distance(c, by_size, args.max_k)
                jac, jsize = nearest_jaccard(c, sets_t)

                # lead: months STRICTLY BEFORE t in which c was present at all
                lead = 0
                for j in range(i - 1, -1, -1):
                    if H(months[j])[1].get(c, 0.0) > 0:
                        lead += 1
                    else:
                        break

                rows.append(dict(
                    horizon=k, month_t=mt, month_tk=mtk, year=mt[:4],
                    usable=usable, n_seqs_t=tot_t, n_seqs_tk=tot_tk,
                    set_size=len(c), freq_after=f_after,
                    add_dist=(kk if kk is not None else -1),
                    n_sources=nsrc,
                    nearest_jaccard=jac, nearest_size=jsize,
                    lead_before_t=(np.nan if k == 1 else lead),
                    key="|".join(str(x) for x in sorted(c))))

    if not rows:
        log("no events at these thresholds")
        return
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    def dist_table(d, label):
        log(f"\n  {label}   (n={len(d)}, unique={d['key'].nunique()})")
        if not len(d):
            return
        log(f"    {'add_dist':>10}{'n':>7}{'share':>8}{'med_size':>10}"
            f"{'med_srcs':>10}{'med_jacc':>10}")
        for v in sorted(d["add_dist"].unique()):
            s = d[d["add_dist"] == v]
            lbl = "no subset" if v == -1 else str(int(v))
            log(f"    {lbl:>10}{len(s):>7}{len(s)/len(d):>8.2%}"
                f"{s['set_size'].median():>10.0f}"
                f"{s['n_sources'].median():>10.1f}"
                f"{s['nearest_jaccard'].median():>10.3f}")

    log("=" * 82)
    log("Q1  HOW FAR ARE ESTABLISHING SETS FROM CIRCULATING ONES?")
    log("=" * 82)
    for k in args.horizons:
        dist_table(df[df.horizon == k], f"h={k}  ALL month pairs")

    log("\n" + "=" * 82)
    log(f"Q2  RESTRICTED TO USABLE MONTHS (both >= {args.min_seqs} sequences)")
    log("=" * 82)
    for k in args.horizons:
        dist_table(df[(df.horizon == k) & df.usable], f"h={k}  USABLE only")

    log("\n  events by year (h=3 if available):")
    sub = df[df.horizon == (3 if 3 in args.horizons else args.horizons[0])]
    if len(sub):
        log(sub.groupby("year").agg(
            events=("key", "size"), unique=("key", "nunique"),
            usable=("usable", "sum"), med_seqs=("n_seqs_t", "median"),
            med_size=("set_size", "median"),
            frac_d1=("add_dist", lambda s: float((s == 1).mean())),
        ).round(3).to_string())

    log("\n" + "-" * 82)
    log("READ")
    log("-" * 82)
    u = df[df.usable]
    for k in args.horizons:
        d = u[u.horizon == k]
        if not len(d):
            continue
        d1 = float((d["add_dist"] == 1).mean())
        far = float((d["add_dist"] == -1).mean())
        log(f"  h={k}: {d['key'].nunique():>4} unique usable events | "
            f"dist-1 {d1:.1%} | no-subset {far:.1%} | "
            f"median size {d['set_size'].median():.0f}")
    log("")
    d3 = u[u.horizon == (3 if 3 in args.horizons else args.horizons[0])]
    if len(d3):
        d1 = float((d3["add_dist"] == 1).mean())
        n = d3["key"].nunique()
        if d1 > 0.5:
            log("  Most establishing sets are ONE addition from something circulating.")
            log("  The frontier is the right generator and you have a real target.")
        elif d1 < 0.2:
            log("  Establishing sets are NOT one-mutation additions. They arrive as")
            log("  whole lineages. The frontier cannot name them, and script 24's")
            log("  +0.02 was measured on a different population of events entirely.")
            log("  Check nearest_jaccard: if that is also low, nothing circulating")
            log("  at t resembles them and they are unforecastable from H_t alone.")
        else:
            log("  Mixed. Split the analysis: the dist-1 events are forecastable by")
            log("  local expansion, the rest are not, and they are different problems.")
        log(f"\n  Usable unique events at h=3: {n}")
        if n < 50:
            log("  Below 50 -- too few to train on, whatever the reachability says.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
