#!/usr/bin/env python
"""
79_pango_identity.py

The question
------------
Script 78 showed the cluster-growth signal is almost entirely an artefact of
correspondence. With one pooled partition, cluster identity is handed to you and
momentum comes out at 0.801. Cluster each month causally and match by nearest
consensus, and it falls to 0.041 -- with 26% of matches not even mutual, and a
growth disagreement of 3.0 log units where they are ambiguous.

The diagnosis was that mutation-set similarity gives you the right GROUPS but no
IDENTITY. Ancestry does: a Pango lineage is defined phylogenetically, so B.1.617.2
in November and in December is the same object by construction, and no matching
step is needed.

So: does using Pango as the identity restore the signal?

This is NOT script 50. That used Pango as the GROUPING UNIT, replacing clusters
entirely, and found signals collapse -- Delta is hundreds of AY.x sublineages, so
the grouping is too fine. Here Pango is used as the CORRESPONDENCE: group at a
chosen level of the hierarchy, and let the label carry identity across months.
Granularity is swept explicitly so the two effects are separated.

Three outcomes and what each means
----------------------------------
  momentum near 0.80  identity was the whole problem. Ancestry-based labels fix
                      it, and cluster-based growth results should be redone with
                      phylogenetic identity rather than distance matching.
  momentum near 0.04  identity was not the problem. The pooled result was
                      leaking something else, or cluster growth is genuinely
                      memoryless and the pooled number is an artefact of the
                      partition itself.
  momentum in between at some granularity
                      identity helps but only at the right level of the
                      hierarchy, which is a statement about what unit variant
                      dynamics actually operate on.

One honest caveat, which belongs in any write-up: Pango designations are made
retrospectively. A lineage has a name because someone later decided it deserved
one, and that decision used data from after the months analysed here. Assignment
of an individual sequence to an already-designated lineage is phylogenetic and
needs no future data, but the existence of the label does. So this is not a
clean real-time forecast -- it is a test of whether a stable identity recovers
the signal.

Outputs
-------
outputs/79_lineage_months.csv   month x lineage frequency table
outputs/79_signal.csv           momentum and establishment by granularity
outputs/79_top_lineages.csv     the largest lineages per month, for inspection

Usage
-----
python scripts/79_pango_identity.py
python scripts/79_pango_identity.py --levels 0,1,2,3 --min_mass 0.005
python scripts/79_pango_identity.py --self_test
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# signal measurement -- kept pure so it can be tested
# ----------------------------------------------------------------------------

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 5:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def growth_signal(freq, months, min_mass):
    """
    freq: dict month -> {label: frequency}. The label IS the identity, so no
    matching step exists -- which is the whole point of the comparison.

    momentum    growth at t vs growth at t+1
    establish   growth at t vs mass at t+2
    """
    mom, est, ncl = [], [], []
    for i in range(len(months) - 2):
        m0, m1, m2 = months[i], months[i + 1], months[i + 2]
        f0, f1, f2 = freq[m0], freq[m1], freq[m2]
        keys = [k for k in f0 if f0[k] >= min_mass and k in f1]
        if len(keys) < 5:
            continue
        g0 = np.array([np.log((f1[k] + 1e-6) / (f0[k] + 1e-6)) for k in keys])
        g1 = np.array([np.log((f2.get(k, 0.0) + 1e-6) / (f1[k] + 1e-6))
                       for k in keys])
        m2v = np.array([f2.get(k, 0.0) for k in keys])
        mom.append(spearman(g0, g1))
        est.append(spearman(g0, m2v))
        ncl.append(len(keys))
    return {
        "momentum_growth": float(np.nanmean(mom)) if mom else np.nan,
        "momentum_positive_share": (float(np.nanmean(
            [v > 0 for v in mom if not np.isnan(v)])) if mom else np.nan),
        "establish_growth": float(np.nanmean(est)) if est else np.nan,
        "months": int(np.sum([not np.isnan(v) for v in mom])),
        "mean_units_per_month": float(np.mean(ncl)) if ncl else np.nan,
    }


def collapse(lineage, level):
    """
    level 0 keeps the full label. level n truncates to n dot-separated parts,
    so BA.2.86.1.1 becomes BA (1), BA.2 (2), BA.2.86 (3).
    """
    if level <= 0:
        return lineage
    return ".".join(str(lineage).split(".")[:level])


def self_test():
    print("self-test")

    assert collapse("BA.2.86.1.1", 0) == "BA.2.86.1.1"
    assert collapse("BA.2.86.1.1", 1) == "BA"
    assert collapse("BA.2.86.1.1", 3) == "BA.2.86"
    assert collapse("B.1.617.2", 2) == "B.1"
    print("  lineage collapsing                               ok")

    months = [f"2021-{i:02d}" for i in range(1, 13)] + \
             [f"2022-{i:02d}" for i in range(1, 13)]

    # momentum PRESENT: each lineage has a fixed growth rate
    rng = np.random.default_rng(0)
    n = 12
    rates = np.exp(rng.normal(0, 0.3, n))
    mass = np.full(n, 100.0)
    fq = {}
    for m in months:
        fq[m] = {f"L{k}": mass[k] / mass.sum() for k in range(n)}
        mass = np.clip(mass * rates, 1.0, 1e9)
    r_yes = growth_signal(fq, months, 0.0)

    # momentum ABSENT: growth redrawn every month
    mass = np.full(n, 100.0)
    fq2 = {}
    for m in months:
        fq2[m] = {f"L{k}": mass[k] / mass.sum() for k in range(n)}
        mass = np.clip(mass * np.exp(rng.normal(0, 0.3, n)), 1.0, 1e9)
    r_no = growth_signal(fq2, months, 0.0)

    print(f"  autocorrelated growth -> momentum "
          f"{r_yes['momentum_growth']:+.3f}          ok")
    print(f"  memoryless growth     -> momentum "
          f"{r_no['momentum_growth']:+.3f}          ok")
    assert r_yes["momentum_growth"] > 0.4, r_yes
    assert abs(r_no["momentum_growth"]) < 0.25, r_no
    print("  the statistic separates the two cases            ok")

    # a label that disappears must not crash the growth computation
    fq3 = {months[0]: {"A": 0.5, "B": 0.5},
           months[1]: {"A": 1.0},
           months[2]: {"A": 1.0}}
    growth_signal(fq3, months[:3], 0.0)
    print("  a vanishing lineage is handled                   ok")
    print("all tests passed\n")


# ----------------------------------------------------------------------------
# metadata reading
# ----------------------------------------------------------------------------

def open_metadata(path, usecols, chunksize):
    """
    Reads a possibly zstd-compressed TSV in chunks. Tries pandas' built-in zstd
    support first, then the zstandard module, then the zstd command line tool.
    """
    if not path.endswith(".zst"):
        return pd.read_csv(path, sep="\t", usecols=usecols, dtype=str,
                           chunksize=chunksize, on_bad_lines="skip")
    try:
        return pd.read_csv(path, sep="\t", usecols=usecols, dtype=str,
                           chunksize=chunksize, compression="zstd",
                           on_bad_lines="skip")
    except (ValueError, TypeError, ImportError):
        pass
    try:
        import zstandard as zstd
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        import io
        text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
        return pd.read_csv(text, sep="\t", usecols=usecols, dtype=str,
                           chunksize=chunksize, on_bad_lines="skip")
    except ImportError:
        pass
    try:
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        return pd.read_csv(proc.stdout, sep="\t", usecols=usecols, dtype=str,
                           chunksize=chunksize, on_bad_lines="skip")
    except FileNotFoundError:
        sys.exit("cannot read the .zst file: install zstandard "
                 "(pip install zstandard) or the zstd command line tool")


def detect_columns(path):
    """Read the header only, and report which columns are present."""
    if path.endswith(".zst"):
        try:
            import zstandard as zstd, io
            with open(path, "rb") as fh:
                r = zstd.ZstdDecompressor().stream_reader(fh)
                head = io.TextIOWrapper(r, encoding="utf-8",
                                        errors="replace").readline()
        except ImportError:
            head = subprocess.run(["zstd", "-dc", path], capture_output=True,
                                  text=True).stdout.split("\n", 1)[0]
    else:
        with open(path) as fh:
            head = fh.readline()
    return [c.strip() for c in head.rstrip("\n").split("\t")]


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/raw/metadata.tsv.zst")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--date_col", default=None)
    ap.add_argument("--lineage_col", default=None)
    ap.add_argument("--qc_col", default=None)
    ap.add_argument("--levels", default="0,1,2,3,4")
    ap.add_argument("--min_mass", type=float, default=0.005)
    ap.add_argument("--min_seqs_month", type=int, default=1000)
    ap.add_argument("--start_month", default="2020-03")
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--chunksize", type=int, default=500000)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.metadata):
        sys.exit(f"metadata not found: {args.metadata}\n"
                 "point --metadata at it")

    cols = detect_columns(args.metadata)
    print(f"metadata columns ({len(cols)}): {cols[:14]}"
          f"{' ...' if len(cols) > 14 else ''}")

    def pick(explicit, options, what):
        if explicit:
            if explicit not in cols:
                sys.exit(f"column {explicit!r} not in the file")
            return explicit
        for o in options:
            for c in cols:
                if c.lower().replace(" ", "_") == o:
                    return c
        sys.exit(f"could not find a {what} column; pass it explicitly")

    dcol = pick(args.date_col, ["date", "collection_date", "sample_date"],
                "date")
    lcol = pick(args.lineage_col,
                ["pango_lineage", "pangolin_lineage", "lineage", "pango"],
                "lineage")
    qcol = args.qc_col
    if qcol is None:
        for c in cols:
            if c.lower().replace(" ", "_") in ("qc_overall_status",
                                               "qc.overallstatus"):
                qcol = c
                break
    print(f"using date={dcol!r} lineage={lcol!r} qc={qcol!r}")

    usecols = [dcol, lcol] + ([qcol] if qcol else [])
    counts = defaultdict(lambda: defaultdict(int))
    n_rows = n_kept = 0
    for chunk in open_metadata(args.metadata, usecols, args.chunksize):
        n_rows += len(chunk)
        d = chunk.dropna(subset=[dcol, lcol])
        if qcol:
            d = d[~d[qcol].astype(str).str.lower().eq("bad")]
        # keep only full YYYY-MM-DD dates, so partial dates cannot be
        # misassigned to a month
        ok = d[dcol].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
        d = d[ok]
        d = d[~d[lcol].astype(str).str.lower().isin(
            ["", "none", "unassigned", "nan", "unclassifiable"])]
        for mth, lin in zip(d[dcol].str[:7], d[lcol]):
            counts[mth][lin] += 1
            n_kept += 1
        print(f"  read {n_rows:,} rows, kept {n_kept:,}", end="\r")
    print(f"\nread {n_rows:,} rows, kept {n_kept:,} with a date and a lineage")

    months = sorted(m for m in counts
                    if args.start_month <= m <= args.end_month
                    and sum(counts[m].values()) >= args.min_seqs_month)
    print(f"months with >= {args.min_seqs_month} sequences: {len(months)}"
          f"  ({months[0]} .. {months[-1]})" if months else "no usable months")
    if len(months) < 6:
        sys.exit("too few usable months")

    rows = []
    for m in months:
        tot = sum(counts[m].values())
        for lin, c in counts[m].items():
            rows.append({"month": m, "lineage": lin, "n": c,
                         "freq": c / tot})
    lm = pd.DataFrame(rows)
    lm.to_csv(f"{args.out_dir}/79_lineage_months.csv", index=False)

    top = (lm.sort_values(["month", "freq"], ascending=[True, False])
             .groupby("month").head(5))
    top.to_csv(f"{args.out_dir}/79_top_lineages.csv", index=False)
    print("\nlargest lineage per month (every 6th month):")
    print(top.groupby("month").head(1).iloc[::6][
        ["month", "lineage", "n", "freq"]].round(4).to_string(index=False))

    print("\n" + "=" * 82)
    print("GROWTH SIGNAL WITH PANGO AS THE IDENTITY")
    print("=" * 82)
    out = []
    for lv in [int(x) for x in args.levels.split(",")]:
        freq = {}
        for m in months:
            tot = sum(counts[m].values())
            agg = defaultdict(int)
            for lin, c in counts[m].items():
                agg[collapse(lin, lv)] += c
            freq[m] = {k: v / tot for k, v in agg.items()}
        r = growth_signal(freq, months, args.min_mass)
        r["level"] = lv
        r["label"] = "full" if lv == 0 else f"top-{lv}"
        r["distinct_labels"] = len({collapse(l, lv) for m in months
                                    for l in counts[m]})
        out.append(r)
    sdf = pd.DataFrame(out)[["level", "label", "distinct_labels",
                             "mean_units_per_month", "momentum_growth",
                             "momentum_positive_share", "establish_growth",
                             "months"]]
    sdf.to_csv(f"{args.out_dir}/79_signal.csv", index=False)
    print(sdf.round(4).to_string(index=False))

    print("\nscript 78 on the same quantity, with edit-distance clusters:")
    print("  pooled partition (identity handed over)   momentum  0.8012")
    print("  causal clustering + nearest matching      momentum  0.0408")
    print("  causal, mutual matches only               momentum  0.1476")
    print("\nCALIBRATION: on synthetic lineage tables this statistic returns")
    print("  about +0.6 when growth is autocorrelated by construction and about")
    print("  0.0 when it is redrawn each month, so it separates the two cases.")
    print("\n  Pango momentum near 0.80 -> identity was the whole problem, and")
    print("     cluster-based growth results need phylogenetic identity.")
    print("  near 0.04 -> identity was not the problem: either the pooled")
    print("     number is an artefact of the partition itself, or cluster")
    print("     growth really is memoryless.")
    print("  strongly level-dependent -> identity helps only at the right")
    print("     granularity, which says what unit the dynamics operate on.")
    print("\n  Caveat for the write-up: Pango labels are designated")
    print("  retrospectively. Assigning a sequence to an existing lineage needs")
    print("  no future data, but the existence of the label does. This tests")
    print("  whether a stable identity recovers the signal, not a real-time")
    print("  forecast.")

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
