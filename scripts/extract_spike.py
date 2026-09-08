#!/usr/bin/env python3
"""
Stream a Nextstrain/GISAID-style metadata.tsv.zst and emit per-sequence spike
constellations.  Nothing but small counters is held in memory.

Outputs
  <out>.tsv          one row per kept sequence (exact date preserved)
  <out>_monthly.tsv  diagnostics only: sequences and mean divergence per month
"""

import argparse
import csv
import io
import re
import shutil
import subprocess
import sys
from collections import defaultdict

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def open_zst(path):
    """Yield text lines from a .zst file, streaming."""
    try:
        import zstandard as zstd
        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    except ImportError:
        pass

    exe = shutil.which("zstdcat") or shutil.which("zstd")
    if exe is None:
        sys.exit("Need either the `zstandard` python package or the `zstd` binary.")
    cmd = [exe, path] if exe.endswith("zstdcat") else [exe, "-dc", path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata", help="path to metadata.tsv.zst")
    ap.add_argument("-o", "--out", default="data/proc/spike",
                    help="output prefix (default: data/proc/spike)")
    ap.add_argument("--gene", default="S",
                    help="gene prefix in aaSubstitutions (default: S)")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--no-qc-filter", action="store_true",
                    help="keep rows regardless of QC_overall_status")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    prefix = args.gene + ":"
    counts = defaultdict(int)
    by_month = defaultdict(int)
    div_sum = defaultdict(float)
    div_n = defaultdict(int)

    stream = open_zst(args.metadata)
    reader = csv.reader(stream, delimiter="\t")

    header = next(reader)
    idx = {name: i for i, name in enumerate(header)}
    for required in ("date", "aaSubstitutions"):
        if required not in idx:
            sys.exit(f"column '{required}' not found in header")

    c_date = idx["date"]
    c_aa = idx["aaSubstitutions"]
    c_div = idx.get("divergence")
    c_qc = idx.get("QC_overall_status")
    c_cov = idx.get("coverage")
    c_pango = idx.get("Nextclade_pango", idx.get("pango_lineage"))

    with open(args.out + ".tsv", "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t", lineterminator="\n")
        w.writerow(["date", "month", "pango", "divergence",
                    "n_spike", "constellation"])

        for row in reader:
            counts["total"] += 1
            if len(row) <= c_aa:
                counts["malformed"] += 1
                continue

            date = row[c_date]
            if not DATE_RE.match(date):
                counts["drop_date"] += 1
                continue

            if c_qc is not None and not args.no_qc_filter:
                if row[c_qc] != "good":
                    counts["drop_qc"] += 1
                    continue

            if c_cov is not None:
                try:
                    if float(row[c_cov]) < args.min_coverage:
                        counts["drop_coverage"] += 1
                        continue
                except ValueError:
                    counts["drop_coverage"] += 1
                    continue

            muts = [m for m in row[c_aa].split(",") if m.startswith(prefix)]
            if not muts:
                counts["drop_no_gene"] += 1
                continue

            div = row[c_div] if c_div is not None else ""
            pango = row[c_pango] if c_pango is not None else ""
            month = date[:7]

            w.writerow([date, month, pango, div, len(muts), ",".join(muts)])

            counts["kept"] += 1
            by_month[month] += 1
            try:
                div_sum[month] += float(div)
                div_n[month] += 1
            except ValueError:
                pass

            if counts["total"] % 500_000 == 0:
                print(f"  ...{counts['total']:,} rows read, "
                      f"{counts['kept']:,} kept", file=sys.stderr)

    with open(args.out + "_monthly.tsv", "w", newline="") as fm:
        w = csv.writer(fm, delimiter="\t", lineterminator="\n")
        w.writerow(["month", "n_seq", "mean_divergence"])
        for m in sorted(by_month):
            mean = div_sum[m] / div_n[m] if div_n[m] else ""
            w.writerow([m, by_month[m], f"{mean:.3f}" if mean != "" else ""])

    print("\n--- summary ---", file=sys.stderr)
    for k in ("total", "malformed", "drop_date", "drop_qc",
              "drop_coverage", "drop_no_gene", "kept"):
        print(f"{k:<16}: {counts[k]:,}", file=sys.stderr)
    print(f"months covered  : {len(by_month)}", file=sys.stderr)
    print(f"\nwrote {args.out}.tsv and {args.out}_monthly.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
