#!/usr/bin/env python3
"""
87_pango_from_metadata.py

Build the Pango answer key for the correspondence test.

Streams data/raw/metadata.tsv.zst, pulls `aaSubstitutions` and `pango_lineage`,
keeps only Spike (S:) substitutions, maps each to a node id via posres_vocab.tsv,
and writes one line per DISTINCT mutation set:

    n1,n2,n3<TAB>lineage<TAB>count<TAB>purity

`purity` = fraction of sequences with that set carrying the majority lineage.
Low purity means the set does not determine the lineage -- that ceiling bounds
how well ANY model can score on the correspondence test, so it is reported.

Usage:
  python 87_pango_from_metadata.py \
      --metadata data/raw/metadata.tsv.zst \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --out      data/processed/pango_by_set.tsv \
      --months   2020-03:2021-01
"""
import argparse, csv, subprocess, sys, shutil
from collections import Counter, defaultdict
from pathlib import Path


def months_in_range(spec):
    if not spec:
        return None
    if ":" not in spec:
        return {spec}
    a, b = spec.split(":")
    ya, ma = map(int, a.split("-")); yb, mb = map(int, b.split("-"))
    out, y, m = set(), ya, ma
    while (y, m) <= (yb, mb):
        out.add(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13: m, y = 1, y + 1
    return out


def load_vocab(path):
    """(aa_pos, residue) -> node_idx"""
    pr2node = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pr2node[(int(row["aa_pos"]), row["residue"].strip())] = int(row["node_idx"])
    return pr2node


def open_stream(path):
    if str(path).endswith(".zst"):
        exe = shutil.which("zstdcat") or shutil.which("zstd")
        if exe is None:
            try:
                import zstandard as zstd
            except ImportError:
                sys.exit("need zstdcat on PATH (brew install zstd) or `pip install zstandard`")
            fh = open(path, "rb")
            return zstd.ZstdDecompressor().stream_reader(fh), None
        cmd = [exe, "-dc", str(path)] if exe.endswith("zstdcat") else [exe, "-dc", str(path)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1024 * 1024)
        return p.stdout, p
    return open(path, "rb"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--months", default="", help="e.g. 2020-03:2021-01 (blank = all)")
    ap.add_argument("--gene", default="S", help="gene prefix in aaSubstitutions")
    ap.add_argument("--lineage-col", default="pango_lineage",
                    choices=["pango_lineage", "Nextclade_pango",
                             "Nextstrain_clade", "clade_who", "clade_nextstrain"])
    ap.add_argument("--collapse", type=int, default=0,
                    help="truncate Pango to N dot-levels (2 -> B.1.1.7 stays, "
                         "B.1.1.7.3 -> B.1.1; 0 = no collapse)")
    ap.add_argument("--purity-report", action="store_true",
                    help="print the most ambiguous sets, to diagnose a low ceiling")
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    keep_months = months_in_range(args.months)
    pr2node = load_vocab(args.vocab)
    print(f"vocab: {len(pr2node):,} (pos,residue) pairs")
    if keep_months:
        print(f"months: {min(keep_months)} .. {max(keep_months)}  ({len(keep_months)} months)")

    stream, proc = open_stream(args.metadata)
    header = stream.readline().decode("utf-8", "replace").rstrip("\n").split("\t")
    col = {c: i for i, c in enumerate(header)}
    for need in ("date", args.lineage_col, "aaSubstitutions"):
        if need not in col:
            sys.exit(f"column '{need}' not found in metadata header")
    i_date, i_lin, i_aa = col["date"], col[args.lineage_col], col["aaSubstitutions"]
    pref = args.gene + ":"

    set_lineages = defaultdict(Counter)
    n_rows = n_kept = n_nolin = n_noaa = n_unmapped_row = 0
    unmapped = Counter()

    for raw in stream:
        n_rows += 1
        if n_rows % 1_000_000 == 0:
            print(f"  {n_rows:,} rows, {len(set_lineages):,} distinct sets", flush=True)
        f = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
        if len(f) <= i_aa:
            continue
        d = f[i_date]
        if len(d) < 7 or not d[:4].isdigit():
            continue
        if keep_months and d[:7] not in keep_months:
            continue
        lin = f[i_lin].strip()
        if args.collapse and lin:
            parts = lin.split(".")
            lin = ".".join(parts[:args.collapse + 1])
        if not lin or lin in ("?", "unclassified", "None", "Unassigned"):
            n_nolin += 1; continue
        aa = f[i_aa].strip()
        if not aa:
            n_noaa += 1; continue

        nodes, bad = [], False
        for tok in aa.split(","):
            if not tok.startswith(pref):
                continue
            mut = tok[len(pref):]                      # e.g. D614G
            if len(mut) < 3:
                continue
            new_res = mut[-1]
            pos_s = mut[1:-1]
            if not pos_s.isdigit():
                continue
            key = (int(pos_s), new_res)
            nid = pr2node.get(key)
            if nid is None:
                unmapped[key] += 1; bad = True; continue
            nodes.append(nid)
        if bad:
            n_unmapped_row += 1
        set_lineages[frozenset(nodes)][lin] += 1
        n_kept += 1

    if proc: proc.stdout.close(); proc.wait()

    print(f"\nrows read              {n_rows:,}")
    print(f"rows kept              {n_kept:,}")
    print(f"  no lineage           {n_nolin:,}")
    print(f"  no aaSubstitutions   {n_noaa:,}")
    print(f"  had >=1 unmapped mut {n_unmapped_row:,}")
    print(f"distinct spike sets    {len(set_lineages):,}")
    if unmapped:
        print(f"\ntop unmapped (pos,residue) -- not in your vocabulary:")
        for (p, r), c in unmapped.most_common(10):
            print(f"    {p:>5}{r}  {c:>10,}")
        print("  (deletions and stops are expected here; large counts at "
              "lineage-defining positions are not)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    pur_num = pur_den = 0
    with open(args.out, "w") as fo:
        for s, cnt in set_lineages.items():
            tot = sum(cnt.values())
            if tot < args.min_count:
                continue
            lin, c = cnt.most_common(1)[0]
            purity = c / tot
            pur_num += c; pur_den += tot
            fo.write(",".join(map(str, sorted(s))) + f"\t{lin}\t{tot}\t{purity:.4f}\n")
            n_out += 1
    if args.purity_report:
        amb = sorted(set_lineages.items(),
                     key=lambda kv: -(sum(kv[1].values()) * (1 - kv[1].most_common(1)[0][1] / sum(kv[1].values()))))
        print("\nmost ambiguous sets (these are what drag purity down):")
        for s, cnt in amb[:8]:
            tot = sum(cnt.values()); top = cnt.most_common(4)
            print(f"  |S|={len(s):<3} n={tot:>9,}  purity={top[0][1]/tot:.3f}  "
                  + "  ".join(f"{l}:{c:,}" for l, c in top))

    print(f"\nwrote {n_out:,} sets -> {args.out}")
    if pur_den:
        print(f"sequence-weighted purity = {pur_num/pur_den:.4f}")
        print("  This is the CEILING for the correspondence test: no model can")
        print("  assign lineage from the mutation set better than the set itself")
        print("  determines the lineage. Report it alongside any ARI.")


if __name__ == "__main__":
    main()
