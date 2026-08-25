#!/usr/bin/env python3
"""
106_add_deletions.py

89% of sequences carry a spike deletion and none are represented: our
vocabulary records substitutions only. The cost, measured by script 105:

  87% of sequences sit in substitution-sets that hide two or more different
     deletion patterns -- the model cannot tell them apart
  40 of 45 lineage groups sharing a substitution fingerprint would separate
  82% of birth events were recorded as "nothing changed" because the only
     change was a deletion

This adds deletion nodes and rebuilds the monthly sets.

WHY NUCLEOTIDE RANGES, NOT CODON POSITIONS
  The obvious move is to convert each deleted nucleotide range into spike
  codons. It does not work reliably. Aligners LEFT-ALIGN indels, so a deletion
  is reported at the leftmost equivalent coordinate rather than on a codon
  boundary:

      reported 21765-21770   known as del69/70   codon 69 truly starts at 21767
      reported 21992-21994   known as del144     codon 144 truly starts at 21992

  The first is shifted by two nucleotides, the second by none. No single offset
  reproduces both, and guessing would put deletions at the wrong positions.

  So a deletion node is identified by its nucleotide range directly:

      node  "del:21765-21770"

  Exact, unambiguous, and sufficient: two sequences share a deletion node only
  if the aligner reported the same range, which is what distinguishes the
  lineages that currently merge. Codon positions can be recovered later from a
  lineage-defining-deletion table if a biological label is wanted.

OUTPUT
  posres_vocab_withdel.tsv     original nodes plus one row per deletion range
  <month>_occupied.pkl         monthly sets, deletion nodes included

Usage:
  python 106_add_deletions.py \
      --metadata data/raw/metadata.tsv.zst \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --out-dir  data/processed/full_data_graphs_withdel \
      --months   2020-03:2024-12 --min-seqs 50
"""
import argparse, csv, pickle, subprocess, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

SPIKE_START, SPIKE_END = 21563, 25384

# ranges that are known lineage-defining deletions, for the report only
KNOWN = {
    (21765, 21770): "del69/70   Alpha, Omicron BA.1",
    (21991, 21993): "del144     Alpha, Omicron BA.1",
    (21992, 21994): "del144     Alpha, Omicron BA.1",
    (21633, 21641): "del24-26   Omicron BA.2",
    (22029, 22034): "del156/157 Delta",
    (22194, 22196): "del211     Omicron BA.1",
    (21987, 21995): "del143-145 Omicron BA.1",
}


def months_in_range(spec):
    if not spec: return None
    if ":" not in spec: return {spec}
    a, b = spec.split(":")
    ya, ma = map(int, a.split("-")); yb, mb = map(int, b.split("-"))
    out, y, m = set(), ya, ma
    while (y, m) <= (yb, mb):
        out.add(f"{y:04d}-{m:02d}"); m += 1
        if m == 13: m, y = 1, y + 1
    return out


def open_stream(path):
    if str(path).endswith(".zst"):
        exe = shutil.which("zstdcat") or shutil.which("zstd")
        if exe is None:
            try: import zstandard as zstd
            except ImportError: sys.exit("need zstdcat or `pip install zstandard`")
            return zstd.ZstdDecompressor().stream_reader(open(path, "rb")), None
        p = subprocess.Popen([exe, "-dc", str(path)], stdout=subprocess.PIPE,
                             bufsize=1 << 20)
        return p.stdout, p
    return open(path, "rb"), None


def spike_del_ranges(field):
    """'21633-21641,28362-28370' -> (('21633','21641'),) restricted to spike."""
    out = []
    for tok in field.split(","):
        if not tok: continue
        a, _, b = tok.partition("-")
        if not b: b = a
        try: a, b = int(a), int(b)
        except ValueError: continue
        if b < SPIKE_START or a > SPIKE_END: continue
        out.append((a, b))
    return tuple(sorted(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--months", default="")
    ap.add_argument("--min-seqs", type=int, default=50,
                    help="a deletion range needs this many sequences to get a node")
    args = ap.parse_args()

    keep = months_in_range(args.months)
    pr2node, maxid = {}, -1
    rows = []
    with open(args.vocab) as f:
        r = csv.DictReader(f, delimiter="\t")
        fields = r.fieldnames
        for row in r:
            nid = int(row["node_idx"]); maxid = max(maxid, nid)
            pr2node[(int(row["aa_pos"]), row["residue"].strip())] = nid
            rows.append(row)
    print(f"existing vocabulary: {len(pr2node):,} substitution nodes")

    # ---- pass 1: count deletion ranges ----
    print("\npass 1: counting deletion ranges ...", flush=True)
    stream, proc = open_stream(args.metadata)
    hdr = stream.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")
    col = {c: i for i, c in enumerate(hdr)}
    i_d, i_del, i_aa = col["date"], col["deletions"], col["aaSubstitutions"]

    rng_count = Counter(); n = 0
    for raw in stream:
        n += 1
        if n % 2_000_000 == 0: print(f"  {n:,}", flush=True)
        f = raw.decode("utf-8", "replace").rstrip("\r\n").split("\t")
        if len(f) <= max(i_del, i_aa): continue
        d = f[i_d]
        if len(d) < 7 or not d[:4].isdigit(): continue
        if keep and d[:7] not in keep: continue
        for rg in spike_del_ranges(f[i_del]):
            rng_count[rg] += 1
    if proc: proc.stdout.close(); proc.wait()

    ranges = [rg for rg, c in rng_count.items() if c >= args.min_seqs]
    ranges.sort()
    print(f"\n  {len(rng_count):,} distinct spike deletion ranges seen")
    print(f"  {len(ranges):,} appear in >= {args.min_seqs} sequences -> given a node")

    print(f"\n  {'range':>18}{'sequences':>12}   known as")
    for rg in sorted(ranges, key=lambda r: -rng_count[r])[:12]:
        print(f"  {str(rg[0])+'-'+str(rg[1]):>18}{rng_count[rg]:>12,}"
              f"   {KNOWN.get(rg, '')}")

    del2node = {}
    for rg in ranges:
        maxid += 1
        del2node[rg] = maxid

    # ---- write the extended vocabulary ----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vpath = out_dir / "posres_vocab_withdel.tsv"
    with open(vpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows: w.writerow(row)
        for rg, nid in del2node.items():
            # aa_pos is a nominal codon so downstream tools that expect one work;
            # the node's identity is the range, recorded in `residue`.
            nominal = (rg[0] - SPIKE_START) // 3 + 1
            w.writerow({"node_idx": nid, "aa_pos": nominal,
                        "residue": f"del{rg[0]}-{rg[1]}",
                        "raw_count": rng_count[rg]})
    print(f"\nwrote {vpath}  ({len(pr2node)+len(del2node):,} nodes)")

    # ---- pass 2: rebuild monthly sets ----
    print("\npass 2: rebuilding monthly sets ...", flush=True)
    stream, proc = open_stream(args.metadata)
    stream.readline()
    monthly = defaultdict(Counter)
    n = n_kept = n_del = 0
    for raw in stream:
        n += 1
        if n % 2_000_000 == 0: print(f"  {n:,}", flush=True)
        f = raw.decode("utf-8", "replace").rstrip("\r\n").split("\t")
        if len(f) <= max(i_del, i_aa): continue
        d = f[i_d]
        if len(d) < 7 or not d[:4].isdigit(): continue
        ym = d[:7]
        if keep and ym not in keep: continue
        nodes = []
        for tok in f[i_aa].split(","):
            if not tok.startswith("S:"): continue
            m = tok[2:]
            if len(m) < 3 or not m[1:-1].isdigit(): continue
            nid = pr2node.get((int(m[1:-1]), m[-1]))
            if nid is not None: nodes.append(nid)
        got = False
        for rg in spike_del_ranges(f[i_del]):
            nid = del2node.get(rg)
            if nid is not None: nodes.append(nid); got = True
        if got: n_del += 1
        monthly[ym][frozenset(nodes)] += 1
        n_kept += 1
    if proc: proc.stdout.close(); proc.wait()

    print(f"\n  {n_kept:,} sequences, {n_del:,} carrying at least one "
          f"deletion node ({n_del/max(n_kept,1):.1%})")

    for ym, sets in sorted(monthly.items()):
        with open(out_dir / f"{ym}_occupied.pkl", "wb") as f:
            pickle.dump(dict(sets), f)
    print(f"  wrote {len(monthly)} monthly files to {out_dir}")

    # ---- what changed ----
    print("\n" + "=" * 70)
    print("WHAT CHANGED")
    print("=" * 70)
    tot_sets = sum(len(s) for s in monthly.values())
    print(f"\n  distinct sets per month, with deletions:")
    for ym in sorted(monthly)[:6]:
        print(f"    {ym}   {len(monthly[ym]):>7,} sets   "
              f"{sum(monthly[ym].values()):>10,} sequences")
    print(f"""
  Rerun with this vocabulary and out-dir:
    96  is a new lineage reachable       -- deletions are what separate the
                                            sublineages that currently merge
    97  birth events                      -- 82% were recorded as "nothing
                                            changed"; they should now appear
    98  can the new row be named          -- previously scored against labels
                                            that mostly did not exist

  The profile HMM result is unaffected. Its degeneracy comes from the alignment
  being given, and a deletion node keeps every sequence at a fixed set of
  possible positions.
""")


if __name__ == "__main__":
    main()
