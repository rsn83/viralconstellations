#!/usr/bin/env python3
"""
88_purity_diagnostic.py

One pass over metadata. Reports the correspondence-test ceiling broken out by
(a) month and (b) spike set size, so you can see WHERE spike stops determining
lineage rather than just that it does on average.

Usage:
  python 88_purity_diagnostic.py \
      --metadata data/raw/metadata.tsv.zst \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --months   2020-03:2024-12 \
      --collapse 2
"""
import argparse, csv, subprocess, shutil, sys
from collections import Counter, defaultdict


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
                             bufsize=1024 * 1024)
        return p.stdout, p
    return open(path, "rb"), None


def purity(counter_map):
    """counter_map: key -> Counter(lineage).  -> (weighted purity, n)"""
    num = den = 0
    for c in counter_map.values():
        t = sum(c.values())
        num += c.most_common(1)[0][1]; den += t
    return (num / den if den else float("nan")), den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--months", default="")
    ap.add_argument("--gene", default="S")
    ap.add_argument("--lineage-col", default="pango_lineage")
    ap.add_argument("--collapse", type=int, default=0)
    ap.add_argument("--drop-unmapped", action="store_true",
                    help="drop sequences with any mutation missing from the vocab "
                         "(default keeps them with the mutation silently removed)")
    args = ap.parse_args()

    keep = months_in_range(args.months)
    pr2node = {}
    with open(args.vocab) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pr2node[(int(row["aa_pos"]), row["residue"].strip())] = int(row["node_idx"])

    stream, proc = open_stream(args.metadata)
    hdr = stream.readline().decode("utf-8", "replace").rstrip("\n").split("\t")
    col = {c: i for i, c in enumerate(hdr)}
    i_d, i_l, i_a = col["date"], col[args.lineage_col], col["aaSubstitutions"]
    pref = args.gene + ":"

    by_month = defaultdict(lambda: defaultdict(Counter))   # month -> set -> Counter
    by_size  = defaultdict(lambda: defaultdict(Counter))   # sizebin -> set -> Counter
    overall  = defaultdict(Counter)
    n_rows = n_kept = n_drop = 0

    for raw in stream:
        n_rows += 1
        if n_rows % 2_000_000 == 0:
            print(f"  {n_rows:,} rows ...", flush=True)
        f = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
        if len(f) <= i_a: continue
        d = f[i_d]
        if len(d) < 7 or not d[:4].isdigit(): continue
        ym = d[:7]
        if keep and ym not in keep: continue
        lin = f[i_l].strip()
        if not lin or lin in ("?", "unclassified", "None", "Unassigned",
                              "unclassifiable"): continue
        if args.collapse:
            lin = ".".join(lin.split(".")[:args.collapse + 1])
        nodes, bad = [], False
        for tok in f[i_a].split(","):
            if not tok.startswith(pref): continue
            mut = tok[len(pref):]
            if len(mut) < 3 or not mut[1:-1].isdigit(): continue
            nid = pr2node.get((int(mut[1:-1]), mut[-1]))
            if nid is None: bad = True; continue
            nodes.append(nid)
        if bad and args.drop_unmapped:
            n_drop += 1; continue
        s = frozenset(nodes); k = len(s)
        by_month[ym][s][lin] += 1
        by_size[min(k, 10)][s][lin] += 1
        overall[s][lin] += 1
        n_kept += 1

    if proc: proc.stdout.close(); proc.wait()
    print(f"\nrows read {n_rows:,}   kept {n_kept:,}   dropped(unmapped) {n_drop:,}")

    p_all, n_all = purity(overall)
    print(f"\nOVERALL sequence-weighted purity = {p_all:.4f}  on {n_all:,} sequences")

    print(f"\n{'set size |S|':<14}{'#seq':>12}{'share':>9}{'purity':>10}")
    cum_n = cum_num = 0
    for k in sorted(by_size):
        p, n = purity(by_size[k])
        lbl = f"{k}" if k < 10 else "10+"
        print(f"{lbl:<14}{n:>12,}{n/n_all:>9.1%}{p:>10.4f}")
        if k >= 2: cum_n += n; cum_num += p * n
    if cum_n:
        print(f"{'|S|>=2 pooled':<14}{cum_n:>12,}{cum_n/n_all:>9.1%}{cum_num/cum_n:>10.4f}")

    print(f"\n{'month':<10}{'#seq':>10}{'#sets':>9}{'purity':>10}   {'top lineage':<16}")
    for ym in sorted(by_month):
        p, n = purity(by_month[ym])
        lc = Counter()
        for c in by_month[ym].values(): lc.update(c)
        print(f"{ym:<10}{n:>10,}{len(by_month[ym]):>9,}{p:>10.4f}   "
              f"{lc.most_common(1)[0][0]:<16}")

    print("""
READ
  purity = accuracy of the best possible classifier that sees ONLY the spike
  mutation set. It is the ceiling for the correspondence test; report any ARI
  next to the purity for the same window or the number cannot be interpreted.

  Low purity at |S|<=1 is expected and is not a defect of the representation:
  Pango is called on the whole genome, so lineages differing only outside spike
  are indistinguishable here by construction.

  The month column is the one to act on. If purity is low in 2020 and high from
  mid-2021, run the correspondence experiment on the later window and report the
  early window with its ceiling stated. If it stays low throughout, spike
  underdetermines lineage in general -- a stronger and more important finding.
""")


if __name__ == "__main__":
    main()
