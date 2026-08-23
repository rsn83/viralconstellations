#!/usr/bin/env python3
"""
97_count_birth_events.py

Before designing a birth model, count how much supervision exists.

A BIRTH EVENT is a (parent lineage, child lineage, mutation acquired, month)
tuple extracted from GISAID metadata. Pango names encode parentage
(B.1.617.2 -> AY.4 via the alias system, BA.2 -> BA.5, etc.), so the events
are already labelled -- they do not have to be invented.

This script answers four questions:

  1  How many birth events are there at all?
  2  How many mutations does each child add to its parent? (decides whether a
     one-mutation birth rule is enough)
  3  Of the added mutations, how many were ALREADY circulating in some other
     lineage (borrowable) versus genuinely novel (raised from zero)?
  4  How many events survive a prevalence filter, i.e. how many births actually
     established rather than being sequencing noise?

The answer to (1) after filtering caps how many features a birth model can fit.

Usage:
  python 97_count_birth_events.py \
      --metadata data/raw/metadata.tsv.zst \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --months   2020-03:2024-12 \
      --min-seqs 100
"""
import argparse, csv, subprocess, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


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


# Pango alias table: recycled prefixes that hide real parentage.
# Only the common ones; enough to resolve most parents in 2020-2024.
ALIAS = {
    "AY": "B.1.617.2", "BA": "B.1.1.529", "BB": "B.1.617.2.5",
    "BC": "B.1.1.529.1.1.1", "BD": "B.1.1.529.1.17.2", "BE": "B.1.1.529.5.3.1",
    "BF": "B.1.1.529.5.2.1", "BG": "B.1.1.529.2.12.1", "BH": "B.1.1.529.2.38.3",
    "BJ": "B.1.1.529.2.10.1", "BK": "B.1.1.529.5.1.10", "BL": "B.1.1.529.2.75.1",
    "BM": "B.1.1.529.2.75.3", "BN": "B.1.1.529.2.75.5", "BQ": "B.1.1.529.5.3.1.1.1.1",
    "BR": "B.1.1.529.2.75.4", "BS": "B.1.1.529.2.3.2", "BU": "B.1.1.529.5.2.16",
    "BV": "B.1.1.529.5.2.20", "BW": "B.1.1.529.5.6.2", "BY": "B.1.1.529.2.75.6",
    "BZ": "B.1.1.529.5.2.3", "CH": "B.1.1.529.2.75.3.4.1.1",
    "CL": "B.1.1.529.5.1.29", "DV": "B.1.1.529.2.75.3.4.1.1.1.1.1",
    "EG": "B.1.1.529.2.75.3.4.1.1.1.1.1.1.1", "HV": "B.1.1.529.2.86.1.1.11",
    "JN": "B.1.1.529.2.86.1.1", "KP": "B.1.1.529.2.86.1.1.11.1.3",
    "XBB": "recombinant", "XEC": "recombinant",
}


def expand(lin):
    """B.1.617.2 stays; AY.4 -> B.1.617.2.4 ; BA.5.2 -> B.1.1.529.5.2"""
    if not lin or lin[0] == "X": return lin
    head, _, tail = lin.partition(".")
    base = ALIAS.get(head)
    if base is None or base == "recombinant": return lin
    return base + ("." + tail if tail else "")


def parent_of(lin):
    """Immediate parent in the expanded Pango hierarchy."""
    e = expand(lin)
    if "." not in e or e.startswith("X"): return None
    return e.rsplit(".", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--months", default="")
    ap.add_argument("--min-seqs", type=int, default=100,
                    help="a lineage must reach this many sequences to count as established")
    ap.add_argument("--gene", default="S")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    keep = months_in_range(args.months)
    pr2node, names = {}, {}
    with open(args.vocab) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            nid = int(row["node_idx"])
            pr2node[(int(row["aa_pos"]), row["residue"].strip())] = nid
            names[nid] = f"{row['aa_pos']}{row['residue'].strip()}"

    stream, proc = open_stream(args.metadata)
    hdr = stream.readline().decode("utf-8", "replace").rstrip("\n").split("\t")
    col = {c: i for i, c in enumerate(hdr)}
    i_d, i_l, i_a = col["date"], col["pango_lineage"], col["aaSubstitutions"]
    pref = args.gene + ":"

    # lineage -> counter over nodes, count, first month
    lin_nodes = defaultdict(Counter)
    lin_n = Counter()
    lin_first = {}
    node_first = {}                       # node -> first month seen anywhere
    n_rows = 0

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
        nodes = []
        for tok in f[i_a].split(","):
            if not tok.startswith(pref): continue
            mut = tok[len(pref):]
            if len(mut) < 3 or not mut[1:-1].isdigit(): continue
            nid = pr2node.get((int(mut[1:-1]), mut[-1]))
            if nid is None: continue
            nodes.append(nid)
            if nid not in node_first or ym < node_first[nid]:
                node_first[nid] = ym
        lin_n[lin] += 1
        lin_nodes[lin].update(nodes)
        if lin not in lin_first or ym < lin_first[lin]:
            lin_first[lin] = ym

    if proc: proc.stdout.close(); proc.wait()
    print(f"\nrows read {n_rows:,}   lineages seen {len(lin_n):,}")

    # ---- consensus fingerprint per lineage: nodes in >50% of its sequences ----
    fp = {l: {n for n, c in cnt.items() if c > .5 * lin_n[l]}
          for l, cnt in lin_nodes.items()}

    est = {l for l in lin_n if lin_n[l] >= args.min_seqs}
    print(f"lineages with >= {args.min_seqs} sequences: {len(est):,}")

    # ---- birth events ----
    events = []
    no_parent = 0
    exp2lin = {}
    for l in est:
        exp2lin[expand(l)] = l
    for child in sorted(est):
        pe = parent_of(child)
        if pe is None: no_parent += 1; continue
        # walk up until we find an ancestor we actually observed
        while pe and pe not in exp2lin:
            pe = pe.rsplit(".", 1)[0] if "." in pe else None
        if pe is None: no_parent += 1; continue
        parent = exp2lin[pe]
        if parent == child: continue
        added = fp[child] - fp[parent]
        lost = fp[parent] - fp[child]
        events.append(dict(child=child, parent=parent, month=lin_first[child],
                           n_add=len(added), n_lost=len(lost), added=added,
                           n_seq=lin_n[child]))

    print(f"\nbirth events with a resolvable observed parent: {len(events):,}"
          f"   (no parent found: {no_parent:,})")
    nx = sum(1 for l in est if l.startswith("X"))
    print(f"  of the no-parent cases, recombinants (X*): {nx:,}")
    if not events:
        sys.exit("no events -- check the alias table or widen --months")

    # ---- Q2 how many mutations added ----
    na_all = np.array([e["n_add"] for e in events])
    real = [e for e in events if e["n_add"] > 0]
    na = np.array([e["n_add"] for e in real])
    print(f"\n  {int((na_all==0).sum()):,} of {len(events):,} children "
          f"({(na_all==0).mean():.1%}) have the SAME spike fingerprint as their")
    print(f"  parent -- they are Pango splits defined outside spike, not spike")
    print(f"  birth events. Real spike births: {len(real):,}")
    print("\n" + "=" * 70)
    print("Q2   how many spike mutations does a child add to its parent?")
    print("=" * 70)
    print(f"\n  among the {len(real):,} events that DO change spike:")
    print(f"\n  {'added':<8}{'events':>9}{'share':>9}{'cumulative':>12}")
    cum = 0
    for k in range(1, 11):
        c = int((na == k).sum()) if k < 10 else int((na >= 10).sum())
        cum += c
        lbl = str(k) if k < 10 else "10+"
        print(f"  {lbl:<8}{c:>9,}{c/len(na):>9.1%}{cum/len(na):>12.1%}")
    print(f"\n  median {np.median(na):.0f}   mean {na.mean():.2f}   max {na.max()}")
    print(f"  a ONE-mutation birth rule covers {(na==1).mean():.1%} of real spike"
          f" births;  one-or-two covers {(na<=2).mean():.1%}")

    # ---- Q3 borrowable vs novel ----
    print("\n" + "=" * 70)
    print("Q3   of the added mutations, how many were already circulating?")
    print("=" * 70)
    print("""
  Two senses of 'already circulating' matter and are different:
    (a) the mutation had been OBSERVED somewhere before -- so it is in the
        vocabulary and a candidate generator can propose it;
    (b) the mutation is HIGH in some existing group's fingerprint -- so it can
        be reached by mixing groups.
  This test measures (a). Script 96 measures (b). F486V passes (a) and fails
  (b): it existed at low frequency before BA.5 but no group carried it.
""")
    borrow = novel = 0
    per_event = []
    for e in real:
        b = n = 0
        for nd in e["added"]:
            # was this node seen anywhere BEFORE this lineage first appeared?
            if node_first.get(nd, "9999") < e["month"]: b += 1
            else: n += 1
        borrow += b; novel += n
        per_event.append((b, n))
    tot = borrow + novel
    print(f"\n  added mutations total          {tot:,}")
    print(f"    already seen elsewhere first {borrow:,}  ({borrow/max(tot,1):.1%})")
    print(f"    first appearance ever        {novel:,}  ({novel/max(tot,1):.1%})")
    only_b = sum(1 for b, n in per_event if n == 0 and b > 0)
    only_n = sum(1 for b, n in per_event if b == 0 and n > 0)
    both = sum(1 for b, n in per_event if b > 0 and n > 0)
    R = max(len(real), 1)
    print(f"\n  events needing ONLY previously-seen mutations {only_b:,}  ({only_b/R:.1%})")
    print(f"  events needing ONLY first-ever mutations      {only_n:,}  ({only_n/R:.1%})")
    print(f"  events needing BOTH                           {both:,}  ({both/R:.1%})")

    # ---- Q4 how many established ----
    print("\n" + "=" * 70)
    print("Q4   how much supervision survives a prevalence filter?")
    print("=" * 70)
    print(f"\n  {'min sequences':<16}{'events':>9}")
    print(f"  {'':<16}{'all':>9}{'spike births':>15}")
    for thr in (10, 100, 1_000, 10_000, 100_000):
        c = sum(1 for e in events if e["n_seq"] >= thr)
        cr = sum(1 for e in real if e["n_seq"] >= thr)
        print(f"  {thr:<16,}{c:>9,}{cr:>15,}")

    print("\n" + "=" * 70)
    print("WHAT THIS MEANS FOR A BIRTH MODEL")
    print("=" * 70)
    n1k = sum(1 for e in real if e["n_seq"] >= 1000)
    print(f"""
  Supervision available: {len(events):,} events, {n1k:,} of them reaching 1,000+
  sequences. A logistic birth rate log lambda(j,n) = w . features(j,n) needs
  roughly 10-20 events per feature to be fitted honestly, so that caps the
  model at about {max(1, n1k//15)} features if you restrict to established births,
  or {max(1, len(events)//15)} if you use all of them.

  The added-mutation distribution decides the candidate space. If most children
  add one mutation, candidates are (parent, mutation) pairs: K x V, enumerable.
  If most add several, the candidate space is combinatorial and needs either
  multi-step proposals or a different formulation.

  The borrow/novel split decides whether ONE mechanism suffices. Mutations
  already circulating elsewhere can be proposed by copying a column from
  another group. Mutations appearing for the first time cannot -- those need a
  driver of their own (position mutability, structural exposure, whether the
  site has varied before).
""")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write("child\tparent\tmonth\tn_added\tn_lost\tn_seq"
                    "\tadded_nodes\tadded_names\tparent_nodes\n")
            for e in sorted(events, key=lambda e: -e["n_seq"]):
                add = sorted(e["added"])
                par = sorted(fp[e["parent"]])
                f.write(f"{e['child']}\t{e['parent']}\t{e['month']}\t"
                        f"{e['n_add']}\t{e['n_lost']}\t{e['n_seq']}\t"
                        + ",".join(map(str, add)) + "\t"
                        + ",".join(names.get(i, str(i)) for i in add) + "\t"
                        + ",".join(map(str, par)) + "\n")
        print(f"  wrote {len(events):,} events -> {args.out}")


if __name__ == "__main__":
    main()
