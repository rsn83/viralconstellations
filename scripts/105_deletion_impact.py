#!/usr/bin/env python3
"""
105_deletion_impact.py

89% of sequences have a deletion in spike. Our vocabulary records substitutions
only, so none of them are represented. This measures what that costs.

Four questions:

  1  WHICH deletions, and how common?
     Converts the metadata's nucleotide deletion ranges into spike codon
     positions and counts them. Named deletions (69/70, 144, 143-145, 211)
     are flagged.

  2  DO SEQUENCES COLLAPSE ONTO EACH OTHER?
     Two genomes differing only by a deletion have the SAME mutation set in our
     encoding, so the model cannot tell them apart. This counts how many
     distinct sets would split if deletions were added -- and how many
     sequences sit in those sets.

  3  DO LINEAGES COLLAPSE?
     Same question at lineage level: pairs of Pango lineages that share a
     substitution fingerprint but differ in their deletions. These are
     lineages our model must merge into one block.

  4  HOW MANY BIRTH EVENTS INVOLVE A DELETION?
     Of the parent-child events used for the appearance task, how many are
     defined by acquiring a deletion rather than a substitution. Those are
     events the vocabulary cannot express, so no model could have predicted
     them -- which would partly explain the zeros.

Usage:
  python 105_deletion_impact.py \
      --metadata data/raw/metadata.tsv.zst \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --months   2021-06:2022-06
"""
import argparse, csv, subprocess, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

SPIKE_START = 21563          # nucleotide position of spike codon 1
SPIKE_END = 25384

NAMED = {69: "del69/70 (Alpha, Omicron)", 70: "del69/70",
         144: "del144 (Alpha, Omicron BA.1)",
         143: "del143-145 (Omicron BA.1)", 145: "del143-145",
         211: "del211 (Omicron BA.1)", 156: "del156/157 (Delta)",
         157: "del156/157 (Delta)", 24: "del24-26 (Omicron BA.2)",
         25: "del24-26", 26: "del24-26", 27: "del27 (Omicron)"}


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


def codons_from_ranges(field):
    """'21633-21641,28362-28370' -> set of spike codon positions deleted."""
    out = set()
    for tok in field.split(","):
        if not tok: continue
        if "-" in tok:
            a, _, b = tok.partition("-")
        else:
            a = b = tok
        try:
            a, b = int(a), int(b)
        except ValueError:
            continue
        if b < SPIKE_START or a > SPIKE_END: continue
        a = max(a, SPIKE_START); b = min(b, SPIKE_END)
        for nt in range(a, b + 1):
            out.add((nt - SPIKE_START) // 3 + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--months", default="")
    ap.add_argument("--min-seqs", type=int, default=100)
    ap.add_argument("--events", default="", help="birth_events.tsv from script 97")
    args = ap.parse_args()

    keep = months_in_range(args.months)
    pr2node = {}
    with open(args.vocab) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pr2node[(int(row["aa_pos"]), row["residue"].strip())] = int(row["node_idx"])

    stream, proc = open_stream(args.metadata)
    hdr = stream.readline().decode("utf-8", "replace").rstrip("\n").split("\t")
    col = {c: i for i, c in enumerate(hdr)}
    i_d, i_l = col["date"], col["pango_lineage"]
    i_del, i_aa = col["deletions"], col["aaSubstitutions"]

    del_count = Counter()                       # codon -> sequences
    subs_only = defaultdict(Counter)            # subs-set -> deletion-signature counts
    lin_subs = defaultdict(Counter)             # lineage -> subs-set counts
    lin_dels = defaultdict(Counter)             # lineage -> deletion-sig counts
    lin_n = Counter()
    n_rows = n_kept = n_with_del = 0

    for raw in stream:
        n_rows += 1
        if n_rows % 2_000_000 == 0:
            print(f"  {n_rows:,} rows ...", flush=True)
        f = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
        if len(f) <= max(i_del, i_aa): continue
        d = f[i_d]
        if len(d) < 7 or not d[:4].isdigit(): continue
        if keep and d[:7] not in keep: continue
        lin = f[i_l].strip()
        if not lin or lin in ("?", "unclassified", "None", "Unassigned"): continue

        subs = frozenset(
            pr2node[(int(m[1:-1]), m[-1])]
            for tok in f[i_aa].split(",") if tok.startswith("S:")
            for m in [tok[2:]]
            if len(m) > 2 and m[1:-1].isdigit() and (int(m[1:-1]), m[-1]) in pr2node)
        dels = frozenset(codons_from_ranges(f[i_del]))
        if dels: n_with_del += 1
        for c in dels: del_count[c] += 1

        subs_only[subs][dels] += 1
        lin_subs[lin][subs] += 1
        lin_dels[lin][dels] += 1
        lin_n[lin] += 1
        n_kept += 1

    if proc: proc.stdout.close(); proc.wait()
    print(f"\nrows read {n_rows:,}   kept {n_kept:,}   with a spike deletion "
          f"{n_with_del:,} ({n_with_del/max(n_kept,1):.1%})")

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 74)
    print("1  WHICH SPIKE POSITIONS ARE DELETED")
    print("=" * 74)
    print(f"\n  {'codon':>7}{'sequences':>13}{'share':>9}   known as")
    for c, n in del_count.most_common(15):
        print(f"  {c:>7}{n:>13,}{n/max(n_kept,1):>9.1%}   {NAMED.get(c, '')}")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 74)
    print("2  SEQUENCES OUR ENCODING CANNOT TELL APART")
    print("=" * 74)
    merged = {s: c for s, c in subs_only.items() if len(c) > 1}
    n_merged_seq = sum(sum(c.values()) for c in merged.values())
    print(f"\n  distinct substitution-sets                  {len(subs_only):,}")
    print(f"  ...that hide two or more deletion patterns  {len(merged):,}")
    print(f"  sequences inside those sets                 {n_merged_seq:,}"
          f"  ({n_merged_seq/max(n_kept,1):.1%})")
    print(f"\n  worst offenders:")
    print(f"    {'|subs|':>7}{'sequences':>12}{'distinct deletion patterns':>28}")
    for s, c in sorted(merged.items(), key=lambda kv: -sum(kv[1].values()))[:6]:
        print(f"    {len(s):>7}{sum(c.values()):>12,}{len(c):>28}")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 74)
    print("3  LINEAGES OUR ENCODING MUST MERGE")
    print("=" * 74)
    est = [l for l in lin_n if lin_n[l] >= args.min_seqs]
    fp = {l: lin_subs[l].most_common(1)[0][0] for l in est}
    bysub = defaultdict(list)
    for l in est: bysub[fp[l]].append(l)
    clashes = {k: v for k, v in bysub.items() if len(v) > 1}
    print(f"\n  lineages with >= {args.min_seqs} sequences        {len(est):,}")
    print(f"  groups sharing an identical substitution fingerprint {len(clashes):,}")
    print(f"\n  {'lineages sharing a fingerprint':<46}{'differ in deletions?':>22}")
    shown = 0
    for k, v in sorted(clashes.items(), key=lambda kv: -sum(lin_n[l] for l in kv[1])):
        if shown >= 8: break
        ds = {l: lin_dels[l].most_common(1)[0][0] for l in v}
        differ = len(set(ds.values())) > 1
        print(f"  {', '.join(sorted(v)[:5]):<46}{'YES' if differ else 'no':>22}")
        shown += 1
    n_diff = sum(1 for k, v in clashes.items()
                 if len({lin_dels[l].most_common(1)[0][0] for l in v}) > 1)
    print(f"\n  of {len(clashes):,} clashing groups, {n_diff:,} would be separated"
          f" by adding deletions")

    # ---------------------------------------------------------------- 4
    if args.events and Path(args.events).exists():
        print("\n" + "=" * 74)
        print("4  BIRTH EVENTS THAT INVOLVE A DELETION")
        print("=" * 74)
        n_ev = n_del_ev = n_zero_but_del = 0
        with open(args.events) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                ch, pa = row["child"], row["parent"]
                if ch not in lin_dels or pa not in lin_dels: continue
                n_ev += 1
                dc = lin_dels[ch].most_common(1)[0][0] if lin_dels[ch] else frozenset()
                dp = lin_dels[pa].most_common(1)[0][0] if lin_dels[pa] else frozenset()
                if dc != dp:
                    n_del_ev += 1
                    if int(row["n_added"]) == 0: n_zero_but_del += 1
        print(f"\n  events with both lineages seen            {n_ev:,}")
        print(f"  child's deletions differ from parent's    {n_del_ev:,}"
              f"  ({n_del_ev/max(n_ev,1):.1%})")
        print(f"  ...of which we recorded ZERO added        {n_zero_but_del:,}")
        print("""
  The last line is the important one: those are births our encoding records as
  'nothing changed', because the only change was a deletion. They enter the
  appearance task as unpredictable by construction.""")

    print("""
WHAT TO CONCLUDE
  Section 2 says how many sequences are indistinguishable to the model that
  should not be. Section 3 says which lineages are forced into one block.
  Section 4 says how much of the appearance task was impossible.

  None of this touches the profile HMM result: that degeneracy comes from the
  alignment being given, and a gap symbol keeps every sequence at 1,273
  positions. The two are independent.
""")


if __name__ == "__main__":
    main()
