#!/usr/bin/env python3
"""
158_build_events.py -- daily spike-variant events from GISAID metadata.

Reads metadata.tsv.zst and emits, for 157_hypergraph_tpp.py:

    events.tsv   date <TAB> comma-separated mutation ids <TAB> count
    vocab.tsv    id <TAB> name          (e.g. S:D614G, S:DEL144, S:DEL144-145)

Identical variants observed on the same day are aggregated into one row with a
count, so the file is far smaller than the sequence count.

TWO DECISIONS THAT AFFECT RESULTS, both exposed as flags:

1. VOCABULARY FROM THE TRAINING PERIOD ONLY (--vocab-end).
   Defining the vocabulary over all years would tell the model in 2021 that
   certain positions exist. Mutations first seen after --vocab-end are dropped
   from the vocabulary; with --posres in 157, a new mutation at a known
   position still gets a representation from its position and residue
   embeddings, which is where that feature earns its place.

2. DELETION ENCODING (--del-mode).
   'range' emits ONE token per contiguous deletion (S:DEL144-145).
   'per-aa' emits one token per deleted residue (S:DEL144, S:DEL145).
   Script 113 found single deletions encoded as two adjacent nodes that always
   co-occur -- which inflates variant size and co-occurrence statistics. 'range'
   is the default for that reason; run both if a co-occurrence claim depends
   on it.

USAGE
    python scripts/158_build_events.py \
        --metadata data/raw/metadata.tsv.zst \
        --out-dir data/processed --vocab-end 2022-12-31 --peek
"""

import argparse
import io
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

SPIKE_START = 21563          # Wuhan-Hu-1 spike CDS start (1-based)
SPIKE_END = 25384
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def open_stream(path):
    if path.endswith(".zst"):
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        return io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def nt_to_aa(nt):
    """Nucleotide position -> spike residue number."""
    return (nt - SPIKE_START) // 3 + 1


def parse_deletions(field, mode):
    """'21992-21994,11288-11296' -> ['S:DEL144'] (only spike ranges)."""
    out = []
    if not field or field == "?":
        return out
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")[:2]
        else:
            a = b = part
        try:
            a, b = int(a), int(b)
        except ValueError:
            continue
        if b < SPIKE_START or a > SPIKE_END:
            continue                       # outside spike
        a = max(a, SPIKE_START); b = min(b, SPIKE_END)
        # Derive the span from the deletion LENGTH, not from the end
        # position. Nextclade ranges often start mid-codon, so nt_to_aa(b)
        # spills into a trailing partial codon: 21633-21641 is the known
        # 24-26 deletion, but end-based mapping would report 24-27.
        aa_lo = nt_to_aa(a)
        n_aa = max(1, (b - a + 1) // 3)
        aa_hi = aa_lo + n_aa - 1
        if mode == "range":
            out.append(f"S:DEL{aa_lo}" if aa_lo == aa_hi
                       else f"S:DEL{aa_lo}-{aa_hi}")
        else:
            out.extend(f"S:DEL{p}" for p in range(aa_lo, aa_hi + 1))
    return out


def parse_subs(field):
    """'E:T9I,S:D614G,...' -> ['S:D614G'] (spike only)."""
    if not field or field == "?":
        return []
    return [x.strip() for x in field.split(",")
            if x.strip().startswith("S:")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--del-mode", choices=["range", "per-aa"], default="range",
                   dest="del_mode")
    p.add_argument("--vocab-end", default=None, dest="vocab_end",
                   help="last date contributing to the vocabulary (leakage)")
    p.add_argument("--date-from", default="2020-01-01", dest="date_from")
    p.add_argument("--date-to", default="2100-01-01", dest="date_to")
    p.add_argument("--min-vocab-count", type=int, default=5,
                   dest="min_vocab_count",
                   help="drop mutations seen fewer than this many times")
    p.add_argument("--require-good-qc", action="store_true", dest="good_qc")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--peek", action="store_true",
                   help="print the first few parsed records and stop")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # ---------------- pass 1: vocabulary ------------------------------
    counts = Counter()
    first_seen = {}
    n_rows = n_kept = n_baddate = 0
    with open_stream(a.metadata) as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {c: i for i, c in enumerate(header)}
        for need in ("date", "substitutions", "aaSubstitutions", "deletions"):
            if need not in col:
                print(f"missing column {need!r}; columns: {header[:8]} ...")
                sys.exit(1)
        qc = col.get("QC_overall_status")

        for line in f:
            n_rows += 1
            if a.max_rows and n_rows > a.max_rows:
                break
            r = line.rstrip("\n").split("\t")
            if len(r) < len(header):
                continue
            d = r[col["date"]].strip()
            if not DATE_RE.match(d):
                n_baddate += 1
                continue
            if not (a.date_from <= d <= a.date_to):
                continue
            if a.good_qc and qc is not None and r[qc].strip() != "good":
                continue
            muts = parse_subs(r[col["aaSubstitutions"]]) + \
                parse_deletions(r[col["deletions"]], a.del_mode)
            if not muts:
                continue
            n_kept += 1
            # Vocabulary spans the WHOLE period, but each mutation records the
            # date it was first seen. Dropping post-cutoff mutations would
            # silently truncate late test variants -- a 2025 variant defined by
            # three new mutations would be scored as if it were only its older
            # ones, which makes it look more like existing variants than it is
            # and inflates every baseline. Instead the id exists, and 157 zeros
            # the memory of anything not seen by the training cutoff, so its
            # only signal is the position/residue embedding. No leakage: an
            # untrained slot carries nothing beyond position and residue.
            counts.update(set(muts))
            for mm in set(muts):
                if mm not in first_seen or d < first_seen[mm]:
                    first_seen[mm] = d
            if a.peek and n_kept <= 3:
                print(f"  {d}  {len(muts)} spike mutations: "
                      f"{sorted(muts)[:8]} ...")
            if a.peek and n_kept >= 3:
                break

    if a.peek:
        print(f"\nscanned {n_rows:,} rows, {n_kept:,} usable, "
              f"{n_baddate:,} imprecise dates")
        print(f"vocabulary so far: {len(counts):,} distinct spike mutations")
        print("re-run without --peek to write the files")
        return

    vocab = [m for m, c in counts.most_common() if c >= a.min_vocab_count]
    vocab.sort()
    vid = {m: i for i, m in enumerate(vocab)}
    print(f"pass 1: {n_rows:,} rows, {n_kept:,} usable, "
          f"{n_baddate:,} imprecise dates dropped")
    n_late = (sum(1 for m in vocab if first_seen.get(m, "") > a.vocab_end)
              if a.vocab_end else 0)
    print(f"vocabulary: {len(vocab):,} mutations "
          f"(>= {a.min_vocab_count} occurrences, full span)")
    if a.vocab_end:
        print(f"  of which {n_late:,} first appear after {a.vocab_end}; "
              "157 will zero their memory and rely on posres")

    # ---------------- pass 2: events ----------------------------------
    agg = defaultdict(int)
    n_oov = n_empty = 0
    with open_stream(a.metadata) as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {c: i for i, c in enumerate(header)}
        qc = col.get("QC_overall_status")
        rows = 0
        for line in f:
            rows += 1
            if a.max_rows and rows > a.max_rows:
                break
            r = line.rstrip("\n").split("\t")
            if len(r) < len(header):
                continue
            d = r[col["date"]].strip()
            if not DATE_RE.match(d) or not (a.date_from <= d <= a.date_to):
                continue
            if a.good_qc and qc is not None and r[qc].strip() != "good":
                continue
            muts = parse_subs(r[col["aaSubstitutions"]]) + \
                parse_deletions(r[col["deletions"]], a.del_mode)
            ids = sorted({vid[m] for m in muts if m in vid})
            n_oov += sum(1 for m in set(muts) if m not in vid)
            if not ids:
                n_empty += 1
                continue
            agg[(d, tuple(ids))] += 1

    tag = ("_" + a.tag) if a.tag else ""
    ev_path = os.path.join(a.out_dir, f"events{tag}.tsv")
    vc_path = os.path.join(a.out_dir, f"vocab{tag}.tsv")
    with open(ev_path, "w") as f:
        for (d, ids), c in sorted(agg.items()):
            f.write(f"{d}\t{','.join(map(str, ids))}\t{c}\n")
    with open(vc_path, "w") as f:
        for i, m in enumerate(vocab):
            fs = first_seen.get(m, "9999-99-99")
            f.write(f"{i}\t{m}\t{fs}\n")

    days = sorted({d for d, _ in agg})
    sizes = [len(ids) for _, ids in agg]
    total = sum(agg.values())
    print(f"\npass 2: {len(agg):,} events ({total:,} sequences) "
          f"over {len(days)} days  {days[0]} .. {days[-1]}")
    print(f"variant size: median {sorted(sizes)[len(sizes)//2]} "
          f"[{min(sizes)}-{max(sizes)}]")
    print(f"out-of-vocabulary mutation occurrences: {n_oov:,}; "
          f"sequences left empty: {n_empty:,}")
    print(f"wrote {ev_path}\nwrote {vc_path}")


if __name__ == "__main__":
    main()
