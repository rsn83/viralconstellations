"""
Step 5 (v2): Rebuild f_t, g_t, and occupied constellation counts using
(position, residue) nodes instead of position-only nodes.

WHY THIS REPLACES THE EARLIER VERSION
---------------------------------------
The original 05_build_full_data_graphs.py (and 04's matrices) used
"position" as the node/column unit -- e.g. column for spike position 501,
regardless of whether it carries Y, K, or any other residue. That means
"501Y" and "501K" collapse into the same node, which loses exactly which
specific substitution occurred and who it co-occurs with. This version
uses (position, residue) as the node identity throughout -- e.g. node
"501Y" and node "501K" are now distinct.

TWO PASSES OVER THE RAW DATA (unavoidable -- vocabulary must be built
before node indices can be assigned):
  Pass A: stream metadata once, count how often each (position, residue)
          pair occurs (only at positions already in your existing
          position_vocab.tsv, to keep the position-level prevalence
          filtering you already validated). Keep pairs seen at least
          `min_residue_count` times (default 3, matching the EVE flu
          paper's own filtering of mutations seen <3 times as likely
          sequencing noise).
  Pass B: stream metadata again, map each sequence's mutations to
          (position,residue) node indices using the vocabulary from
          Pass A, accumulate g_t / f_t / occupied per month exactly as
          before, just at the finer node granularity.

This will take roughly 2x as long as the original single-pass script,
since it reads the full metadata file twice.

Output: data/processed/full_data_graphs_posres/
  posres_vocab.tsv                 node_idx, aa_pos, residue (one-letter)
  {month}_g_t.npy                  (N, N) co-occurrence, N = len(vocab)
  {month}_f_t.npy                  (N,)   marginal counts
  {month}_occupied.pkl             {frozenset(node_indices): count}
  {month}_n_seq.txt
  index.tsv

Usage:
  python scripts/05b_build_full_data_graphs_posres.py --min_residue_count 3
"""

import sys, argparse
from pathlib import Path
from collections import defaultdict, Counter
import io
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--min_residue_count", type=int, default=3,
                    help="minimum raw observation count for a (position,residue) "
                         "pair to be included in the vocabulary")
args = parser.parse_args()

import yaml
import zstandard
import numpy as np
import pandas as pd
from tqdm import tqdm

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
IDX_TO_AA = {v: k for k, v in AA_INDEX.items()}


def parse_spike_mutations(aa_subs_str: str) -> dict:
    if not aa_subs_str or aa_subs_str in ('?', '', 'nan'):
        return {}
    muts = {}
    for token in aa_subs_str.split(','):
        token = token.strip()
        if not token.startswith('S:'):
            continue
        change = token[2:]
        if '*' in change:
            continue
        try:
            alt_aa = change[-1]
            pos = int(change[1:-1])
            if alt_aa in AA_INDEX:
                muts[pos] = AA_INDEX[alt_aa]
        except (ValueError, IndexError):
            continue
    return muts


def parse_date_month(date_str: str):
    if not date_str or date_str in ('?', '', 'nan'):
        return None
    parts = date_str.split('-')
    if len(parts) < 2:
        return None
    y, m = parts[0], parts[1]
    if len(y) == 4 and y.isdigit() and len(m) == 2 and m.isdigit():
        return f"{y}-{m}"
    return None


def open_metadata_reader(meta_path: Path):
    dctx = zstandard.ZstdDecompressor()
    fh = open(meta_path, "rb")
    reader = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="replace")
    header = next(reader).rstrip().split('\t')

    def col(name, fallbacks=()):
        for n in [name] + list(fallbacks):
            if n in header:
                return header.index(n)
        return None

    aa_idx = col('aaSubstitutions')
    date_idx = col('date', ['collection_date'])
    if aa_idx is None or date_idx is None:
        raise ValueError("Required columns not found in metadata.")
    return fh, reader, aa_idx, date_idx


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    meta_path = ROOT / cfg["paths"]["raw_dir"] / "metadata.tsv.zst"
    vocab_dir = ROOT / cfg["paths"]["vocab_dir"]
    max_muts = cfg["mutations"]["max_mutations_per_seq"]

    out_dir = ROOT / "data" / "processed" / "full_data_graphs_posres"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the existing position-level vocabulary as the set of "in-scope"
    # positions (already prevalence-filtered) -- we're refining granularity
    # within that set, not re-deciding which positions matter.
    existing_vocab = pd.read_csv(vocab_dir / "position_vocab.tsv", sep="\t")
    valid_positions = set(existing_vocab["aa_pos"].tolist())
    print(f"Using {len(valid_positions)} positions from existing position_vocab.tsv")

    # ── PASS A: count (position, residue) occurrences to build vocabulary ──
    print("\nPass A: counting (position, residue) occurrences ...")
    posres_counts = Counter()
    fh, reader, aa_idx, date_idx = open_metadata_reader(meta_path)
    try:
        for line in tqdm(reader, desc="Pass A", unit=" rows"):
            parts = line.rstrip().split('\t')
            if len(parts) <= aa_idx:
                continue
            muts = parse_spike_mutations(parts[aa_idx])
            if len(muts) > max_muts:
                continue
            for pos, res in muts.items():
                if pos in valid_positions:
                    posres_counts[(pos, res)] += 1
    finally:
        fh.close()

    kept = [(pos, res) for (pos, res), c in posres_counts.items() if c >= args.min_residue_count]
    kept.sort()
    posres_to_idx = {pr: i for i, pr in enumerate(kept)}
    N = len(kept)
    print(f"Vocabulary: {N} (position,residue) nodes "
          f"(from {len(posres_counts)} observed, min_count={args.min_residue_count})")

    vocab_rows = [{"node_idx": i, "aa_pos": pos, "residue": IDX_TO_AA[res], "raw_count": posres_counts[(pos,res)]}
                  for (pos, res), i in posres_to_idx.items()]
    vocab_df = pd.DataFrame(vocab_rows)
    vocab_df.to_csv(out_dir / "posres_vocab.tsv", sep="\t", index=False)
    print(f"Wrote: {out_dir / 'posres_vocab.tsv'}")

    # ── PASS B: accumulate g_t / f_t / occupied per month ───────────────
    print("\nPass B: accumulating per-month graphs ...")
    month_g_t = defaultdict(lambda: np.zeros((N, N), dtype=np.float64))
    month_f_t = defaultdict(lambda: np.zeros(N, dtype=np.float64))
    month_occupied = defaultdict(Counter)
    month_n_seq = defaultdict(int)

    fh, reader, aa_idx, date_idx = open_metadata_reader(meta_path)
    try:
        for line in tqdm(reader, desc="Pass B", unit=" rows"):
            parts = line.rstrip().split('\t')
            if len(parts) <= aa_idx:
                continue

            month = parse_date_month(parts[date_idx] if date_idx < len(parts) else '')
            if month is None:
                continue

            muts = parse_spike_mutations(parts[aa_idx])
            if len(muts) > max_muts:
                continue

            month_n_seq[month] += 1

            nodes = sorted({posres_to_idx[(pos, res)] for pos, res in muts.items()
                             if (pos, res) in posres_to_idx})
            if not nodes:
                continue

            g = month_g_t[month]
            f = month_f_t[month]
            for n in nodes:
                f[n] += 1
            for a in range(len(nodes)):
                for b in range(a, len(nodes)):
                    i, j = nodes[a], nodes[b]
                    g[i, j] += 1
                    if i != j:
                        g[j, i] += 1

            month_occupied[month][frozenset(nodes)] += 1
    finally:
        fh.close()

    # ── Write outputs ─────────────────────────────────────────────────
    index_rows = []
    for month in sorted(month_n_seq.keys()):
        np.save(out_dir / f"{month}_g_t.npy", month_g_t[month])
        np.save(out_dir / f"{month}_f_t.npy", month_f_t[month])
        with open(out_dir / f"{month}_occupied.pkl", "wb") as f:
            pickle.dump(dict(month_occupied[month]), f)
        (out_dir / f"{month}_n_seq.txt").write_text(str(month_n_seq[month]))
        index_rows.append({
            "month": month,
            "n_seq_full": month_n_seq[month],
            "n_distinct_constellations_full": len(month_occupied[month]),
        })

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(out_dir / "index.tsv", sep="\t", index=False)
    print(f"\nWrote {len(index_rows)} months to {out_dir}")
    print(f"N (position,residue) nodes = {N}  (vs. {len(valid_positions)} position-only nodes before)")
    print(index_df.to_string(index=False))


if __name__ == "__main__":
    main()
