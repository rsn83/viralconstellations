"""
Step 5b: Recompute per-position frequency (f_t), pairwise co-occurrence
(g_t), and occupied constellation counts from the FULL monthly metadata
-- not the 10k-per-month subsample that 04_build_matrices_from_metadata.py
draws for training the trajectory encoder.

WHY THIS EXISTS
----------------
Check A found a real signal in log-transformed g_t co-occurrence
(AP 0.0179 -> 0.0274, CI clearly excludes zero) computed from the
10k-subsampled M_t matrices. But g_t for rare pairs is noisy at that
sample size -- a real, meaningful co-occurrence that's genuinely
uncommon can show up as 0 in a 10k draw even if it occurs dozens of
times in the full month's real data. The same applies to
new_in_th (appearance labels): a genuinely new constellation could be
under-sampled into non-existence purely by the random 10k draw.

This script re-derives f_t, g_t, and constellation occupancy directly
from EVERY row in metadata.tsv.zst for each month -- no subsampling --
using the SAME position vocabulary (position_vocab.tsv) already built
by 04, so results align column-for-column with your existing matrices.

It does NOT rebuild M_t itself (the (n_seq, P) matrix used to train the
trajectory encoder) -- that's a separate, heavier decision (training on
millions of sequences vs 10k is a real compute tradeoff). This script
only fixes the statistics that feed g_t/label computation, which is
cheap regardless of how many sequences you look at (streaming counts,
not a materialized matrix).

Output: for each month, three files in data/processed/full_data_graphs/:
  {month}_g_t.npy          (P, P) float64 full co-occurrence counts
  {month}_f_t.npy          (P,)   float64 full marginal mutation counts
  {month}_occupied.pkl     {frozenset(col_indices): count} -- full
                            constellation occupancy, keyed by column
                            index (same columns as position_vocab.tsv)
  {month}_n_seq.txt         total sequence count that month (full, not sampled)

Usage:
  python scripts/05_build_full_data_graphs.py
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
import io
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import zstandard
import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse the exact same parsing logic as 04_build_matrices_from_metadata.py
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}


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


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    meta_path = ROOT / cfg["paths"]["raw_dir"] / "metadata.tsv.zst"
    vocab_dir = ROOT / cfg["paths"]["vocab_dir"]
    max_muts = cfg["mutations"]["max_mutations_per_seq"]

    out_dir = ROOT / "data" / "processed" / "full_data_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the EXISTING position vocabulary (from 04's output) so columns
    # align exactly with your current M_t matrices.
    vocab_df = pd.read_csv(vocab_dir / "position_vocab.tsv", sep="\t")
    pos_to_col = dict(zip(vocab_df["aa_pos"], vocab_df["col"]))
    P = len(pos_to_col)
    print(f"Loaded existing vocabulary: P={P} positions "
          f"(from {vocab_dir / 'position_vocab.tsv'})")

    # ── Stream through the FULL metadata file, accumulate per month ──────
    # g_t and f_t accumulated directly as running sums -- no need to hold
    # a giant (n_seq, P) matrix in memory. occupied constellations
    # accumulated as a Counter of frozensets (column indices, matching
    # the same "constellation = set of mutated positions" definition
    # used everywhere else in the project).
    month_g_t = defaultdict(lambda: np.zeros((P, P), dtype=np.float64))
    month_f_t = defaultdict(lambda: np.zeros(P, dtype=np.float64))
    month_occupied = defaultdict(Counter)
    month_n_seq = defaultdict(int)

    dctx = zstandard.ZstdDecompressor()
    with open(meta_path, "rb") as fh:
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

        n_read = n_used = 0
        for line in tqdm(reader, desc="Streaming full metadata", unit=" rows"):
            parts = line.rstrip().split('\t')
            if len(parts) <= aa_idx:
                continue
            n_read += 1

            month = parse_date_month(parts[date_idx] if date_idx < len(parts) else '')
            if month is None:
                continue

            muts = parse_spike_mutations(parts[aa_idx])
            if len(muts) > max_muts:
                continue

            # Map to column indices using the EXISTING vocabulary. Positions
            # not in the vocab (too rare to have cleared 04's prevalence
            # threshold) are dropped -- consistent with how M_t already
            # only encodes vocabulary positions.
            cols = sorted({pos_to_col[p] for p in muts if p in pos_to_col})
            if not cols:
                # sequence has no mutations at any vocabulary position --
                # still counts toward n_seq (denominator for frequency)
                month_n_seq[month] += 1
                continue

            month_n_seq[month] += 1
            n_used += 1

            g = month_g_t[month]
            f = month_f_t[month]
            for c in cols:
                f[c] += 1
            for i in range(len(cols)):
                for j in range(i, len(cols)):
                    g[cols[i], cols[j]] += 1
                    if i != j:
                        g[cols[j], cols[i]] += 1

            month_occupied[month][frozenset(cols)] += 1

    print(f"\nRead {n_read:,} rows, used {n_used:,} with a vocabulary-position mutation")
    print(f"Months found: {len(month_n_seq)}")

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
    print(index_df.to_string(index=False))

    # Quick sanity comparison against the subsampled version, if available
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    sub_index_path = matrix_dir / "index.tsv"
    if sub_index_path.exists():
        sub_index = pd.read_csv(sub_index_path, sep="\t")
        merged = index_df.merge(sub_index[["month", "n_sequences"]], on="month", how="left")
        merged["subsample_ratio"] = merged["n_sequences"] / merged["n_seq_full"]
        print("\nSubsampling ratio (10k-sample n_sequences / full n_seq) by month:")
        print(merged[["month", "n_sequences", "n_seq_full", "subsample_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
