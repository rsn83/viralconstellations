"""
Step 04 (alternative) — Build categorical mutation matrices directly from
the pre-computed aaSubstitutions column in metadata.tsv.zst.

Replaces scripts 01, 02, 03, and the original 04 entirely.
No FASTA processing needed.

aaSubstitutions format: "S:N501Y,S:E484K,ORF1a:K856R,..."
We keep only spike mutations (prefix "S:") and parse the AA change.

Output: identical to the original script 04 —
  data/processed/mutation_matrices/YYYY-MM.npy        (n_seq, P) int8
  data/processed/mutation_matrices/YYYY-MM_posfreq.npy (P, 21) float32
  data/processed/mutation_matrices/index.tsv
  data/vocab/position_vocab.tsv
"""

import sys
from pathlib import Path
from collections import defaultdict
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import zstandard
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Amino acid encoding (same as script 04) ───────────────────────────────────
AA_LIST  = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}  # A=1..Y=20, 0=reference

def parse_spike_mutations(aa_subs_str: str) -> dict:
    """
    Parse aaSubstitutions string → {aa_position: residue_index}.
    Keeps only spike (S:) mutations. Skips stop codons (*).

    Example input: "S:N501Y,S:E484K,ORF1a:K856R,N:R203K"
    Returns: {501: AA_INDEX['Y'], 484: AA_INDEX['K']}
    """
    if not aa_subs_str or aa_subs_str in ('?', '', 'nan'):
        return {}
    muts = {}
    for token in aa_subs_str.split(','):
        token = token.strip()
        if not token.startswith('S:'):
            continue
        change = token[2:]          # e.g. "N501Y"
        if '*' in change:           # skip stop codons
            continue
        try:
            ref_aa  = change[0]
            alt_aa  = change[-1]
            pos_str = change[1:-1]
            pos     = int(pos_str)
            if alt_aa in AA_INDEX:
                muts[pos] = AA_INDEX[alt_aa]
        except (ValueError, IndexError):
            continue
    return muts

def parse_date_month(date_str: str):
    """Return 'YYYY-MM' or None."""
    if not date_str or date_str in ('?', '', 'nan'):
        return None
    parts = date_str.split('-')
    if len(parts) < 2:
        return None
    y, m = parts[0], parts[1]
    if len(y) == 4 and y.isdigit() and len(m) == 2 and m.isdigit():
        return f"{y}-{m}"
    return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg        = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    meta_path  = ROOT / cfg["paths"]["raw_dir"] / "metadata.tsv.zst"
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    vocab_dir  = ROOT / cfg["paths"]["vocab_dir"]
    matrix_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    seqs_per_month  = cfg["sampling"]["sequences_per_month"]
    max_per_country = cfg["sampling"].get("max_per_country", 1500)
    seed            = cfg["sampling"]["seed"]
    min_prev        = cfg["mutations"]["min_position_prevalence"]
    max_muts        = cfg["mutations"]["max_mutations_per_seq"]
    min_month_n     = cfg["sampling"]["min_month_total"]

    rng = np.random.default_rng(seed)

    print("Reading metadata...")
    print(f"Sampling: {seqs_per_month}/month, max {max_per_country}/country")

    # ── Pass 1: read all rows, group by month ─────────────────────────────
    # Store (country, spike_mutation_dict) per month
    month_data = defaultdict(list)   # month -> list of (country, {pos: res_idx})

    dctx = zstandard.ZstdDecompressor()
    with open(meta_path, "rb") as fh:
        reader = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8",
                                  errors="replace")
        header = next(reader).rstrip().split('\t')

        # Column indices
        def col(name, fallbacks=()):
            for n in [name] + list(fallbacks):
                if n in header:
                    return header.index(n)
            return None

        aa_idx  = col('aaSubstitutions')
        date_idx = col('date', ['collection_date'])
        country_idx = col('country', ['Country'])
        gb_idx  = col('genbank_accession')

        if aa_idx is None:
            raise ValueError("No aaSubstitutions column found.")
        if date_idx is None:
            raise ValueError("No date column found.")

        print(f"Columns: aaSubstitutions={aa_idx}, date={date_idx}, "
              f"country={country_idx}, genbank={gb_idx}")

        n_read = n_kept = 0
        for line in tqdm(reader, desc="Reading metadata", unit=" rows"):
            parts = line.rstrip().split('\t')
            if len(parts) <= aa_idx:
                continue
            n_read += 1

            month = parse_date_month(
                parts[date_idx] if date_idx < len(parts) else ''
            )
            if month is None:
                continue

            country = (parts[country_idx].strip()
                       if country_idx is not None and country_idx < len(parts)
                       else 'Unknown')

            muts = parse_spike_mutations(parts[aa_idx])
            if len(muts) > max_muts:
                continue

            month_data[month].append((country, muts))
            n_kept += 1

    print(f"Read {n_read:,} rows, kept {n_kept:,} with valid date+spike muts")
    print(f"Months found: {len(month_data)}")

    # ── Stratified sampling per month ─────────────────────────────────────
    print(f"\nSampling up to {seqs_per_month}/month...")
    sampled_month_data = {}   # month -> list of {pos: res_idx}

    for month in sorted(month_data.keys()):
        rows = month_data[month]
        if len(rows) < min_month_n:
            continue

        # Group by country
        by_country = defaultdict(list)
        for country, muts in rows:
            by_country[country].append(muts)

        # Cap per country
        pool = []
        for country, seq_list in by_country.items():
            if len(seq_list) > max_per_country:
                idx = rng.choice(len(seq_list), max_per_country, replace=False)
                seq_list = [seq_list[i] for i in idx]
            pool.extend(seq_list)

        # Sample from pool
        if len(pool) > seqs_per_month:
            idx  = rng.choice(len(pool), seqs_per_month, replace=False)
            pool = [pool[i] for i in idx]

        sampled_month_data[month] = pool
        print(f"  {month}: {len(pool):,} sequences "
              f"({len(by_country)} countries)")

    # ── Pass 2: build position vocabulary ────────────────────────────────
    print("\nBuilding position vocabulary...")
    position_obs = defaultdict(lambda: defaultdict(int))
    total_seqs = sum(len(v) for v in sampled_month_data.values())

    for muts_list in sampled_month_data.values():
        for muts in muts_list:
            for pos, res_idx in muts.items():
                position_obs[pos][res_idx] += 1

    threshold = int(min_prev * total_seqs)
    variable_positions = sorted(
        pos for pos, obs in position_obs.items()
        if sum(obs.values()) >= threshold
    )
    P = len(variable_positions)
    pos_to_col = {pos: col for col, pos in enumerate(variable_positions)}
    print(f"Variable positions (>={min_prev*100:.2f}% prevalence): {P}")

    # Write position vocabulary
    rows = []
    for col_idx, pos in enumerate(variable_positions):
        total_alt = sum(position_obs[pos].values())
        # Find most common alternate residue for labeling
        most_common_res = max(position_obs[pos], key=position_obs[pos].get)
        most_common_aa  = AA_LIST[most_common_res - 1] if most_common_res > 0 else '?'
        rows.append({
            "col":           col_idx,
            "aa_pos":        pos,
            "top_alt_aa":    most_common_aa,
            "total_alt_obs": total_alt,
        })
    vocab_df = pd.DataFrame(rows)
    vocab_df.to_csv(vocab_dir / "position_vocab.tsv", sep="\t", index=False)
    print(f"Wrote: {vocab_dir / 'position_vocab.tsv'}")

    print("\nTop 20 most variable spike positions:")
    for _, r in vocab_df.nlargest(20, "total_alt_obs").iterrows():
        print(f"  S:{r['aa_pos']:4d}  top_alt={r['top_alt_aa']}  "
              f"obs={r['total_alt_obs']:,}")

    # ── Pass 3: encode categorical matrices ───────────────────────────────
    print("\nEncoding categorical matrices...")
    index_rows = []

    for month in sorted(sampled_month_data.keys()):
        muts_list = sampled_month_data[month]
        n = len(muts_list)
        if n == 0:
            continue

        # (n, P) int8: 0=reference, 1-20=amino acid
        mat = np.zeros((n, P), dtype=np.int8)
        for i, muts in enumerate(muts_list):
            for pos, res_idx in muts.items():
                if pos in pos_to_col:
                    mat[i, pos_to_col[pos]] = res_idx

        # (P, 21) float32: per-position residue frequency distribution
        posfreq = np.zeros((P, 21), dtype=np.float32)
        for j in range(P):
            for k in range(21):
                posfreq[j, k] = float((mat[:, j] == k).sum()) / n

        np.save(matrix_dir / f"{month}.npy",         mat)
        np.save(matrix_dir / f"{month}_posfreq.npy", posfreq)

        mean_muts = float((mat > 0).sum(axis=1).mean())
        index_rows.append({
            "month":       month,
            "n_sequences": n,
            "mean_muts":   round(mean_muts, 2),
            "P_variable":  P,
        })

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(matrix_dir / "index.tsv", sep="\t", index=False)
    print(f"\nWrote {len(index_rows)} monthly matrices")
    print(index_df.to_string(index=False))

    # Sanity: plot mean mutation count over time
    print("\nMutation count trajectory (molecular clock check):")
    for _, r in index_df.iterrows():
        bar = "█" * int(r["mean_muts"])
        print(f"  {r['month']}  {r['mean_muts']:5.1f}  {bar}")


if __name__ == "__main__":
    main()
