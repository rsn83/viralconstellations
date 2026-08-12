"""
Step 2: Extract spike region from all sequences and join with metadata dates.

The aligned FASTA has no dates — sequence IDs need to be matched against
metadata.tsv.zst to get collection date and country. This script:

  1. Reads metadata → builds a dict: strain_id -> (date_str, country)
  2. Streams the aligned FASTA (never loads it all into memory)
  3. For each sequence:
       - Looks up its date and country in the metadata dict
       - Slices the alignment to spike columns [start_col:end_col+1]
       - Checks spike coverage (fraction non-gap, non-N)
       - If it passes, writes a record to a per-month FASTA file
  4. Writes a summary TSV: n_sequences per month before/after filtering

Why stream rather than load?
  17M sequences × ~30k nt = ~500 GB uncompressed. We never have that in memory.
  Streaming with zstd decompression is ~2-3 GB RAM regardless of input size.

Outputs:
  data/processed/spike_fasta/YYYY-MM.fasta   (one per month, uncompressed)
  data/processed/spike_fasta/summary.tsv     (stats per month)
"""

import sys
import io
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import zstandard as zstd
import pandas as pd
from tqdm import tqdm

# ── helpers ──────────────────────────────────────────────────────────────────

def read_zstd_tsv(path: Path) -> pd.DataFrame:
    print(f"Reading {path.name} ...")
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            return pd.read_csv(
                io.TextIOWrapper(reader, encoding="utf-8"),
                sep="\t",
                low_memory=False,
                dtype=str,
            )


def stream_fasta_zstd(path: Path):
    """
    Yield (header_str, sequence_str) from a zstd-compressed FASTA.
    Uses a simple line-based parser to avoid loading BioPython's SeqRecord
    overhead for 17M records — ~3x faster at this scale.
    """
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            header = None
            seq_lines = []
            for line in text:
                line = line.rstrip()
                if line.startswith(">"):
                    if header is not None:
                        yield header, "".join(seq_lines)
                    header = line[1:]  # strip >
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if header is not None:
                yield header, "".join(seq_lines)


def parse_date_month(date_str: str):
    """
    Return "YYYY-MM" from a date string, or None if not parseable.
    Nextstrain dates can be: "2021-03-15", "2021-03-XX", "2021-03", "2021-XX-XX"
    We need at least year and month.
    """
    if not date_str or date_str in ("?", "NA", "nan", ""):
        return None
    parts = date_str.split("-")
    if len(parts) < 2:
        return None
    year, month = parts[0], parts[1]
    if len(year) != 4 or not year.isdigit():
        return None
    if len(month) != 2 or not month.isdigit():
        return None
    return f"{year}-{month}"


def spike_coverage(spike_seq: str) -> float:
    """Fraction of positions that are actual bases (not gap or N)."""
    good = sum(1 for c in spike_seq if c not in ("-", "N", "n", "?"))
    return good / len(spike_seq) if spike_seq else 0.0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))

    raw_dir      = ROOT / cfg["paths"]["raw_dir"]
    fasta_path   = raw_dir / "aligned.fasta.zst"
    meta_path    = raw_dir / "metadata.tsv.zst"
    spike_dir    = ROOT / cfg["paths"]["spike_fasta_dir"]
    vocab_dir    = ROOT / cfg["paths"]["vocab_dir"]
    spike_dir.mkdir(parents=True, exist_ok=True)

    min_coverage = cfg["spike"]["min_coverage"]

    # Read spike alignment columns
    coords = (vocab_dir / "spike_coords.txt").read_text().split()
    start_col, end_col = int(coords[0]), int(coords[1])
    print(f"Spike columns: {start_col}-{end_col}")

    # ── Step A: Build metadata lookup ─────────────────────────────────────
    meta = read_zstd_tsv(meta_path)

    # Identify the strain/ID column (Nextstrain uses 'strain')
    id_col = None
    for candidate in ["strain", "accession_id", "gisaid_epi_isl", "name", "seqName"]:
        if candidate in meta.columns:
            id_col = candidate
            break
    if id_col is None:
        print(f"Available columns: {list(meta.columns)}")
        raise ValueError("Cannot find ID column in metadata. Check column names above.")

    # Date column
    date_col = None
    for candidate in ["date", "collection_date", "date_collected"]:
        if candidate in meta.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("Cannot find date column in metadata.")

    # Country column (for stratified sampling in the next step)
    country_col = None
    for candidate in ["country", "Country", "geo_loc_name_country"]:
        if candidate in meta.columns:
            country_col = candidate
            break

    print(f"Using columns: id={id_col}, date={date_col}, country={country_col}")
    print(f"Metadata rows: {len(meta)}")

    # Build lookup: strain_id -> (month, country)
    lookup = {}
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Building lookup"):
        sid    = str(row[id_col]).strip()
        month  = parse_date_month(str(row.get(date_col, "")))
        country = str(row.get(country_col, "Unknown")).strip() if country_col else "Unknown"
        if month is not None:
            lookup[sid] = (month, country)

    print(f"Sequences with valid month: {len(lookup)}")

    # ── Step B: Stream FASTA, slice spike, write per-month files ──────────
    # Open one file handle per month (lazy — created on first write)
    month_handles = {}
    month_counts  = defaultdict(lambda: {"total": 0, "passed": 0})

    def get_handle(month: str):
        if month not in month_handles:
            p = spike_dir / f"{month}.fasta"
            month_handles[month] = open(p, "w")
        return month_handles[month]

    print("Streaming aligned FASTA and extracting spike region...")
    n_total = 0
    n_no_meta = 0
    n_low_cov = 0
    n_written = 0

    for header, full_seq in tqdm(stream_fasta_zstd(fasta_path), desc="Sequences"):
        n_total += 1

        # The strain ID is everything up to the first space or pipe
        strain_id = header.split()[0].split("|")[0].strip()

        if strain_id not in lookup:
            n_no_meta += 1
            continue

        month, country = lookup[strain_id]
        month_counts[month]["total"] += 1

        # Check sequence is long enough to contain spike
        if len(full_seq) <= end_col:
            n_low_cov += 1
            continue

        # Slice spike columns
        spike_seq = full_seq[start_col : end_col + 1]

        # Quality filter
        cov = spike_coverage(spike_seq)
        if cov < min_coverage:
            n_low_cov += 1
            continue

        month_counts[month]["passed"] += 1
        n_written += 1

        # Write to per-month FASTA
        # Header format: >strain_id|country  (country used for stratified sampling)
        fh = get_handle(month)
        fh.write(f">{strain_id}|{country}\n{spike_seq}\n")

    # Close all handles
    for fh in month_handles.values():
        fh.close()

    print(f"\nSummary:")
    print(f"  Total sequences in FASTA: {n_total:,}")
    print(f"  No metadata match:        {n_no_meta:,}")
    print(f"  Low spike coverage:       {n_low_cov:,}")
    print(f"  Written to monthly FASTAs:{n_written:,}")
    print(f"  Months written:           {len(month_handles)}")

    # Write summary TSV
    rows = []
    for month in sorted(month_counts):
        rows.append({
            "month":  month,
            "total":  month_counts[month]["total"],
            "passed": month_counts[month]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary_path = spike_dir / "summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"\nWrote: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
