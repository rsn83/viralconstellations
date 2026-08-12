"""
Step 3: Stratified monthly sampling.

Why stratify by country?
  UK, USA, and Denmark together contributed ~50% of all GISAID sequences
  during 2021-2022, but represent ~10% of global cases. Naive random sampling
  would make the training data heavily reflect sequencing effort, not viral
  prevalence. We cap each country's contribution per month.

Sampling logic:
  For each month:
    1. Count sequences per country.
    2. Cap each country at min(n_country, max_per_country).
    3. Sample up to sequences_per_month from the resulting pool, uniformly.
    4. If the pool is smaller than sequences_per_month, take all of it.

This gives:
  - No single country dominates (cap)
  - Geographic diversity is preserved (proportional within cap)
  - Total volume is controlled (sequences_per_month)

Outputs:
  data/processed/monthly_samples/YYYY-MM_ids.txt  (one ID per line)
  data/processed/monthly_samples/sampling_report.tsv
"""

import sys
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import pandas as pd
from tqdm import tqdm


def read_fasta_headers(path: Path):
    """
    Stream a (plain, not zstd) FASTA and yield (strain_id, country) from headers.
    Header format (written by Step 2): >strain_id|country
    """
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split("|")
                strain_id = parts[0]
                country   = parts[1] if len(parts) > 1 else "Unknown"
                yield strain_id, country


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))

    spike_dir   = ROOT / cfg["paths"]["spike_fasta_dir"]
    sample_dir  = ROOT / cfg["paths"]["monthly_sample_dir"]
    sample_dir.mkdir(parents=True, exist_ok=True)

    target_n        = cfg["sampling"]["sequences_per_month"]   # e.g. 10000
    max_per_country = cfg["sampling"].get("max_per_country")   # e.g. 1500 or null
    seed            = cfg["sampling"]["seed"]
    min_month_total = cfg["sampling"]["min_month_total"]

    random.seed(seed)

    summary_rows = []

    fasta_files = sorted(spike_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta files found in {spike_dir}. Run Step 2 first.")

    print(f"Found {len(fasta_files)} monthly FASTA files.")

    for fasta_path in tqdm(fasta_files, desc="Sampling months"):
        month = fasta_path.stem  # "YYYY-MM"

        # Read all IDs, grouped by country
        country_ids = defaultdict(list)
        for sid, country in read_fasta_headers(fasta_path):
            country_ids[country].append(sid)

        n_total = sum(len(v) for v in country_ids.values())

        if n_total < min_month_total:
            print(f"  {month}: {n_total} sequences < min {min_month_total}, skipping.")
            continue

        # Apply country cap
        pool = []
        for country, ids in country_ids.items():
            if max_per_country is not None and len(ids) > max_per_country:
                ids = random.sample(ids, max_per_country)
            pool.extend(ids)

        # Sample from pool
        n_pool = len(pool)
        if n_pool <= target_n:
            sampled = pool
        else:
            sampled = random.sample(pool, target_n)

        # Write sampled IDs
        out_path = sample_dir / f"{month}_ids.txt"
        out_path.write_text("\n".join(sampled) + "\n")

        summary_rows.append({
            "month":            month,
            "total_sequences":  n_total,
            "n_countries":      len(country_ids),
            "pool_after_cap":   n_pool,
            "sampled":          len(sampled),
        })

    summary = pd.DataFrame(summary_rows)
    report_path = sample_dir / "sampling_report.tsv"
    summary.to_csv(report_path, sep="\t", index=False)
    print(f"\nWrote: {report_path}")
    print(summary.to_string(index=False))

    # Rough check: how many sequences across all months?
    total_sampled = summary["sampled"].sum()
    print(f"\nTotal sequences across all months: {total_sampled:,}")
    print(f"Months included: {len(summary)}")


if __name__ == "__main__":
    main()
