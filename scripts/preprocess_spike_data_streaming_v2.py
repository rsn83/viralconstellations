"""
Preprocess spike mutations: stream metadata, write haplotypes directly to CSV.
No buffering. True streaming.

Usage:
    python scripts/preprocess_spike_data.py

Outputs:
    data/processed/spike_haplotypes_monthly.csv
"""

import subprocess
import csv
from collections import Counter, defaultdict
from datetime import datetime
import re
from pathlib import Path

# Paths
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

METADATA_FILE = DATA_DIR / "metadata.tsv.zst"
OUTPUT_FILE = PROCESSED_DIR / "spike_haplotypes_monthly.csv"

print("=" * 60)
print("SPIKE MUTATION PREPROCESSING (STREAMING, NO BUFFERING)")
print("=" * 60)

print(f"\n[STEP 1] Checking input file...")
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"{METADATA_FILE} not found. Run from project root.")
print(f"  ✓ Found: {METADATA_FILE}")

print(f"\n[STEP 2] Starting decompression and counting positions...")
process = subprocess.Popen(
    f'zstd -dc {METADATA_FILE}',
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# PASS 1: Count spike positions to find top-52
print("  Reading all sequences to find top-52 positions...")
header_line = process.stdout.readline().strip()
header = header_line.split('\t')

strain_idx = header.index('strain')
date_idx = header.index('date')
aa_subs_idx = header.index('aaSubstitutions')

def extract_spike_mutations_with_residues(aa_subs_str):
    if not aa_subs_str or aa_subs_str == "?":
        return {}
    
    positions = {}
    for mut in aa_subs_str.split(','):
        if mut.startswith('S:'):
            match = re.search(r'S:([A-Z])(\d+)([A-Z\*])', mut)
            if match:
                pos = int(match.group(2))
                derived_residue = match.group(3)
                positions[pos] = derived_residue
    return positions

pos_counts = Counter()
row_count = 0
last_print = 0
skipped_no_date = 0
skipped_no_mutations = 0

for line in process.stdout:
    row_count += 1
    
    if row_count - last_print >= 100000:
        print(f"  ...read {row_count:,} rows, found {len(pos_counts)} spike positions so far")
        last_print = row_count
    
    parts = line.strip().split('\t')
    if len(parts) <= max(strain_idx, date_idx, aa_subs_idx):
        skipped_no_mutations += 1
        continue
    
    date_str = parts[date_idx]
    aa_subs = parts[aa_subs_idx] if aa_subs_idx < len(parts) else ""
    
    if not date_str or date_str == "?":
        skipped_no_date += 1
        continue
    
    try:
        if len(date_str) == 4:
            month_str = f"{date_str}-01"
        else:
            month_str = date_str[:7]
        datetime.strptime(month_str, "%Y-%m")
    except:
        skipped_no_date += 1
        continue
    
    positions_residues = extract_spike_mutations_with_residues(aa_subs)
    if positions_residues:
        pos_counts.update(positions_residues.keys())

process.wait()
print(f"  ✓ PASS 1 complete: read {row_count:,} rows")
print(f"    Total spike positions found: {len(pos_counts)}")

# Find top-52
all_positions = sorted(pos_counts.keys())
top_52 = sorted(all_positions[:52] if len(all_positions) >= 52 else all_positions)
print(f"  ✓ Top 52 positions: {top_52[:10]}... (showing first 10)")

# PASS 2: Stream through again, write haplotypes directly to CSV
print(f"\n[STEP 3] PASS 2: Streaming haplotypes and writing to CSV...")
print(f"  Output: {OUTPUT_FILE}")

process = subprocess.Popen(
    f'zstd -dc {METADATA_FILE}',
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Skip header
process.stdout.readline()

# Open CSV for writing
csv_file = open(OUTPUT_FILE, 'w', newline='')
csv_writer = csv.DictWriter(csv_file, fieldnames=['month', 'haplotype', 'count'])
csv_writer.writeheader()

# Per-month buffer (keep in memory only current month to dedupe)
current_month = None
current_month_haplotypes = defaultdict(int)
row_count = 0
last_print = 0
csv_rows_written = 0

for line in process.stdout:
    row_count += 1
    
    if row_count - last_print >= 100000:
        print(f"  ...pass 2: {row_count:,} rows processed, {csv_rows_written:,} haplotypes written")
        last_print = row_count
    
    parts = line.strip().split('\t')
    if len(parts) <= max(strain_idx, date_idx, aa_subs_idx):
        continue
    
    date_str = parts[date_idx]
    aa_subs = parts[aa_subs_idx] if aa_subs_idx < len(parts) else ""
    
    if not date_str or date_str == "?":
        continue
    
    try:
        if len(date_str) == 4:
            month_str = f"{date_str}-01"
        else:
            month_str = date_str[:7]
        datetime.strptime(month_str, "%Y-%m")
    except:
        continue
    
    # When month changes, flush previous month to disk
    if month_str != current_month:
        if current_month is not None and current_month_haplotypes:
            # Write all haplotypes from previous month
            for haplotype, count in current_month_haplotypes.items():
                csv_writer.writerow({'month': current_month, 'haplotype': haplotype, 'count': count})
                csv_rows_written += 1
            csv_file.flush()  # Force to disk
            print(f"  ✓ Flushed month {current_month}: {len(current_month_haplotypes)} distinct haplotypes")
        
        current_month = month_str
        current_month_haplotypes = defaultdict(int)
    
    # Extract and add to current month
    positions_residues = extract_spike_mutations_with_residues(aa_subs)
    
    if positions_residues:
        # Build haplotype string from top-52 only
        hap_parts = []
        for pos in top_52:
            if pos in positions_residues:
                hap_parts.append(f"{pos}:{positions_residues[pos]}")
            else:
                hap_parts.append(f"{pos}:wt")
        haplotype_str = ";".join(hap_parts)
        current_month_haplotypes[haplotype_str] += 1

# Flush last month
if current_month is not None and current_month_haplotypes:
    for haplotype, count in current_month_haplotypes.items():
        csv_writer.writerow({'month': current_month, 'haplotype': haplotype, 'count': count})
        csv_rows_written += 1
    print(f"  ✓ Flushed month {current_month}: {len(current_month_haplotypes)} distinct haplotypes")

csv_file.close()
process.wait()

print(f"\n[STEP 4] Verification...")
file_size_mb = OUTPUT_FILE.stat().st_size / 1e6
output_df = __import__('pandas').read_csv(OUTPUT_FILE)
print(f"  ✓ File written: {OUTPUT_FILE}")
print(f"    Size: {file_size_mb:.1f} MB")
print(f"    Total rows: {len(output_df):,}")
print(f"    Months: {output_df['month'].nunique()}")
print(f"    Total haplotypes: {len(output_df):,}")

print(f"\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
