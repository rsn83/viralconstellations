"""
Preprocess spike mutations: extract top-52 positions with residues,
deduplicate by month, cache as CSV. Streams metadata to avoid freezing.

Usage:
    python scripts/preprocess_spike_data.py

Outputs:
    data/processed/spike_haplotypes_monthly.csv
"""

import subprocess
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import re
from pathlib import Path
import sys

# Paths
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

METADATA_FILE = DATA_DIR / "metadata.tsv.zst"
OUTPUT_FILE = PROCESSED_DIR / "spike_haplotypes_monthly.csv"

print(f"Loading metadata from {METADATA_FILE}...")
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"{METADATA_FILE} not found. Run from project root.")

# Stream decompression
print("Decompressing and parsing metadata (this may take a few minutes)...")
process = subprocess.Popen(
    f'zstd -dc {METADATA_FILE}',
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Read header
header_line = process.stdout.readline().strip()
header = header_line.split('\t')

strain_idx = header.index('strain')
date_idx = header.index('date')
aa_subs_idx = header.index('aaSubstitutions')

print(f"Columns: strain={strain_idx}, date={date_idx}, aaSubstitutions={aa_subs_idx}")

# Extract spike mutations with residues
def extract_spike_mutations_with_residues(aa_subs_str):
    """Parse 'S:E484K,S:N501Y,...' and return dict {position: residue}"""
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

# Stream parse
print("\nParsing sequences...")
pos_counts = Counter()
monthly_data = defaultdict(lambda: defaultdict(int))

row_count = 0
last_print = 0

for line in process.stdout:
    row_count += 1
    
    # Print progress every 100k rows
    if row_count - last_print >= 100000:
        print(f"  Processed {row_count:,} rows...")
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
    
    # Extract spike mutations
    positions_residues = extract_spike_mutations_with_residues(aa_subs)
    
    if positions_residues:
        pos_counts.update(positions_residues.keys())
        
        # Create haplotype string
        hap_parts = []
        for pos in sorted(positions_residues.keys()):
            hap_parts.append(f"{pos}:{positions_residues[pos]}")
        haplotype_str = ";".join(hap_parts)
        
        monthly_data[month_str][haplotype_str] += 1

process.wait()
print(f"\nFinished reading {row_count:,} rows")

# Find top-52 positions
print("\nFinding top-52 positions...")
top_52 = sorted([pos for pos, _ in pos_counts.most_common(52)])
print(f"Top 52 positions: {top_52}")

# Rebuild data with only top-52
print("\nFiltering to top-52 positions...")
rows = []
month_count = 0

for month in sorted(monthly_data.keys()):
    month_count += 1
    if month_count % 10 == 0:
        print(f"  Processing month {month_count}...")
    
    haplotype_counts = defaultdict(int)
    
    for haplotype_str, count in monthly_data[month].items():
        # Parse haplotype and filter to top-52
        hap_dict = {}
        for part in haplotype_str.split(';'):
            pos_res = part.split(':')
            pos = int(pos_res[0])
            res = pos_res[1]
            if pos in top_52:
                hap_dict[pos] = res
        
        # Rebuild filtered haplotype string
        filtered_parts = []
        for pos in top_52:
            if pos in hap_dict:
                filtered_parts.append(f"{pos}:{hap_dict[pos]}")
            else:
                filtered_parts.append(f"{pos}:wt")
        
        filtered_haplotype = ";".join(filtered_parts)
        haplotype_counts[filtered_haplotype] += count
    
    # Write to rows
    for haplotype, count in haplotype_counts.items():
        rows.append({
            'month': month,
            'haplotype': haplotype,
            'count': count
        })

print(f"\nProcessed {len(monthly_data)} months")
print(f"Total haplotypes: {len(rows)}")

# Save to CSV
print(f"\nWriting to {OUTPUT_FILE}...")
output_df = pd.DataFrame(rows)
output_df.to_csv(OUTPUT_FILE, index=False)

file_size_mb = OUTPUT_FILE.stat().st_size / 1e6
print(f"Done. File size: {file_size_mb:.1f} MB")
print(f"Months: {output_df['month'].nunique()}")
print(f"Total haplotypes: {len(output_df)}")
