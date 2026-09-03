"""
One pass: extract all mutations, deduplicate to unique haplotypes (~350k),
count positions on unique haplotypes, find top-52, output.
"""

import subprocess
import csv
from collections import Counter, defaultdict
from datetime import datetime
import re
from pathlib import Path
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

METADATA_FILE = DATA_DIR / "metadata.tsv.zst"
OUTPUT_FILE = PROCESSED_DIR / "spike_haplotypes_monthly.csv"

print("SPIKE PREPROCESSING (DEDUPLICATE FIRST)", flush=True)

process = subprocess.Popen(
    ['zstd', '-dc', str(METADATA_FILE)],
    stdout=subprocess.PIPE,
    text=True
)

header = process.stdout.readline().strip().split('\t')
date_idx = header.index('date')
aa_subs_idx = header.index('aaSubstitutions')

def extract_spike_mutations(aa_subs_str):
    if not aa_subs_str or aa_subs_str == "?":
        return {}
    positions = {}
    for mut in aa_subs_str.split(','):
        if mut.startswith('S:'):
            match = re.search(r'S:([A-Z])(\d+)([A-Z\*])', mut)
            if match:
                pos = int(match.group(2))
                residue = match.group(3)
                positions[pos] = residue
    return positions

print("PASS 1: Extracting all mutations and deduplicating...", flush=True)

unique_haplotypes = {}  # haplotype_string → count
pos_counts = Counter()
row_count = 0
last_print = 0

for line in process.stdout:
    row_count += 1
    
    if row_count - last_print >= 100000:
        print(f"  {row_count:,} rows | {len(unique_haplotypes)} unique haplotypes | {len(pos_counts)} positions", flush=True)
        last_print = row_count
    
    parts = line.strip().split('\t')
    if len(parts) <= max(date_idx, aa_subs_idx):
        continue
    
    date_str = parts[date_idx]
    aa_subs = parts[aa_subs_idx]
    
    if not date_str or date_str == "?":
        continue
    
    positions = extract_spike_mutations(aa_subs)
    if not positions:
        continue
    
    # Build haplotype string with ALL positions (no filtering yet)
    hap = ";".join([f"{p}:{positions[p]}" for p in sorted(positions.keys())])
    unique_haplotypes[hap] = unique_haplotypes.get(hap, 0) + 1
    
    # Count positions
    pos_counts.update(positions.keys())

process.wait()

print(f"  ✓ PASS 1 done: {len(unique_haplotypes):,} unique haplotypes", flush=True)
print(f"  Total positions: {len(pos_counts)}", flush=True)

# Find top-52
top_52 = sorted([p for p, _ in pos_counts.most_common(52)])
print(f"  Top 52: {top_52}", flush=True)

print("\nPASS 2: Filtering to top-52 and organizing by month...", flush=True)

# Now re-process to group by month
process = subprocess.Popen(
    ['zstd', '-dc', str(METADATA_FILE)],
    stdout=subprocess.PIPE,
    text=True
)

header = process.stdout.readline()  # skip

month_haplotypes = defaultdict(lambda: defaultdict(int))
row_count = 0

for line in process.stdout:
    row_count += 1
    
    if row_count % 100000 == 0:
        print(f"  {row_count:,} rows", flush=True)
    
    parts = line.strip().split('\t')
    if len(parts) <= max(date_idx, aa_subs_idx):
        continue
    
    date_str = parts[date_idx]
    aa_subs = parts[aa_subs_idx]
    
    if not date_str or date_str == "?":
        continue
    
    try:
        month_str = f"{date_str}-01" if len(date_str) == 4 else date_str[:7]
        datetime.strptime(month_str, "%Y-%m")
    except:
        continue
    
    positions = extract_spike_mutations(aa_subs)
    if not positions:
        continue
    
    # Build haplotype filtered to top-52
    hap = ";".join([f"{p}:{positions.get(p, 'wt')}" for p in top_52])
    month_haplotypes[month_str][hap] += 1

process.wait()

print(f"  ✓ PASS 2 done", flush=True)
print(f"\nWriting to {OUTPUT_FILE}...", flush=True)

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['month', 'haplotype', 'count'])
    writer.writeheader()
    
    for month in sorted(month_haplotypes.keys()):
        for hap, count in month_haplotypes[month].items():
            writer.writerow({'month': month, 'haplotype': hap, 'count': count})

file_size_mb = OUTPUT_FILE.stat().st_size / 1e6
print(f"Done. File: {file_size_mb:.1f} MB", flush=True)
