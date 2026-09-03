"""
Spike mutation mutual information analysis over time.

Extracts spike mutations from metadata, identifies top-52 positions,
computes pairwise MI per month, and plots structure evolution.

Usage:
    python scripts/spike_mi_analysis.py

Outputs:
    results/spike_mi_over_time.png
    results/spike_mi_summary.txt
"""

import subprocess
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import re
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

METADATA_FILE = DATA_DIR / "metadata.tsv.zst"

print(f"Loading metadata from {METADATA_FILE}...")
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"{METADATA_FILE} not found. Run from project root.")

# Decompress and read metadata
result = subprocess.run(
    f'zstd -dc {METADATA_FILE}',
    shell=True,
    capture_output=True,
    text=True
)

lines = result.stdout.strip().split('\n')
header = lines[0].split('\t')

# Find column indices
strain_idx = header.index('strain')
date_idx = header.index('date')
aa_subs_idx = header.index('aaSubstitutions')

print(f"Columns: strain={strain_idx}, date={date_idx}, aaSubstitutions={aa_subs_idx}")

# Parse data
data = []
for line in lines[1:]:
    parts = line.split('\t')
    if len(parts) <= max(strain_idx, date_idx, aa_subs_idx):
        continue
    
    strain = parts[strain_idx]
    date_str = parts[date_idx]
    aa_subs = parts[aa_subs_idx] if aa_subs_idx < len(parts) else ""
    
    if not date_str or date_str == "?":
        continue
    
    # Parse date to YYYY-MM format
    try:
        if len(date_str) == 4:  # Just year
            month_str = f"{date_str}-01"
        else:
            month_str = date_str[:7]  # YYYY-MM
        datetime.strptime(month_str, "%Y-%m")
    except:
        continue
    
    data.append({
        'strain': strain,
        'month': month_str,
        'aa_subs': aa_subs
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} sequences")

# Extract spike mutations and positions
print("\nExtracting spike mutations...")

def extract_spike_mutations(aa_subs_str):
    """Parse 'S:E484K,S:N501Y,...' and return dict {position: 1}"""
    if not aa_subs_str or aa_subs_str == "?":
        return {}
    
    positions = {}
    for mut in aa_subs_str.split(','):
        if mut.startswith('S:'):
            # Parse S:E484K → position 484
            match = re.search(r'S:([A-Z])(\d+)', mut)
            if match:
                pos = int(match.group(2))
                positions[pos] = 1
    return positions

df['spike_positions'] = df['aa_subs'].apply(extract_spike_mutations)

# Find top-52 positions
print("\nFinding top-52 positions...")
pos_counts = Counter()
for positions in df['spike_positions']:
    pos_counts.update(positions.keys())

top_52 = sorted([pos for pos, _ in pos_counts.most_common(52)])
print(f"Top 52 positions: {top_52}")

# Encode sequences as binary (0/1) for top-52 positions
print("\nBinary encoding sequences...")

def encode_top52(positions_dict, top_52_list):
    """Return binary vector of length 52"""
    return np.array([1 if pos in positions_dict else 0 for pos in top_52_list], dtype=int)

df['binary_top52'] = df['spike_positions'].apply(lambda x: encode_top52(x, top_52))

# Group by month, deduplicate
print("\nGrouping by month and deduplicating...")
monthly_data = defaultdict(list)
for month, group in df.groupby('month'):
    # Deduplicate sequences within the month
    unique_seqs = {}
    for _, row in group.iterrows():
        seq_tuple = tuple(row['binary_top52'])
        if seq_tuple not in unique_seqs:
            unique_seqs[seq_tuple] = 0
        unique_seqs[seq_tuple] += 1
    
    # Convert to numpy array
    seqs_array = np.array([list(s) for s in unique_seqs.keys()], dtype=int)
    monthly_data[month] = seqs_array
    print(f"{month}: {len(group)} sequences, {len(unique_seqs)} distinct haplotypes")

# Compute MI per month
print("\nComputing mutual information...")

def compute_mi(col1, col2):
    """Mutual information between two binary columns"""
    joint = Counter(zip(col1, col2))
    px = Counter(col1)
    py = Counter(col2)
    n = len(col1)
    
    mi = 0.0
    for (x, y), count in joint.items():
        pxy = count / n
        pxi = px[x] / n
        pyi = py[y] / n
        if pxy > 0 and pxi > 0 and pyi > 0:
            mi += pxy * np.log2(pxy / (pxi * pyi))
    return mi

months_sorted = sorted(monthly_data.keys())
mi_matrices = {}
mi_means = []

for month in months_sorted:
    seqs = monthly_data[month]
    if seqs.shape[0] < 2:
        print(f"{month}: too few sequences, skipping")
        continue
    
    mi_matrix = np.zeros((52, 52))
    for i in range(52):
        for j in range(52):
            mi_matrix[i, j] = compute_mi(seqs[:, i], seqs[:, j])
    
    mi_matrices[month] = mi_matrix
    # Mean MI excluding diagonal
    off_diag = mi_matrix[np.triu_indices_from(mi_matrix, k=1)]
    mean_mi = np.mean(off_diag)
    mi_means.append(mean_mi)
    print(f"{month}: mean MI = {mean_mi:.4f}")

# Plot
print("\nPlotting...")
plt.figure(figsize=(14, 5))

# Get months that were actually computed
computed_months = sorted(mi_matrices.keys())

# Plot 1: Mean MI over time
plt.subplot(1, 2, 1)
plt.plot(range(len(mi_means)), mi_means, 'o-', linewidth=2, markersize=6)
plt.xlabel('Month (index)')
plt.ylabel('Mean pairwise MI (off-diagonal)')
plt.title('Correlation structure over time (top-52 spike positions)')
plt.grid(True, alpha=0.3)

# Plot 2: Example pairs (484, 501, 417) if in top-52
plt.subplot(1, 2, 2)
pairs_to_plot = [(484, 501, '484-501'), (417, 484, '417-484'), (417, 501, '417-501')]
for pos1, pos2, label in pairs_to_plot:
    try:
        idx1 = top_52.index(pos1)
        idx2 = top_52.index(pos2)
        pair_mi = [mi_matrices[m][idx1, idx2] for m in computed_months]
        plt.plot(range(len(pair_mi)), pair_mi, 'o-', label=label, linewidth=2, markersize=5)
    except ValueError:
        pass

plt.xlabel('Month (index)')
plt.ylabel('MI')
plt.title('Example position pairs over time')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = RESULTS_DIR / "spike_mi_over_time.png"
plt.savefig(plot_path, dpi=100)
print(f"Saved plot to {plot_path}")

# Write summary
summary_path = RESULTS_DIR / "spike_mi_summary.txt"
with open(summary_path, 'w') as f:
    f.write("Spike Mutation MI Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Top 52 positions: {top_52}\n")
    f.write(f"Months analyzed: {len(mi_means)}\n")
    f.write(f"Date range: {computed_months[0]} to {computed_months[-1]}\n\n")
    f.write(f"Mean MI statistics:\n")
    f.write(f"  Min: {min(mi_means):.4f}\n")
    f.write(f"  Max: {max(mi_means):.4f}\n")
    f.write(f"  Mean: {np.mean(mi_means):.4f}\n")
    f.write(f"  Std: {np.std(mi_means):.4f}\n\n")
    f.write(f"Trend: {'rising' if mi_means[-1] > mi_means[0] else 'falling' if mi_means[-1] < mi_means[0] else 'flat'}\n")
    f.write(f"  First month MI: {mi_means[0]:.4f}\n")
    f.write(f"  Last month MI: {mi_means[-1]:.4f}\n")

print(f"Saved summary to {summary_path}")
