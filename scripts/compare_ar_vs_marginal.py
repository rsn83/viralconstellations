"""
Compare autoregressive (joint) vs marginal-only forecasting.

Test whether learning position dependencies helps forecast accuracy
or whether position independence is sufficient.

Usage:
    python scripts/compare_ar_vs_marginal.py

Outputs:
    results/ar_vs_marginal_comparison.txt
    results/ar_vs_marginal_metrics.csv
"""

import subprocess
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import re
from pathlib import Path
from scipy.special import xlogy

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

strain_idx = header.index('strain')
date_idx = header.index('date')
aa_subs_idx = header.index('aaSubstitutions')

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
    
    try:
        if len(date_str) == 4:
            month_str = f"{date_str}-01"
        else:
            month_str = date_str[:7]
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

# Extract spike mutations
print("\nExtracting spike mutations...")

def extract_spike_mutations(aa_subs_str):
    if not aa_subs_str or aa_subs_str == "?":
        return {}
    
    positions = {}
    for mut in aa_subs_str.split(','):
        if mut.startswith('S:'):
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

def encode_top52(positions_dict, top_52_list):
    return np.array([1 if pos in positions_dict else 0 for pos in top_52_list], dtype=int)

df['binary_top52'] = df['spike_positions'].apply(lambda x: encode_top52(x, top_52))

# Group by month
print("\nGrouping by month...")
monthly_data = {}
for month, group in df.groupby('month'):
    unique_seqs = {}
    for _, row in group.iterrows():
        seq_tuple = tuple(row['binary_top52'])
        unique_seqs[seq_tuple] = unique_seqs.get(seq_tuple, 0) + 1
    
    seqs_array = np.array([list(s) for s in unique_seqs.keys()], dtype=int)
    monthly_data[month] = seqs_array

months_sorted = sorted(monthly_data.keys())
print(f"Months: {len(months_sorted)}")

# Helper functions
def compute_marginals(seqs):
    """Compute per-position frequency for each position"""
    n_seqs = seqs.shape[0]
    marginals = {}
    for i in range(52):
        p_1 = np.mean(seqs[:, i])
        marginals[i] = {'p_1': p_1, 'p_0': 1 - p_1}
    return marginals

def score_marginal_only(test_seqs, marginals):
    """Log-likelihood under marginal model (independence)"""
    ll = 0.0
    for seq in test_seqs:
        for i in range(52):
            p = marginals[i]['p_1'] if seq[i] == 1 else marginals[i]['p_0']
            if p > 0:
                ll += np.log2(p)
    return ll / len(test_seqs)

def compute_joint(seqs):
    """Compute full joint distribution (exact for small data)"""
    # Store counts for observed sequences
    joint = {}
    for seq in seqs:
        seq_tuple = tuple(seq)
        joint[seq_tuple] = joint.get(seq_tuple, 0) + 1
    
    # Normalize
    total = sum(joint.values())
    for key in joint:
        joint[key] /= total
    
    return joint

def score_joint(test_seqs, joint, marginals_for_smoothing=None):
    """Log-likelihood under joint model"""
    ll = 0.0
    for seq in test_seqs:
        seq_tuple = tuple(seq)
        if seq_tuple in joint:
            ll += np.log2(joint[seq_tuple])
        else:
            # Unseen sequence: fall back to marginals (Laplace smoothing alternative)
            if marginals_for_smoothing:
                p_seq = 1.0
                for i in range(52):
                    p = marginals_for_smoothing[i]['p_1'] if seq[i] == 1 else marginals_for_smoothing[i]['p_0']
                    p_seq *= p
                if p_seq > 0:
                    ll += np.log2(p_seq)
    
    return ll / len(test_seqs) if len(test_seqs) > 0 else 0.0

# Run evaluation: train on months 1–N-1, test on month N
print("\nRunning evaluation (train ≤ month, test = month+1)...")
results = []

for i in range(len(months_sorted) - 1):
    train_month = months_sorted[i]
    test_month = months_sorted[i + 1]
    
    train_seqs = monthly_data[train_month]
    test_seqs = monthly_data[test_month]
    
    if train_seqs.shape[0] < 2 or test_seqs.shape[0] < 2:
        continue
    
    # Marginal model
    marginals = compute_marginals(train_seqs)
    ll_marginal = score_marginal_only(test_seqs, marginals)
    
    # Joint model
    joint = compute_joint(train_seqs)
    ll_joint = score_joint(test_seqs, joint, marginals)
    
    # Comparison
    diff = ll_joint - ll_marginal
    
    results.append({
        'train_month': train_month,
        'test_month': test_month,
        'll_marginal': ll_marginal,
        'll_joint': ll_joint,
        'diff': diff,
        'better': 'joint' if diff > 0 else 'marginal'
    })
    
    print(f"{train_month} → {test_month}: marginal={ll_marginal:.4f}, joint={ll_joint:.4f}, diff={diff:.4f} ({results[-1]['better']})")

# Summary
results_df = pd.DataFrame(results)
print(f"\nSummary:")
print(f"Joint better: {(results_df['diff'] > 0).sum()} / {len(results_df)}")
print(f"Mean difference: {results_df['diff'].mean():.4f}")

# Save
results_df.to_csv(RESULTS_DIR / "ar_vs_marginal_metrics.csv", index=False)

with open(RESULTS_DIR / "ar_vs_marginal_comparison.txt", 'w') as f:
    f.write("AR (Joint) vs Marginal-Only Comparison\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Joint better: {(results_df['diff'] > 0).sum()} / {len(results_df)}\n")
    f.write(f"Marginal better: {(results_df['diff'] < 0).sum()} / {len(results_df)}\n")
    f.write(f"Mean LL difference (joint - marginal): {results_df['diff'].mean():.4f}\n")
    f.write(f"Std: {results_df['diff'].std():.4f}\n\n")
    f.write("Interpretation:\n")
    f.write("- Positive diff: joint (AR) captures dependencies and helps\n")
    f.write("- Negative diff: independence assumption is sufficient, AR overhead hurts\n")
    f.write("- Near zero: positions are effectively independent\n")

print(f"\nResults saved to {RESULTS_DIR}")
