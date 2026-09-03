"""
Test: does learning joint distribution (AR) beat marginal-only?

Load preprocessed haplotypes, train on month t, score month t+1 under both models.
"""

import pandas as pd
from collections import Counter
import numpy as np
from pathlib import Path

DATA_FILE = Path("data/processed/spike_haplotypes_monthly.csv")
print("Loading preprocessed haplotypes...", flush=True)

df = pd.read_csv(DATA_FILE)
months = sorted(df['month'].unique())
print(f"  Months: {months[0]} to {months[-1]} ({len(months)} months)", flush=True)

results = []

for i in range(len(months) - 1):
    train_month = months[i]
    test_month = months[i + 1]
    
    train_data = df[df['month'] == train_month]
    test_data = df[df['month'] == test_month]
    
    if len(train_data) < 2 or len(test_data) < 2:
        continue
    
    print(f"\n{train_month} → {test_month}", flush=True)
    print(f"  Train: {train_data['count'].sum():,} sequences, {len(train_data)} haplotypes", flush=True)
    print(f"  Test: {test_data['count'].sum():,} sequences, {len(test_data)} haplotypes", flush=True)
    
    # Parse haplotypes into per-position frequencies
    def parse_haplotype(hap_str, top_52):
        """Convert 'pos:residue;pos:residue;...' to dict"""
        result = {}
        for part in hap_str.split(';'):
            pos, res = part.split(':')
            result[int(pos)] = res
        return result
    
    # Extract positions
    positions = set()
    for hap in train_data['haplotype']:
        for part in hap.split(';'):
            positions.add(int(part.split(':')[0]))
    positions = sorted(positions)
    
    # Train: compute marginals and joint
    marginals = {}
    joint = {}
    total_sequences = train_data['count'].sum()
    
    for _, row in train_data.iterrows():
        hap_dict = parse_haplotype(row['haplotype'], positions)
        count = row['count']
        
        # Update marginals
        for pos in positions:
            if pos not in marginals:
                marginals[pos] = Counter()
            res = hap_dict.get(pos, 'wt')
            marginals[pos][res] += count
        
        # Update joint
        joint[row['haplotype']] = joint.get(row['haplotype'], 0) + count
    
    # Normalize marginals
    for pos in marginals:
        total = sum(marginals[pos].values())
        marginals[pos] = {res: count / total for res, count in marginals[pos].items()}
    
    # Normalize joint
    joint = {hap: count / total_sequences for hap, count in joint.items()}
    
    # Test: score under both models
    ll_marginal = 0.0
    ll_joint = 0.0
    test_sequences = test_data['count'].sum()
    
    for _, row in test_data.iterrows():
        hap = row['haplotype']
        count = row['count']
        
        # Marginal score
        hap_dict = parse_haplotype(hap, positions)
        prob_marginal = 1.0
        for pos in positions:
            res = hap_dict.get(pos, 'wt')
            prob_marginal *= marginals[pos].get(res, 1e-9)
        
        if prob_marginal > 0:
            ll_marginal += count * np.log2(prob_marginal)
        
        # Joint score
        if hap in joint:
            ll_joint += count * np.log2(joint[hap])
        else:
            # Unseen haplotype: use marginal as fallback
            ll_joint += count * np.log2(prob_marginal)
    
    # Normalize
    ll_marginal /= test_sequences
    ll_joint /= test_sequences
    
    diff = ll_joint - ll_marginal
    better = "joint" if diff > 0 else "marginal"
    
    print(f"  Marginal LL: {ll_marginal:.6f}")
    print(f"  Joint LL:    {ll_joint:.6f}")
    print(f"  Diff (joint - marginal): {diff:.6f} ({better})", flush=True)
    
    results.append({
        'train_month': train_month,
        'test_month': test_month,
        'll_marginal': ll_marginal,
        'll_joint': ll_joint,
        'diff': diff,
        'better': better
    })

# Summary
print(f"\n" + "="*60, flush=True)
print("SUMMARY", flush=True)
print("="*60, flush=True)

results_df = pd.DataFrame(results)
joint_wins = (results_df['diff'] > 0).sum()
total = len(results_df)

print(f"Joint better: {joint_wins}/{total}", flush=True)
print(f"Mean difference: {results_df['diff'].mean():.6f}", flush=True)
print(f"Std: {results_df['diff'].std():.6f}", flush=True)

if joint_wins > total / 2:
    print("\n→ Conclusion: Joint (AR) helps. Position dependencies matter.", flush=True)
else:
    print("\n→ Conclusion: Marginal-only is competitive. Positions are effectively independent.", flush=True)

