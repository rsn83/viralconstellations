"""
Test: Chow-Liu tree (AR/joint) vs marginal on NOVEL haplotypes.

Novel = appears at t+1 but not at t.
Does position structure help forecast new combinations?
"""

import pandas as pd
from collections import Counter
import numpy as np
from pathlib import Path
import itertools

DATA_FILE = Path("data/processed/spike_haplotypes_monthly.csv")
print("Loading preprocessed haplotypes...", flush=True)

df = pd.read_csv(DATA_FILE)
months = sorted(df['month'].unique())
print(f"  Months: {months[0]} to {months[-1]} ({len(months)} months)\n", flush=True)

def parse_haplotype(hap_str):
    """Convert 'pos:residue;pos:residue;...' to dict"""
    result = {}
    for part in hap_str.split(';'):
        pos, res = part.split(':')
        result[int(pos)] = res
    return result

def compute_mutual_information(col1, col2):
    """MI between two binary columns (residue present or not)"""
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

def fit_chow_liu_tree(positions_list, haplotypes_data):
    """Fit Chow-Liu tree: MST of positions by mutual information"""
    n_pos = len(positions_list)
    
    # Compute MI for all pairs
    mi_matrix = np.zeros((n_pos, n_pos))
    for i, j in itertools.combinations(range(n_pos), 2):
        pos_i, pos_j = positions_list[i], positions_list[j]
        
        col_i = []
        col_j = []
        for hap_dict, count in haplotypes_data:
            for _ in range(count):
                col_i.append(1 if hap_dict.get(pos_i) != 'wt' else 0)
                col_j.append(1 if hap_dict.get(pos_j) != 'wt' else 0)
        
        mi = compute_mutual_information(col_i, col_j)
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi
    
    # Build MST (Chow-Liu): parent of each node
    parents = [-1] * n_pos
    parents[0] = -1
    
    visited = {0}
    edges = []
    
    for _ in range(n_pos - 1):
        best_mi = -np.inf
        best_i, best_j = None, None
        
        for i in visited:
            for j in range(n_pos):
                if j not in visited and mi_matrix[i, j] > best_mi:
                    best_mi = mi_matrix[i, j]
                    best_i, best_j = i, j
        
        if best_j is not None:
            visited.add(best_j)
            parents[best_j] = best_i
    
    return parents

def score_under_tree(haplotype_dict, positions_list, parents, train_marginals, train_conditionals):
    """Score haplotype under tree: p(root) * product p(child | parent)"""
    if not parents:
        return None
    
    score = 1.0
    
    for i, pos in enumerate(positions_list):
        res = haplotype_dict.get(pos, 'wt')
        
        if parents[i] == -1:
            # Root: use marginal
            score *= train_marginals[pos].get(res, 1e-9)
        else:
            # Child: use conditional
            parent_pos = positions_list[parents[i]]
            parent_res = haplotype_dict.get(parent_pos, 'wt')
            
            if (parent_pos, parent_res, pos) in train_conditionals:
                cond_dist = train_conditionals[(parent_pos, parent_res, pos)]
                score *= cond_dist.get(res, 1e-9)
            else:
                # Fallback to marginal
                score *= train_marginals[pos].get(res, 1e-9)
    
    return score

results = []

for i in range(len(months) - 1):
    train_month = months[i]
    test_month = months[i + 1]
    
    train_data = df[df['month'] == train_month]
    test_data = df[df['month'] == test_month]
    
    if len(train_data) < 2 or len(test_data) < 2:
        continue
    
    # Identify novel haplotypes
    train_haplotypes = set(train_data['haplotype'])
    test_haplotypes = test_data[~test_data['haplotype'].isin(train_haplotypes)]
    
    if len(test_haplotypes) == 0:
        continue
    
    print(f"{train_month} → {test_month}", flush=True)
    print(f"  Novel haplotypes: {len(test_haplotypes)}", flush=True)
    
    # Extract positions
    positions = set()
    for hap in train_data['haplotype']:
        for part in hap.split(';'):
            positions.add(int(part.split(':')[0]))
    positions_list = sorted(positions)
    
    # Parse training haplotypes
    train_haps_parsed = []
    for _, row in train_data.iterrows():
        hap_dict = parse_haplotype(row['haplotype'])
        train_haps_parsed.append((hap_dict, row['count']))
    
    # Compute marginals
    train_marginals = {}
    for pos in positions_list:
        counts = Counter()
        for hap_dict, count in train_haps_parsed:
            res = hap_dict.get(pos, 'wt')
            counts[res] += count
        total = sum(counts.values())
        train_marginals[pos] = {res: c / total for res, c in counts.items()}
    
    # Fit Chow-Liu tree
    parents = fit_chow_liu_tree(positions_list, train_haps_parsed)
    
    # Compute conditionals for tree
    train_conditionals = {}
    for pos_j, parent_i in enumerate(parents):
        if parent_i >= 0:
            pos_i = positions_list[parent_i]
            pos_j_val = positions_list[pos_j]
            
            for parent_res in ['wt', 'A', 'C', 'G', 'T', '*']:  # Common residues
                counts = Counter()
                for hap_dict, count in train_haps_parsed:
                    if hap_dict.get(pos_i, 'wt') == parent_res:
                        res_j = hap_dict.get(pos_j_val, 'wt')
                        counts[res_j] += count
                
                if sum(counts.values()) > 0:
                    total = sum(counts.values())
                    train_conditionals[(pos_i, parent_res, pos_j_val)] = {res: c / total for res, c in counts.items()}
    
    # Score novel haplotypes under both models
    ll_marginal = 0.0
    ll_tree = 0.0
    total_novel = test_haplotypes['count'].sum()
    
    for _, row in test_haplotypes.iterrows():
        hap_dict = parse_haplotype(row['haplotype'])
        count = row['count']
        
        # Marginal
        prob_marginal = 1.0
        for pos in positions_list:
            res = hap_dict.get(pos, 'wt')
            prob_marginal *= train_marginals[pos].get(res, 1e-9)
        
        if prob_marginal > 0:
            ll_marginal += count * np.log2(prob_marginal)
        
        # Tree
        prob_tree = score_under_tree(hap_dict, positions_list, parents, train_marginals, train_conditionals)
        if prob_tree and prob_tree > 0:
            ll_tree += count * np.log2(prob_tree)
    
    ll_marginal /= total_novel
    ll_tree /= total_novel
    diff = ll_tree - ll_marginal
    better = "tree" if diff > 0 else "marginal"
    
    print(f"  Marginal LL: {ll_marginal:.6f}", flush=True)
    print(f"  Tree LL:     {ll_tree:.6f}", flush=True)
    print(f"  Diff: {diff:.6f} ({better})\n", flush=True)
    
    results.append({
        'train_month': train_month,
        'test_month': test_month,
        'll_marginal': ll_marginal,
        'll_tree': ll_tree,
        'diff': diff,
        'better': better
    })

# Summary
print("="*60, flush=True)
print("SUMMARY: Tree (AR) vs Marginal on NOVEL haplotypes", flush=True)
print("="*60, flush=True)

if results:
    results_df = pd.DataFrame(results)
    tree_wins = (results_df['diff'] > 0).sum()
    total = len(results_df)
    
    print(f"Tree better: {tree_wins}/{total}", flush=True)
    print(f"Mean difference: {results_df['diff'].mean():.6f}", flush=True)
    print(f"Std: {results_df['diff'].std():.6f}\n", flush=True)
    
    if tree_wins > total / 2:
        print("→ Position structure (tree/AR) helps forecast novel haplotypes.", flush=True)
    else:
        print("→ Position independence is sufficient. Tree doesn't add value.", flush=True)
else:
    print("No novel haplotypes found.", flush=True)
