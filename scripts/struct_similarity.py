#!/usr/bin/env python3
"""
struct_similarity.py -- Structural similarity of co-occurrence matrices.

For each pair of months (t1, t2), compute the cosine similarity between
their full co-occurrence matrices treated as sparse vectors:

  sim(t1, t2) = <C_t1, C_t2>_F / (||C_t1||_F * ||C_t2||_F)

where C_t[i,j] = fraction of population mass in variants containing
both mutations i and j.

Unlike the Spearman analysis which only looks at shared active pairs,
this uses ALL pairs from BOTH months -- pairs absent in one month
contribute zero, penalizing structural dissimilarity.

Usage:
  python scripts/struct_similarity.py \
    --events data/processed/events_v3.tsv \
    --out results/struct_similarity.png
"""
import argparse, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

def load_monthly(path):
    by_month = defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit(): continue
            date, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            s = frozenset(int(x) for x in muts.split(',') if x)
            if s: by_month[date[:7]][s] += cnt
    months = sorted(by_month.keys())
    var_mass = {}
    for ym in months:
        tot = sum(by_month[ym].values()) or 1.0
        var_mass[ym] = {s: v/tot for s, v in by_month[ym].items()}
    return var_mass, months

def cooc_sparse(var_mass_ym):
    """Sparse co-occurrence vector: dict of (i,j) -> frequency, i<j."""
    cooc = defaultdict(float)
    for v, w in var_mass_ym.items():
        ml = sorted(v)
        for i in range(len(ml)):
            for j in range(i+1, min(i+8, len(ml))):
                cooc[(ml[i], ml[j])] += w
    # normalize to unit Frobenius norm
    norm = np.sqrt(sum(v**2 for v in cooc.values())) or 1.0
    return {k: v/norm for k, v in cooc.items()}

def cosine_sim(c1, c2):
    """Cosine similarity between two sparse co-occurrence vectors.
    Since both are already unit-normalized, this is just their dot product.
    Pairs absent in one month contribute 0 -- penalizes structural difference."""
    # dot product over shared keys only (absent pairs contribute 0)
    shared = set(c1.keys()) & set(c2.keys())
    return float(sum(c1[k] * c2[k] for k in shared))

def run(a):
    var_mass, months = load_monthly(a.events)
    n = len(months)
    print(f"{n} months")

    # subsample for tractability
    if n > 40:
        step = max(1, n // 40)
        sample = list(range(0, n, step))
        if sample[-1] != n-1: sample.append(n-1)
    else:
        sample = list(range(n))
    sample_months = [months[i] for i in sample]
    ns = len(sample_months)
    print(f"using {ns} sampled months")

    # precompute normalized co-occurrence vectors
    print("computing co-occurrence vectors...")
    coocs = {}
    for ym in sample_months:
        coocs[ym] = cooc_sparse(var_mass[ym])

    # compute cosine similarity matrix
    print("computing structural similarities...")
    sim = np.zeros((ns, ns))
    for i in range(ns):
        sim[i, i] = 1.0
        for j in range(i+1, ns):
            s = cosine_sim(coocs[sample_months[i]],
                          coocs[sample_months[j]])
            sim[i, j] = sim[j, i] = s
        if i % 5 == 0:
            print(f"  {i}/{ns}...")

    # decay curve
    gaps, vals = [], []
    for i in range(ns):
        for j in range(i+1, ns):
            gaps.append(j - i)
            vals.append(sim[i, j])
    gaps, vals = np.array(gaps), np.array(vals)
    max_gap = int(gaps.max())
    gap_x, gap_mean, gap_std = [], [], []
    for g in range(1, max_gap+1):
        v = vals[gaps==g]
        if len(v) >= 2:
            gap_x.append(g)
            gap_mean.append(v.mean())
            gap_std.append(v.std())

    # ── plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # left: similarity matrix
    ax = axes[0]
    tick_step = max(1, ns // 8)
    tick_idx  = list(range(0, ns, tick_step))
    tick_labs = [sample_months[i] for i in tick_idx]

    im = ax.imshow(sim, cmap='RdYlGn', vmin=0, vmax=1,
                  aspect='auto', interpolation='nearest')
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labs, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labs, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                label='Cosine similarity\n(0 = completely different structure,\n 1 = identical structure)')
    ax.set_title('Structural Similarity of Co-occurrence Matrices\nBetween All Pairs of Months',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Month', fontsize=10)

    # right: decay
    ax = axes[1]
    gx = np.array(gap_x)
    gm = np.array(gap_mean)
    gs = np.array(gap_std)
    ax.plot(gx, gm, color='#2962FF', lw=2)
    ax.fill_between(gx, (gm-gs).clip(0), gm+gs, alpha=0.2, color='#2962FF')
    ax.axhline(0, color='red', lw=1, linestyle='--', alpha=0.7,
              label='Similarity = 0 (no structure overlap)')
    ax.axhline(0.5, color='green', lw=1, linestyle='--', alpha=0.5,
              label='Similarity = 0.5 (moderate overlap)')
    # find where drops below 0.1
    cross = next((gx[i] for i in range(len(gx)) if gm[i] < 0.1), None)
    if cross:
        ax.axvline(cross, color='orange', lw=1.5, linestyle=':',
                  label=f'Similarity < 0.1 at gap = {cross:.0f} months')
    ax.set_xlabel('Time gap between months (months)', fontsize=10)
    ax.set_ylabel('Cosine similarity between\nco-occurrence matrices (mean ± std)',
                 fontsize=10)
    ax.set_title('How Fast Does Co-occurrence Structure\nBecome Unrecognizable?',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    fig.suptitle(
        'Structural Similarity of Co-occurrence Matrices Across Time\n'
        'Cosine similarity between full sparse co-occurrence matrices '
        '(absent pairs contribute 0 — penalises structural difference)',
        fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(a.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"saved {a.out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--out', default='results/struct_similarity.png')
    run(p.parse_args())

if __name__ == '__main__':
    main()
