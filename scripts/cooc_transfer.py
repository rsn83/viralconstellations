#!/usr/bin/env python3
"""
cooc_transfer.py -- Does co-occurrence structure transfer across time?

For every pair of months (t1, t2), compute the Spearman rank correlation
between their full co-occurrence frequency vectors over ALL mutation pairs.

If structure transfers: correlation stays high across time gaps.
If non-stationary: correlation decays rapidly with time gap.

Usage:
  python scripts/cooc_transfer.py \
    --events data/processed/events_v3.tsv \
    --out results/cooc_transfer.png \
    --top-pairs 5000
"""
import argparse, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr

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

def cooc_vector(var_mass_ym, top_pairs):
    """Co-occurrence frequency for a fixed set of pairs."""
    cooc = defaultdict(float)
    for v, w in var_mass_ym.items():
        ml = sorted(v)
        for i in range(len(ml)):
            for j in range(i+1, min(i+6, len(ml))):
                cooc[(ml[i],ml[j])] += w
    return np.array([cooc.get(p, 0.0) for p in top_pairs])

def run(a):
    var_mass, months = load_monthly(a.events)
    print(f"{len(months)} months loaded")

    # find globally top pairs across all months
    print("computing global top pairs...")
    global_cooc = defaultdict(float)
    for ym in months:
        for v, w in var_mass[ym].items():
            ml = sorted(v)
            for i in range(len(ml)):
                for j in range(i+1, min(i+6, len(ml))):
                    global_cooc[(ml[i],ml[j])] += w
    top_pairs = sorted(global_cooc, key=lambda p: -global_cooc[p])[:a.top_pairs]
    print(f"using top {len(top_pairs)} pairs")

    # compute co-occurrence vector for each month
    print("computing monthly vectors...")
    vecs = []
    for ym in months:
        vecs.append(cooc_vector(var_mass[ym], top_pairs))
    vecs = np.array(vecs)  # (n_months, n_pairs)

    # compute Spearman correlation matrix
    print("computing Spearman correlations...")
    n = len(months)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            r, _ = spearmanr(vecs[i], vecs[j])
            corr_matrix[i,j] = corr_matrix[j,i] = r if np.isfinite(r) else 0.0
        if i % 10 == 0:
            print(f"  {i}/{n}...")

    # also compute correlation vs time gap
    gaps, corrs = [], []
    for i in range(n):
        for j in range(i+1, n):
            gaps.append(j - i)
            corrs.append(corr_matrix[i,j])

    # ── plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        'Co-occurrence Structure Transfer Across Time\n'
        f'Spearman rank correlation of co-occurrence vectors '
        f'(top {a.top_pairs} mutation pairs)',
        fontsize=12, fontweight='bold')

    # Plot A: full correlation matrix
    ax = axes[0]
    tick_idx  = [i for i,m in enumerate(months) if m[5:] in ('01','07')]
    tick_labs = [months[i] for i in tick_idx]
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0,
                  aspect='auto')
    ax.set_xticks(tick_idx); ax.set_xticklabels(tick_labs, rotation=45,
                                                  ha='right', fontsize=7)
    ax.set_yticks(tick_idx); ax.set_yticklabels(tick_labs, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                label='Spearman r')
    ax.set_title('A: Month × Month Correlation Matrix\n'
                'Green diagonal = self-correlation = 1\n'
                'Off-diagonal: does structure transfer?',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Month', fontsize=9)
    ax.set_ylabel('Month', fontsize=9)
    ax.text(0.01, -0.18,
        'What is measured: for each pair of months (t1, t2), '
        'compute the Spearman rank correlation between their '
        'co-occurrence frequency vectors over all top mutation pairs.\n'
        'Green = high correlation (structure transfers). '
        'Red = near-zero (structure does not transfer).\n'
        'Non-stationarity: if structure is stable, the matrix would be '
        'uniformly green. If non-stationary, only the diagonal '
        'and nearby months would be green — which is what we expect.',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='#F8F8F8', ec='#CCCCCC', pad=4))

    # Plot B: correlation vs time gap
    ax = axes[1]
    # bin by gap
    max_gap = max(gaps)
    bins = list(range(0, max_gap+2, 1))
    bin_corr = defaultdict(list)
    for g, c in zip(gaps, corrs):
        bin_corr[g].append(c)
    gap_x = sorted(bin_corr.keys())
    gap_mean = [np.mean(bin_corr[g]) for g in gap_x]
    gap_std  = [np.std(bin_corr[g]) for g in gap_x]

    ax.plot(gap_x, gap_mean, color='#2962FF', lw=2)
    ax.fill_between(gap_x,
                   np.array(gap_mean)-np.array(gap_std),
                   np.array(gap_mean)+np.array(gap_std),
                   alpha=0.2, color='#2962FF')
    ax.axhline(0, color='red', lw=1, linestyle='--', alpha=0.7,
              label='Zero correlation (no transfer)')
    ax.axhline(0.5, color='green', lw=1, linestyle='--', alpha=0.5,
              label='r=0.5 (moderate transfer)')
    ax.set_xlabel('Time gap (months)', fontsize=10)
    ax.set_ylabel('Spearman r (mean ± std)', fontsize=10)
    ax.set_title('B: Correlation Decay With Time Gap\n'
                'How fast does co-occurrence structure become unrecognizable?',
                fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 1.1)
    ax.text(0.01, -0.22,
        'What is measured: for each pair of months separated by gap g, '
        'compute Spearman r. Plot mean ± std across all pairs with that gap.\n'
        'How this shows non-stationarity: if the process were stationary, '
        'correlation would remain high even at large gaps.\n'
        'If correlation drops to ~0 within 6-12 months, the co-occurrence '
        'structure is completely replaced on that timescale — '
        'no sliding window model can learn a transferable signal.',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='#F8F8F8', ec='#CCCCCC', pad=4))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(a.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"saved {a.out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--out', default='results/cooc_transfer.png')
    p.add_argument('--top-pairs', type=int, default=5000, dest='top_pairs')
    run(p.parse_args())

if __name__ == '__main__':
    main()
