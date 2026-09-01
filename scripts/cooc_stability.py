#!/usr/bin/env python3
"""
cooc_stability.py -- Is co-occurrence structure stable across time?

For each pair of months (t1, t2), compute Spearman rank correlation
between their co-occurrence matrices restricted to mutations active
in BOTH months.

If structure is stable: correlation stays high even at large time gaps.
If non-stationary: correlation decays rapidly -- anything learned at
time t1 is useless for predicting at t2.

Usage:
  python scripts/cooc_stability.py \
    --events data/processed/events_v3.tsv \
    --out results/cooc_stability.png
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

def cooc_dict(var_mass_ym):
    """Pairwise co-occurrence mass for all mutation pairs in a month."""
    cooc = defaultdict(float)
    for v, w in var_mass_ym.items():
        ml = sorted(v)
        for i in range(len(ml)):
            for j in range(i+1, min(i+8, len(ml))):
                cooc[(ml[i], ml[j])] += w
    return cooc

def spearman_shared(cooc1, cooc2):
    """Spearman correlation restricted to pairs active in BOTH months."""
    shared = set(cooc1.keys()) & set(cooc2.keys())
    if len(shared) < 10:
        return float('nan')
    v1 = np.array([cooc1[p] for p in shared])
    v2 = np.array([cooc2[p] for p in shared])
    r, _ = spearmanr(v1, v2)
    return float(r) if np.isfinite(r) else float('nan')

def run(a):
    var_mass, months = load_monthly(a.events)
    n = len(months)
    print(f"{n} months")

    # subsample months for tractability if too many
    if n > 40:
        step = max(1, n // 40)
        sample = list(range(0, n, step))
        if sample[-1] != n-1: sample.append(n-1)
    else:
        sample = list(range(n))
    sample_months = [months[i] for i in sample]
    ns = len(sample_months)
    print(f"using {ns} sampled months")

    # precompute co-occurrence dicts
    print("computing co-occurrence matrices...")
    coocs = {}
    for ym in sample_months:
        coocs[ym] = cooc_dict(var_mass[ym])

    # compute Spearman correlation matrix
    print("computing correlations...")
    corr = np.full((ns, ns), float('nan'))
    for i in range(ns):
        corr[i, i] = 1.0
        for j in range(i+1, ns):
            r = spearman_shared(coocs[sample_months[i]],
                               coocs[sample_months[j]])
            corr[i, j] = corr[j, i] = r
        if i % 5 == 0:
            print(f"  {i}/{ns}...")

    # correlation vs time gap
    gaps, vals = [], []
    for i in range(ns):
        for j in range(i+1, ns):
            if not np.isnan(corr[i,j]):
                gaps.append(j - i)
                vals.append(corr[i,j])
    gaps, vals = np.array(gaps), np.array(vals)
    # bin means
    max_gap = int(gaps.max())
    gap_mean, gap_std, gap_x = [], [], []
    for g in range(1, max_gap+1):
        v = vals[gaps==g]
        if len(v) >= 2:
            gap_mean.append(v.mean())
            gap_std.append(v.std())
            gap_x.append(g)

    # ── plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # left: full correlation matrix
    ax = axes[0]
    tick_step = max(1, ns // 8)
    tick_idx  = list(range(0, ns, tick_step))
    tick_labs = [sample_months[i] for i in tick_idx]

    im = ax.imshow(corr, cmap='RdYlGn', vmin=-0.2, vmax=1.0,
                  aspect='auto', interpolation='nearest')
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labs, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labs, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                label='Spearman r')
    ax.set_title('Co-occurrence Structure Correlation\nBetween All Pairs of Months',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Month', fontsize=10)

    # right: correlation decay
    ax = axes[1]
    gx = np.array(gap_x)
    gm = np.array(gap_mean)
    gs = np.array(gap_std)
    ax.plot(gx, gm, color='#2962FF', lw=2)
    ax.fill_between(gx, gm-gs, gm+gs, alpha=0.2, color='#2962FF')
    ax.axhline(0, color='red', lw=1, linestyle='--', alpha=0.7,
              label='r = 0 (no structure transfer)')
    ax.axhline(0.5, color='green', lw=1, linestyle='--', alpha=0.5,
              label='r = 0.5 (moderate transfer)')
    # mark where correlation crosses 0
    cross = next((gx[i] for i in range(len(gx)) if gm[i] < 0.1), None)
    if cross:
        ax.axvline(cross, color='orange', lw=1.5, linestyle=':',
                  label=f'r < 0.1 at gap = {cross:.0f} months')
    ax.set_xlabel('Time gap between months (months)', fontsize=10)
    ax.set_ylabel('Spearman r between co-occurrence matrices\n(restricted to shared active mutations)',
                 fontsize=10)
    ax.set_title('How Fast Does Co-occurrence Structure\nBecome Unrecognizable?',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 1.1)

    fig.suptitle(
        'Co-occurrence Structure Stability Across Time\n'
        'Spearman rank correlation between monthly co-occurrence matrices '
        '(pairs active in both months only)',
        fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(a.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"saved {a.out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--out', default='results/cooc_stability.png')
    run(p.parse_args())

if __name__ == '__main__':
    main()
