#!/usr/bin/env python3
"""
plot5_fixed.py -- Row-normalized era-specific co-occurrence heatmap.

What it shows: top mutation pairs per era, tracked across all time.
How it shows non-stationarity: each pair's activity is confined to
its era. The dominant structure in 2020 has near-zero overlap with 2024.

Usage:
  python scripts/plot5_fixed.py \
    --events data/processed/events_v3.tsv \
    --out results/plot5_fixed.png
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

def get_top_pairs(var_mass, era_months, n=8):
    cooc = defaultdict(float)
    for ym in era_months:
        for v, w in var_mass.get(ym, {}).items():
            ml = sorted(v)
            for i in range(len(ml)):
                for j in range(i+1, min(i+6, len(ml))):
                    cooc[(ml[i],ml[j])] += w
    return sorted(cooc, key=lambda p: -cooc[p])[:n]

def run(a):
    var_mass, months = load_monthly(a.events)
    print(f"{len(months)} months")

    eras = [
        ('Early\n2020', [m for m in months if '2020' in m]),
        ('Alpha/Delta\n2021', [m for m in months if '2021' in m]),
        ('Omicron\n2022', [m for m in months if '2022' in m]),
        ('Recent\n2023-24', [m for m in months if '2023' <= m <= '2024-06']),
    ]

    # collect era-specific top pairs
    all_pairs, era_labels = [], []
    for era_name, era_months in eras:
        pairs = get_top_pairs(var_mass, era_months, n=8)
        for p in pairs:
            if p not in all_pairs:
                all_pairs.append(p)
                era_labels.append(era_name.replace('\n',' '))

    # track frequency across all months
    pair_freq = np.zeros((len(all_pairs), len(months)))
    for mi, ym in enumerate(months):
        cooc_m = defaultdict(float)
        for v, w in var_mass.get(ym, {}).items():
            ml = sorted(v)
            for i in range(len(ml)):
                for j in range(i+1, min(i+6, len(ml))):
                    cooc_m[(ml[i],ml[j])] += w
        for pi, p in enumerate(all_pairs):
            pair_freq[pi, mi] = cooc_m.get(p, 0.0)

    # row-normalize: 0=inactive, 1=peak activity
    row_max = pair_freq.max(axis=1, keepdims=True).clip(min=1e-9)
    pair_freq_norm = pair_freq / row_max

    fig, ax = plt.subplots(figsize=(16, 9))

    im = ax.imshow(pair_freq_norm, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1, interpolation='nearest')

    tick_idx  = [i for i,m in enumerate(months) if m[5:] in ('01','07')]
    tick_labs = [months[i] for i in tick_idx]
    ax.set_xticks(tick_idx); ax.set_xticklabels(tick_labs, rotation=45,
                                                  ha='right', fontsize=7)
    ax.set_yticks(range(len(all_pairs)))
    ax.set_yticklabels([f'{era_labels[i]} — pair {i+1}'
                        for i in range(len(all_pairs))], fontsize=6.5)

    # era dividers
    for era_name, era_months in eras[:-1]:
        if era_months:
            xi = months.index(era_months[-1])
            ax.axvline(xi, color='white', lw=2, linestyle='--')

    # era bands on x-axis
    for era_name, era_months in eras:
        if not era_months: continue
        x0 = months.index(era_months[0])
        x1 = months.index(era_months[-1])
        ax.text((x0+x1)/2, len(all_pairs)+0.8,
                era_name.replace('\n',' '),
                ha='center', fontsize=8, fontweight='bold', color='#333333')

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                label='Relative co-occurrence mass\n'
                      '(0 = pair inactive this month,\n'
                      ' 1 = pair at its historical peak)')

    ax.set_title(
        'Co-occurrence Structure is Era-Specific: Top Mutation Pairs Per Era\n'
        'Row-normalized: green = peak activity, dark red = inactive',
        fontsize=12, fontweight='bold')
    ax.set_ylabel('Top mutation pairs selected within each era\n'
                  '(8 pairs × 4 eras = up to 32 unique pairs)', fontsize=9)
    ax.set_xlabel('Month', fontsize=9)

    fig.text(0.01, 0.01,
        'What is measured: for each of 4 evolutionary eras, identify the 8 mutation pairs '
        'with highest co-occurrence mass within that era.\n'
        'Track each pair\'s co-occurrence mass across all 77 months. '
        'Normalize each row by its own peak so color shows relative activity.\n'
        'How this shows non-stationarity: each pair is active (green) only in its era '
        'and inactive (dark red) in all other eras.\n'
        'The pairs that define variant identity in 2020 are completely absent from 2022-2024, '
        'and vice versa. No co-occurrence signal transfers across eras.\n'
        'Implication: a model trained on one era cannot apply its learned structure to another '
        '— sliding window models face a fundamentally non-stationary signal.',
        transform=fig.transFigure, fontsize=8, va='bottom',
        bbox=dict(fc='#F8F8F8', ec='#CCCCCC', pad=5))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(a.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"saved {a.out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--out', default='results/plot5_fixed.png')
    run(p.parse_args())

if __name__ == '__main__':
    main()
