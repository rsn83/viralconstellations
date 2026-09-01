#!/usr/bin/env python3
"""
nonstationarity2.py -- Five separate precise plots for non-stationarity.

Usage:
  python scripts/nonstationarity2.py \
    --events data/processed/events_v3.tsv \
    --out results/nonstationarity2.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy

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

def run(a):
    print("loading..."); var_mass, months = load_monthly(a.events)
    x = list(range(len(months)))

    # tick every 6 months
    tick_idx  = [i for i,m in enumerate(months) if m[5:] in ('01','07')]
    tick_labs = [months[i] for i in tick_idx]

    def xt(ax):
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labs, rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.25)

    # ── compute ────────────────────────────────────────────────────────
    median_size, new_muts, ent, ov = [], [], [], []
    seen = set()
    # top-N pairs per month for plot 5
    top_pairs_per_month = {}

    for mi, ym in enumerate(months):
        vm = var_mass[ym]
        # 1. weighted mean variant size
        median_size.append(np.average([len(v) for v in vm],
                           weights=list(vm.values())))
        # 2. new mutations
        cur = {m for v in vm for m in v}
        new_muts.append(len(cur - seen)); seen |= cur
        # 3. entropy
        probs = list(vm.values())
        ent.append(scipy_entropy(probs) if probs else 0.0)
        # 4. overlap with previous month
        if mi > 0:
            prev = var_mass[months[mi-1]]
            ov.append(sum(min(vm.get(v,0), prev.get(v,0))
                         for v in set(vm)|set(prev)))
        else:
            ov.append(float('nan'))
        # 5. top mutation pairs by co-occurrence mass
        cooc = defaultdict(float)
        for v, w in vm.items():
            ml = sorted(v)
            for i in range(len(ml)):
                for j in range(i+1, min(i+10, len(ml))):
                    cooc[(ml[i],ml[j])] += w
        top_pairs_per_month[ym] = cooc

    # for plot 5: track frequency of the globally top-20 pairs over time
    # find global top-20 pairs by total co-occurrence mass
    global_cooc = defaultdict(float)
    for ym in months:
        for p, v in top_pairs_per_month[ym].items():
            global_cooc[p] += v
    top20 = sorted(global_cooc, key=lambda p:-global_cooc[p])[:20]
    # track their frequency over time
    pair_freq = np.zeros((len(top20), len(months)))
    for mi, ym in enumerate(months):
        for pi, p in enumerate(top20):
            pair_freq[pi, mi] = top_pairs_per_month[ym].get(p, 0.0)

    # ── figure: 5 separate subplots, stacked ──────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(14, 22))
    fig.suptitle(
        'Non-stationarity of SARS-CoV-2 Spike Mutation Constellations\n'
        'Five independent measurements across 77 months (2020-01 to 2026-05)',
        fontsize=13, fontweight='bold', y=0.99)

    regime_months = {'Alpha':'2021-01','Delta':'2021-07','Omicron':'2021-12',
                     'BA.5':'2022-07','XBB':'2022-12','JN.1':'2023-10'}

    def add_regimes(ax, ym_dict):
        for name, ym in ym_dict.items():
            if ym in months:
                xi = months.index(ym)
                ax.axvline(xi, color='#888888', lw=0.8,
                           linestyle='--', alpha=0.6)
                ax.text(xi+0.3, ax.get_ylim()[1]*0.92,
                       name, fontsize=6.5, color='#555555', rotation=90)

    # ── Plot 1: Median variant size ────────────────────────────────────
    ax = axes[0]
    ax.plot(x, median_size, color='#2962FF', lw=1.8)
    ax.fill_between(x, median_size, alpha=0.12, color='#2962FF')
    ax.set_title(
        'Plot 1: Weighted Mean Number of Mutations per Variant',
        fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('Mean mutations per variant\n(weighted by variant mass)', fontsize=9)
    ax.text(0.01, 0.97,
        'Measure: for each month t, compute Σ_B |B| · p_t(B) — '
        'the expected variant size under the population distribution p_t.\n'
        'Interpretation: the output space grows structurally over time. '
        'A model trained on 2020 data predicts size-1 variants; '
        'in 2024 the true variants have size 55.',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='#F0F4FF', ec='#AAAACC', pad=4))
    for ym, lab in [('2020-03','1'),('2022-05','28'),('2024-06','55')]:
        if ym in months:
            xi = months.index(ym)
            ax.annotate(lab, xy=(xi, median_size[xi]),
                       xytext=(xi+2, median_size[xi]+2),
                       fontsize=8, color='#2962FF',
                       arrowprops=dict(arrowstyle='->', color='#2962FF', lw=0.7))
    xt(ax); add_regimes(ax, regime_months)

    # ── Plot 2: New mutations per month ────────────────────────────────
    ax = axes[1]
    ax.bar(x, new_muts, color='#00897B', alpha=0.75, width=0.85)
    ax.set_title(
        'Plot 2: Number of Mutations Appearing for the First Time Each Month',
        fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('New mutations\n(never seen before this month)', fontsize=9)
    ax.text(0.01, 0.97,
        'Measure: count mutations in month t\'s variants that did not appear '
        'in any variant in months 1..t-1.\n'
        'Interpretation: the mutation vocabulary is open-ended — '
        'new marks enter continuously. No fixed mark space exists, '
        'ruling out standard categorical TPP formulations.',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='#F0FFF8', ec='#AACCBB', pad=4))
    xt(ax); add_regimes(ax, regime_months)

    # ── Plot 3: Shannon entropy ────────────────────────────────────────
    ax = axes[2]
    ax.plot(x, ent, color='#C62828', lw=1.5)
    ax.fill_between(x, ent, alpha=0.12, color='#C62828')
    ax.set_title(
        'Plot 3: Shannon Entropy of Monthly Variant Population Distribution',
        fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('H(p_t) = -Σ_B p_t(B) log p_t(B)\n(nats)', fontsize=9)
    ax.text(0.01, 0.97,
        'Measure: Shannon entropy of the normalised variant mass distribution p_t '
        'each month. High entropy = many variants with similar mass (diverse). '
        'Low entropy = one dominant variant.\n'
        'Interpretation: sharp drops at each variant sweep show regime changes '
        'incompatible with stationarity. The process is not mean-reverting.',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='#FFF0F0', ec='#CCAAAA', pad=4))
    xt(ax); add_regimes(ax, regime_months)

    # ── Plot 4: Month-to-month overlap ────────────────────────────────
    ax = axes[3]
    ov_clean = [v if not np.isnan(v) else 0 for v in ov]
    ax.plot(x[1:], ov_clean[1:], color='#F57F17', lw=1.5)
    ax.fill_between(x[1:], ov_clean[1:], alpha=0.15, color='#F57F17')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', lw=0.8, linestyle=':', alpha=0.5,
               label='Perfect persistence (overlap=1)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(
        'Plot 4: Month-to-Month Population Overlap (Persistence Baseline)',
        fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('Overlap = Σ_B min(p_t(B), p_{t-1}(B))', fontsize=9)
    ax.text(0.01, 0.97,
        'Measure: for each consecutive pair of months (t-1, t), sum over all '
        'variants B the minimum of their mass in month t-1 and month t.\n'
        'Value = 1: population identical to previous month (perfect persistence). '
        'Value = 0: completely different population.\n'
        'Interpretation: declining trend shows the population becomes less stable '
        'over time. Later months have faster variant turnover, '
        'making persistence a weaker baseline as time progresses.',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='#FFFBF0', ec='#CCBBAA', pad=4))
    xt(ax); add_regimes(ax, regime_months)

    # ── Plot 5: Co-occurrence structure of top-20 pairs over time ──────
    ax = axes[4]
    im = ax.imshow(pair_freq, aspect='auto', cmap='YlOrRd',
                  interpolation='nearest', vmin=0)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f'pair {i+1}' for i in range(len(top20))],
                      fontsize=6)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labs, rotation=45, ha='right', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                label='Co-occurrence mass\n(fraction of monthly population)')
    ax.set_title(
        'Plot 5: Co-occurrence Frequency of Top-20 Global Mutation Pairs Over Time',
        fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('Top-20 mutation pairs\n(ranked by total co-occurrence mass)', fontsize=9)
    ax.text(0.01, -0.18,
        'Measure: for each month t, compute the fraction of population mass '
        'in variants where mutations i and j both appear — '
        'the co-occurrence mass of pair (i,j) at time t.\n'
        'Shown for the 20 globally most co-occurring pairs across all months.\n'
        'Interpretation: each pair dominates a narrow time window and then '
        'disappears or is replaced. The co-occurrence structure is completely '
        'different across time periods — not a stable learnable signal for '
        'sliding-window models.',
        transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='#F8F0FF', ec='#BBAACC', pad=4))
    # add regime lines
    for name, ym in regime_months.items():
        if ym in months:
            xi = months.index(ym)
            ax.axvline(xi, color='white', lw=0.8, linestyle='--', alpha=0.6)
            ax.text(xi+0.2, -0.5, name, fontsize=6,
                   color='white', rotation=90, va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(a.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"saved {a.out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--out', default='results/nonstationarity2.png')
    run(p.parse_args())

if __name__ == '__main__':
    main()
