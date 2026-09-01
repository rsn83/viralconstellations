#!/usr/bin/env python3
"""
five_plots.py -- Five separate publication-quality plots.

Plots 1-4: Non-stationarity of SARS-CoV-2 spike mutation constellations.
Plot 5:    Co-occurrence predictability -- do pairs in future variants
           already co-occur before those variants appear?

Usage:
  python scripts/five_plots.py \
    --events data/processed/events_v3.tsv \
    --out-dir results/
"""
import argparse, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy

BLUE   = '#2962FF'
TEAL   = '#00897B'
RED    = '#C62828'
AMBER  = '#F57F17'
DARK   = '#1A1A2E'

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

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"saved {path}")

def xticks(ax, months):
    x = list(range(len(months)))
    tick_idx  = [i for i,m in enumerate(months) if m[5:] in ('01','07')]
    tick_labs = [months[i] for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labs, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return x

def regime_lines(ax, months):
    regimes = {'Alpha':'2021-01','Delta':'2021-07',
               'Omicron':'2021-12','BA.5':'2022-07',
               'XBB':'2022-12','JN.1':'2023-10'}
    ylo, yhi = ax.get_ylim()
    for name, ym in regimes.items():
        if ym in months:
            xi = months.index(ym)
            ax.axvline(xi, color='#999999', lw=0.8, linestyle='--', alpha=0.6)
            ax.text(xi+0.3, yhi*0.93, name,
                   fontsize=6.5, color='#666666', rotation=90, va='top')

def run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    print("loading data...")
    var_mass, months = load_monthly(a.events)
    x = list(range(len(months)))
    print(f"{len(months)} months")

    # ── pre-compute all metrics ────────────────────────────────────────
    mean_size, new_muts, ent, ov = [], [], [], []
    seen = set()
    for mi, ym in enumerate(months):
        vm = var_mass[ym]
        # 1
        mean_size.append(np.average([len(v) for v in vm],
                         weights=list(vm.values())) if vm else 0)
        # 2
        cur = {m for v in vm for m in v}
        new_muts.append(len(cur - seen)); seen |= cur
        # 3
        probs = list(vm.values())
        ent.append(scipy_entropy(probs) if probs else 0.0)
        # 4
        if mi > 0:
            prev = var_mass[months[mi-1]]
            ov.append(sum(min(vm.get(v,0), prev.get(v,0))
                         for v in set(vm)|set(prev)))
        else:
            ov.append(float('nan'))

    # ── PLOT 1: variant size ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, mean_size, color=BLUE, lw=2)
    ax.fill_between(x, mean_size, alpha=0.12, color=BLUE)
    for ym, lab in [('2020-03','1'),('2022-01','28'),('2024-06','55')]:
        if ym in months:
            xi = months.index(ym)
            ax.annotate(lab, xy=(xi, mean_size[xi]),
                       xytext=(xi+2, mean_size[xi]+2),
                       fontsize=9, color=BLUE,
                       arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
    xticks(ax, months)
    regime_lines(ax, months)
    ax.set_title('Weighted Mean Number of Mutations per Variant Over Time',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Mean mutations per variant\n(weighted by variant frequency)', fontsize=10)
    fig.subplots_adjust(left=0.12)
    savefig(fig, os.path.join(a.out_dir, 'plot1_variant_size.png'))

    # ── PLOT 2: new mutations ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, new_muts, color=TEAL, alpha=0.8, width=0.85)
    xticks(ax, months)
    regime_lines(ax, months)
    ax.set_title('Number of Mutations Appearing for the First Time Each Month',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('New mutations\n(absent from all prior months)', fontsize=10)
    fig.subplots_adjust(left=0.12)
    savefig(fig, os.path.join(a.out_dir, 'plot2_new_mutations.png'))

    # ── PLOT 3: entropy ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, ent, color=RED, lw=1.8)
    ax.fill_between(x, ent, alpha=0.12, color=RED)
    xticks(ax, months)
    regime_lines(ax, months)
    ax.set_title('Shannon Entropy of Monthly Variant Population Distribution',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('H(p_t) = −Σ p_t(B) log p_t(B)  [nats]', fontsize=10)
    fig.subplots_adjust(left=0.12)
    savefig(fig, os.path.join(a.out_dir, 'plot3_entropy.png'))

    # ── PLOT 4: month-to-month overlap ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ov_c = [v if not np.isnan(v) else 0 for v in ov]
    ax.plot(x[1:], ov_c[1:], color=AMBER, lw=1.8)
    ax.fill_between(x[1:], ov_c[1:], alpha=0.15, color=AMBER)
    ax.axhline(1.0, color='gray', lw=0.8, linestyle=':',
              label='Perfect persistence (overlap = 1)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='upper right')
    xticks(ax, months)
    regime_lines(ax, months)
    ax.set_title('Month-to-Month Population Overlap',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Σ_B min(p_t(B), p_{t−1}(B))\n[1 = identical to previous month,  0 = completely different]',
                 fontsize=10)
    fig.subplots_adjust(left=0.14)
    savefig(fig, os.path.join(a.out_dir, 'plot4_overlap.png'))

    # ── PLOT 5: co-occurrence predictability ──────────────────────────
    # For each month t and horizon h, what fraction of mutation pairs
    # in NEW variants at t+h were already co-occurring at t?
    print("computing co-occurrence predictability...")
    horizons = [1, 2, 3, 6]
    window = 3  # months of history to check co-occurrence

    results = {h: [] for h in horizons}
    result_months = []

    for mi, ym in enumerate(months):
        if mi < window: continue

        # build pair set from window months before t
        window_pairs = set()
        for wm in months[mi-window:mi]:
            for v in var_mass[wm]:
                ml = sorted(v)
                for i in range(len(ml)):
                    for j in range(i+1, min(i+6, len(ml))):
                        window_pairs.add((ml[i], ml[j]))

        current_vars = set(var_mass[ym].keys())
        result_months.append(mi)

        for h in horizons:
            if mi + h >= len(months):
                results[h].append(float('nan')); continue

            future_ym = months[mi + h]
            future_vars = var_mass[future_ym]
            new_vars = {v: w for v, w in future_vars.items()
                       if v not in current_vars}
            if not new_vars:
                results[h].append(float('nan')); continue

            # for each new variant, what fraction of its pairs
            # were already in window_pairs?
            fracs, weights = [], []
            for v, w in new_vars.items():
                ml = sorted(v)
                pairs_in_v = [(ml[i],ml[j])
                              for i in range(len(ml))
                              for j in range(i+1, min(i+6, len(ml)))]
                if not pairs_in_v: continue
                known = sum(1 for p in pairs_in_v if p in window_pairs)
                fracs.append(known / len(pairs_in_v))
                weights.append(w)

            if fracs:
                results[h].append(
                    np.average(fracs, weights=weights))
            else:
                results[h].append(float('nan'))

    fig, ax = plt.subplots(figsize=(12, 4))
    colors_h = {1:BLUE, 2:TEAL, 3:AMBER, 6:RED}
    for h in horizons:
        vals = [results[h][i] for i in range(len(result_months))]
        xi   = result_months
        clean_x = [xi[i] for i in range(len(xi))
                   if not np.isnan(vals[i])]
        clean_v = [vals[i] for i in range(len(vals))
                   if not np.isnan(vals[i])]
        ax.plot(clean_x, clean_v, color=colors_h[h], lw=1.8,
               label=f'h = {h} month{"s" if h>1 else ""}')

    ax.axhline(1.0, color='gray', lw=0.8, linestyle=':',
              label='All pairs already known (= 1)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower right')
    xticks(ax, months)
    regime_lines(ax, months)
    ax.set_title('Fraction of Mutation Pairs in New Variants\nAlready Co-occurring in Prior 3 Months',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Month of origin', fontsize=10)
    ax.set_ylabel('Weighted fraction of pairs\nin new variants already known',
                 fontsize=10)
    fig.subplots_adjust(left=0.12)
    savefig(fig, os.path.join(a.out_dir, 'plot5_cooc_predictability.png'))
    print("all done.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--out-dir', default='results/', dest='out_dir')
    run(p.parse_args())

if __name__ == '__main__':
    main()
