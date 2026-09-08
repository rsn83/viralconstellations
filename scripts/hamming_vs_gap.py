#!/usr/bin/env python3
"""
Hamming distance to nearest observed predecessor, as a function of the gap
in months.

Question it answers: does a longer observation gap show more accumulated
substitutions?  If the slope is flat, dt carries no information and the
substitution rate is not identified from these data.

Design note: for each new spike variant first seen in month t, the minimum
distance is computed SEPARATELY against each prior month t-1, t-2, ... t-W.
Taking a single nearest neighbour over the whole window and reading off its
gap would bias the result -- distance and gap would be jointly minimised.

Memory: the file is never loaded.  Substitution strings are interned to
ints, distinct variants are deduplicated on first pass, and the candidate
index holds only a rolling W-month window.

Usage:
    zstdcat data/raw/metadata.tsv.zst | python3 hamming_vs_gap.py --out results/
"""

import argparse
import sys
from collections import defaultdict

# ----------------------------------------------------------------- config

DEFAULT_WINDOW = 6          # months of history to look back
PROBE_RARE = 3              # index/probe on this many rarest substitutions
MAX_CANDIDATES = 20000      # per (variant, prior-month) probe, safety cap


# ------------------------------------------------------------------ parse

def month_index(datestr):
    """'2022-03-14' or '2022-03' -> integer month.  None if unusable."""
    if not datestr or len(datestr) < 7:
        return None
    y, m = datestr[:4], datestr[5:7]
    if not (y.isdigit() and m.isdigit()):
        return None
    m = int(m)
    if not 1 <= m <= 12:
        return None
    return int(y) * 12 + (m - 1)


def spike_subs(field, keep_all_genes=False):
    """aaSubstitutions -> tuple of the spike entries."""
    if not field:
        return ()
    out = []
    for tok in field.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if keep_all_genes or tok.startswith('S:'):
            out.append(tok)
    return tuple(out)


def stream_variants(fh, cols, args):
    """One pass.  Yields nothing; returns {variant_tuple: first_month}."""
    i_date = cols['date']
    i_aa = cols['aaSubstitutions']
    i_qc = cols.get('QC_overall_status')
    i_del = cols.get('deletions')
    ncol = len(cols)

    intern = {}
    first_seen = {}
    n_rows = n_kept = 0

    for line in fh:
        n_rows += 1
        f = line.rstrip('\n').split('\t')
        if len(f) < ncol:
            continue

        if i_qc is not None and args.qc:
            if f[i_qc] not in ('good', 'mediocre'):
                continue

        mo = month_index(f[i_date])
        if mo is None:
            continue

        subs = spike_subs(f[i_aa], args.all_genes)
        if args.deletions and i_del is not None and f[i_del]:
            subs = subs + tuple('del:' + d.strip()
                                for d in f[i_del].split(',') if d.strip())
        if not subs:
            continue

        ids = []
        for s in subs:
            v = intern.get(s)
            if v is None:
                v = len(intern)
                intern[s] = v
            ids.append(v)
        key = tuple(sorted(set(ids)))

        prev = first_seen.get(key)
        if prev is None or mo < prev:
            first_seen[key] = mo
        n_kept += 1

        if n_rows % 500000 == 0:
            print(f'  {n_rows:,} rows, {len(first_seen):,} distinct variants',
                  file=sys.stderr)

    print(f'  done: {n_rows:,} rows, {n_kept:,} usable, '
          f'{len(first_seen):,} distinct variants, {len(intern):,} substitutions',
          file=sys.stderr)
    return first_seen


# --------------------------------------------------------------- distance

def hamming(a, b):
    """Symmetric-difference size of two sorted int tuples."""
    i = j = shared = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] == b[j]:
            shared += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return la + lb - 2 * shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    ap.add_argument('--out', default='.')
    ap.add_argument('--min-month', type=int, default=None,
                    help='skip variants before this month index (maturity cutoff)')
    ap.add_argument('--deletions', action='store_true')
    ap.add_argument('--all-genes', action='store_true',
                    help='use every gene, not just spike')
    ap.add_argument('--qc', action='store_true', default=True)
    ap.add_argument('--no-qc', dest='qc', action='store_false')
    ap.add_argument('--sample-per-month', type=int, default=2000,
                    help='cap on new variants scored per month (0 = all). '
                         'The distance distribution is a per-variant statistic, '
                         'so a random sample estimates it fine.')
    ap.add_argument('--max-candidates', type=int, default=3000,
                    help='cap on candidates examined per (variant, prior month)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    header = sys.stdin.readline().rstrip('\n').split('\t')
    cols = {name: k for k, name in enumerate(header)}
    for need in ('date', 'aaSubstitutions'):
        if need not in cols:
            sys.exit(f'missing column: {need}')

    print('pass 1: streaming metadata', file=sys.stderr)
    first_seen = stream_variants(sys.stdin, cols, args)
    if not first_seen:
        sys.exit('no usable rows')

    # variants grouped by the month they first appear
    by_month = defaultdict(list)
    for key, mo in first_seen.items():
        by_month[mo].append(key)
    months = sorted(by_month)

    # global substitution frequency -> lets us probe on rare ones
    freq = defaultdict(int)
    for key in first_seen:
        for s in key:
            freq[s] += 1

    def probe_keys(key):
        return sorted(key, key=lambda s: freq[s])[:PROBE_RARE]

    # inverted index, one per month, rare-substitution postings only
    index = {}
    for mo in months:
        idx = defaultdict(list)
        for key in by_month[mo]:
            for s in probe_keys(key):
                idx[s].append(key)
        index[mo] = idx

    print(f'pass 2: {len(months)} months, window={args.window}', file=sys.stderr)

    import random
    rng = random.Random(args.seed)

    rows = []
    for mo in months:
        if args.min_month is not None and mo < args.min_month:
            continue
        news = by_month[mo]
        n_all = len(news)
        if args.sample_per_month and n_all > args.sample_per_month:
            news = rng.sample(news, args.sample_per_month)
        probes = {k: probe_keys(k) for k in news}
        for gap in range(1, args.window + 1):
            prior = mo - gap
            if prior not in index:
                continue
            idx = index[prior]
            cap = args.max_candidates
            for key in news:
                seen = set()
                best = None
                for s in probes[key]:
                    for cand in idx.get(s, ()):
                        if cand in seen:
                            continue
                        seen.add(cand)
                        d = hamming(key, cand)
                        if best is None or d < best:
                            best = d
                            if best <= 1:
                                break          # cannot do better; 0 is same set
                        if len(seen) >= cap:
                            break
                    if (best is not None and best <= 1) or len(seen) >= cap:
                        break
                if best is not None:
                    rows.append((mo, gap, best, len(key)))
        print(f'  month {mo // 12}-{mo % 12 + 1:02d}: '
              f'{len(news)} scored of {n_all} new variants', file=sys.stderr)

    # ------------------------------------------------------------- output
    import os
    os.makedirs(args.out, exist_ok=True)

    raw = os.path.join(args.out, 'hamming_vs_gap.tsv')
    with open(raw, 'w') as fh:
        fh.write('month\tgap_months\tmin_hamming\tvariant_size\n')
        for r in rows:
            fh.write('%d\t%d\t%d\t%d\n' % r)

    agg = defaultdict(list)
    for _, gap, d, _ in rows:
        agg[gap].append(d)

    summary = os.path.join(args.out, 'hamming_vs_gap_summary.tsv')
    with open(summary, 'w') as fh:
        fh.write('gap_months\tn\tmean\tmedian\tfrac_le_1\tfrac_le_2\n')
        print('\ngap  n        mean   median  <=1     <=2', file=sys.stderr)
        for gap in sorted(agg):
            v = sorted(agg[gap])
            n = len(v)
            mean = sum(v) / n
            med = v[n // 2]
            f1 = sum(1 for x in v if x <= 1) / n
            f2 = sum(1 for x in v if x <= 2) / n
            fh.write('%d\t%d\t%.3f\t%d\t%.4f\t%.4f\n' % (gap, n, mean, med, f1, f2))
            print('%-4d %-8d %-6.2f %-7d %-7.3f %.3f'
                  % (gap, n, mean, med, f1, f2), file=sys.stderr)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gaps = sorted(agg)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        ax[0].boxplot([agg[g] for g in gaps], positions=gaps, showfliers=False)
        ax[0].plot(gaps, [sum(agg[g]) / len(agg[g]) for g in gaps],
                   'o-', color='crimson', label='mean')
        ax[0].set_xlabel('gap (months)')
        ax[0].set_ylabel('min Hamming to a variant of that month')
        ax[0].set_title('distance vs observation gap')
        ax[0].legend()

        for g in gaps[:4]:
            v = agg[g]
            mx = min(12, max(v)) if v else 1
            hist = [sum(1 for x in v if x == d) / len(v) for d in range(mx + 1)]
            ax[1].plot(range(mx + 1), hist, 'o-', label=f'gap={g}')
        ax[1].set_xlabel('min Hamming distance')
        ax[1].set_ylabel('fraction of new variants')
        ax[1].set_title('distance distribution')
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'hamming_vs_gap.png'), dpi=150)
        print(f'\nwrote {args.out}/hamming_vs_gap.png', file=sys.stderr)
    except ImportError:
        print('\nmatplotlib not available; TSVs written', file=sys.stderr)

    print(f'wrote {raw}\nwrote {summary}', file=sys.stderr)


if __name__ == '__main__':
    main()
