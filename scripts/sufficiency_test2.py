#!/usr/bin/env python3
"""
Sufficiency: is the spike sequence enough, or does the substitution rate also
depend on the non-spike background a lineage carries?

This is the corollary's condition, tested directly.  The forward equation
closes exactly when lineages sharing a state have the same outward rates.  Our
state is the spike sequence.  Two genomes with identical spike can sit on very
different full-genome backgrounds, and if that background changes what happens
next, the collapse in the proof fails.

  sufficiency holds  ->  tree already marginalised, drop it
  sufficiency fails  ->  c must enter the state, and the effect size says how
                         much background context is worth

WHY NOT PREDECESSOR IDENTITY
----------------------------
An earlier version of this test grouped arrivals by child spike sequence and
split by which predecessor they came from.  That cannot work: a spike sequence
is NEW only once, so it is reached from exactly one predecessor by
construction, and the test yields zero groups.  Same trap as the recurrence
statistic.

WHY NOT A PANGO LABEL
---------------------
It would work, but a lineage label is a model output whose nomenclature is
assigned retrospectively.  The non-spike substitutions are the background
itself, observed directly in the same column, with nothing inferred.

DESIGN
------
1. Parse each genome into (spike set, non-spike set).
2. Group genomes by spike set.  Within a group, keep the distinct non-spike
   backgrounds -- these are lineages that share our state but differ in what
   we cannot see.
3. For each such group, look at month t+1 for genomes whose spike is the
   group's spike plus exactly one substitution.  Assign each to whichever
   background in the group it is closest to (Jaccard on non-spike sets);
   descendants inherit background, so nearest-background is the natural
   assignment.
4. Compare the distribution of acquired spike substitutions across
   backgrounds, by Jensen-Shannon divergence, aggregated over groups.

NULL
----
Shuffle the background assignment within each group.  Group sizes and the
pooled substitution distribution are preserved; only the pairing is destroyed.

Usage
-----
  zstdcat data/raw/metadata.tsv.zst | python3 sufficiency_test.py \
      --out results/sufficiency --perms 500
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict


def month_index(d):
    if not d or len(d) < 7:
        return None
    y, m = d[:4], d[5:7]
    if not (y.isdigit() and m.isdigit()):
        return None
    m = int(m)
    return int(y) * 12 + (m - 1) if 1 <= m <= 12 else None


def month_label(mo):
    return f'{mo // 12}-{mo % 12 + 1:02d}'


def stream(fh, cols, args):
    """Returns obs[month][(spike, nonspike)] = count."""
    i_date, i_aa = cols['date'], cols['aaSubstitutions']
    i_qc = cols.get('QC_overall_status')
    ncol = len(cols)
    intern = {}
    obs = defaultdict(lambda: defaultdict(int))
    n = 0
    for line in fh:
        n += 1
        f = line.rstrip('\n').split('\t')
        if len(f) < ncol:
            continue
        if i_qc is not None and args.qc and f[i_qc] not in ('good', 'mediocre'):
            continue
        mo = month_index(f[i_date])
        if mo is None or not f[i_aa]:
            continue
        sp, ns = [], []
        for tok in f[i_aa].split(','):
            tok = tok.strip()
            if not tok or ':' not in tok:
                continue
            v = intern.get(tok)
            if v is None:
                v = len(intern)
                intern[tok] = v
            (sp if tok.startswith('S:') else ns).append(v)
        if not sp or not ns:
            continue
        obs[mo][(tuple(sorted(set(sp))), tuple(sorted(set(ns))))] += 1
        if n % 1000000 == 0:
            print(f'  {n:,} rows', file=sys.stderr)
    print(f'  {n:,} rows, {len(intern):,} substitutions', file=sys.stderr)
    return obs, intern


def jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def js_divergence(pa, pb):
    na, nb = sum(pa.values()), sum(pb.values())
    if na == 0 or nb == 0:
        return 0.0
    d = 0.0
    for k in set(pa) | set(pb):
        p, q = pa.get(k, 0) / na, pb.get(k, 0) / nb
        m = 0.5 * (p + q)
        if p > 0:
            d += 0.5 * p * math.log(p / m)
        if q > 0:
            d += 0.5 * q * math.log(q / m)
    return d


def group_divergence(labelled):
    """labelled: [(background_id, acquired_substitution)]."""
    by_bg = defaultdict(lambda: defaultdict(int))
    for bg, s in labelled:
        by_bg[bg][s] += 1
    bgs = list(by_bg)
    if len(bgs) < 2:
        return None
    tot = cnt = 0.0
    for i in range(len(bgs)):
        for j in range(i + 1, len(bgs)):
            tot += js_divergence(by_bg[bgs[i]], by_bg[bgs[j]])
            cnt += 1
    return tot / cnt if cnt else None


def aggregate(groups):
    vals = [group_divergence(g) for g in groups]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else float('nan'), len(vals)


def permute(groups, rng):
    out = []
    for g in groups:
        bgs = [b for b, _ in g]
        rng.shuffle(bgs)
        out.append([(bgs[i], s) for i, (_, s) in enumerate(g)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.')
    ap.add_argument('--perms', type=int, default=500)
    ap.add_argument('--min-bg-count', type=int, default=3,
                    help='a background must be seen this often to be used')
    ap.add_argument('--min-group', type=int, default=6,
                    help='minimum labelled outcomes in a group')
    ap.add_argument('--max-bg', type=int, default=8,
                    help='most frequent backgrounds kept per spike group')
    ap.add_argument('--maturity', type=int, default=3)
    ap.add_argument('--eval-from', default=None)
    ap.add_argument('--eval-to', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--qc', action='store_true', default=True)
    ap.add_argument('--no-qc', dest='qc', action='store_false')
    args = ap.parse_args()

    header = sys.stdin.readline().rstrip('\n').split('\t')
    cols = {n: k for k, n in enumerate(header)}
    if 'date' not in cols or 'aaSubstitutions' not in cols:
        sys.exit('missing required columns')

    print('streaming metadata', file=sys.stderr)
    obs, intern = stream(sys.stdin, cols, args)
    intern_rev = {v: k for k, v in intern.items()}

    months = sorted(obs)
    if args.maturity and args.maturity < len(months):
        months = months[:-args.maturity]
    if args.eval_from or args.eval_to:
        lo = month_index(args.eval_from + '-15') if args.eval_from else months[0]
        hi = month_index(args.eval_to + '-15') if args.eval_to else months[-1]
        use = [m for m in months if lo <= m <= hi]
    else:
        use = months

    groups = []
    n_spike_groups = n_multi_bg = 0

    print(f'\nbuilding groups over {month_label(use[0])}..'
          f'{month_label(use[-1])}', file=sys.stderr)

    for t in use:
        if t + 1 not in obs:
            continue

        # spike group -> {background: count}, at month t
        by_spike = defaultdict(lambda: defaultdict(int))
        for (sp, ns), c in obs[t].items():
            by_spike[sp][ns] += c

        # month t+1 indexed by spike, for the onward step
        nxt = defaultdict(list)
        for (sp, ns), c in obs[t + 1].items():
            nxt[sp].append((ns, c))

        for sp, bgs in by_spike.items():
            bgs = {b: c for b, c in bgs.items() if c >= args.min_bg_count}
            if len(bgs) < 2:
                continue
            n_spike_groups += 1
            keep = dict(sorted(bgs.items(), key=lambda z: -z[1])[:args.max_bg])
            bg_list = list(keep)

            labelled = []
            spset = set(sp)
            # children: spike is this group's spike plus exactly one
            for child_sp, entries in nxt.items():
                if len(child_sp) != len(sp) + 1:
                    continue
                extra = set(child_sp) - spset
                if len(extra) != 1 or not spset <= set(child_sp):
                    continue
                s = next(iter(extra))
                for ns, c in entries:
                    # assign to the background it most resembles; descendants
                    # inherit background, so nearest is the natural choice
                    best = max(bg_list, key=lambda b: jaccard(b, ns))
                    if jaccard(best, ns) <= 0:
                        continue
                    labelled.extend([(bg_list.index(best), s)] * min(c, 5))

            if (len(labelled) >= args.min_group
                    and len({b for b, _ in labelled}) >= 2):
                n_multi_bg += 1
                groups.append(labelled)

        if t % 6 == 0:
            print(f'  {month_label(t)}: {len(groups):,} groups so far',
                  file=sys.stderr)

    print(f'\n  spike groups with >=2 backgrounds  {n_spike_groups:,}',
          file=sys.stderr)
    print(f'  usable groups                      {len(groups):,}',
          file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'sufficiency_test.tsv')

    if len(groups) < 20:
        print('\nNOT ENOUGH GROUPS -- the test is underpowered.',
              file=sys.stderr)
        with open(path, 'w') as fh:
            fh.write('status\tn_spike_groups\tn_groups\n')
            fh.write(f'underpowered\t{n_spike_groups}\t{len(groups)}\n')
        return

    real, n_used = aggregate(groups)
    rng = random.Random(args.seed)
    nulls = [aggregate(permute(groups, rng))[0] for _ in range(args.perms)]
    m = sum(nulls) / len(nulls)
    sd = (sum((v - m) ** 2 for v in nulls) / len(nulls)) ** 0.5
    z = (real - m) / sd if sd > 0 else float('nan')
    p = (sum(1 for v in nulls if v >= real) + 1) / (len(nulls) + 1)

    with open(path, 'w') as fh:
        fh.write('status\tn_spike_groups\tn_groups\tjs_real\tjs_null_mean'
                 '\tjs_null_sd\tz\tp_perm\n')
        fh.write(f'ok\t{n_spike_groups}\t{n_used}\t{real:.5f}\t{m:.5f}'
                 f'\t{sd:.5f}\t{z:.2f}\t{p:.4f}\n')

    print(f'\n  groups used   {n_used}', file=sys.stderr)
    print(f'  JS real       {real:.5f}', file=sys.stderr)
    print(f'  JS null       {m:.5f} +/- {sd:.5f}', file=sys.stderr)
    print(f'  z             {z:.2f}', file=sys.stderr)
    print(f'  p (perm)      {p:.4f}', file=sys.stderr)

    if p < 0.05:
        print('\n  -> non-spike background changes what is acquired next. '
              'Sufficiency FAILS\n     for the spike-only state; c must enter '
              'the state. The effect size is\n     how much background '
              'context is worth.', file=sys.stderr)
    else:
        print('\n  -> no detectable dependence on non-spike background. '
              'Spike state alone is\n     sufficient by this test and the '
              'forward equation is exact. This is\n     failure to reject, '
              'not proof -- report power alongside it.', file=sys.stderr)

    print(f'\nwrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
