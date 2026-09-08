#!/usr/bin/env python3
"""
Do substitutions that arrive together cluster, or is co-arrival chance?

This is the last untested corner of the sequence-only route.  Everything so
far has looked at single substitutions: which one gets added, conditioned on
the source.  Three tests came back flat -- six models with identical
log-loss, source-conditioned attention 0.008 nats WORSE than ignoring the
source, six of seven features at the null.

Distance-2 arrivals give something distance-1 cannot: a co-arrival event.
Two substitutions appearing together in one step, observed directly rather
than inferred from co-occurrence in standing sequences.  If certain pairs
recur across different backgrounds, there is structure in the mutation
events themselves.

WHAT IS COMPARED
----------------
For each variant first seen at month t, look for an observed sequence in the
preceding window that is exactly two substitutions away.  The two missing
substitutions are the co-arrival pair.

  real     how concentrated the pair distribution is
  null     same, after shuffling which substitutions co-arrive while
           holding each substitution's own total fixed

The null is the point.  Some pairs are common simply because both members
are common; shuffling preserves that and destroys only the pairing.  A
concentration above the null means pairs recur beyond what the individual
frequencies explain.

WHY POOLED, AND WHY THIS ONE FIRST
----------------------------------
This ignores which background the pair landed on.  The background-specific
version is the interesting one, but it cannot be answered here: 400
substitutions give 160,000 possible pairs, most occur once or never, and
splitting those by source leaves cells that are almost all zero -- the same
event-count wall that defeated the attention test.

So this is a gate.  If pairs do not cluster even pooled, the
background-conditioned version cannot be hiding underneath, and the
sequence-only route closes.  If they do cluster, the harder question becomes
worth the effort.

Usage
-----
  zstdcat data/raw/metadata.tsv.zst | python3 pair_test.py \
      --out results/pairs --eval-from 2022-01 --eval-to 2023-06
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


def stream(fh, cols, qc):
    i_date, i_aa = cols['date'], cols['aaSubstitutions']
    i_qc = cols.get('QC_overall_status')
    ncol = len(cols)
    intern = {}
    counts = defaultdict(lambda: defaultdict(int))
    n = 0
    for line in fh:
        n += 1
        f = line.rstrip('\n').split('\t')
        if len(f) < ncol:
            continue
        if i_qc is not None and qc and f[i_qc] not in ('good', 'mediocre'):
            continue
        mo = month_index(f[i_date])
        if mo is None or not f[i_aa]:
            continue
        ids = []
        for tok in f[i_aa].split(','):
            tok = tok.strip()
            if tok.startswith('S:'):
                v = intern.get(tok)
                if v is None:
                    v = len(intern)
                    intern[tok] = v
                ids.append(v)
        if ids:
            counts[mo][tuple(sorted(set(ids)))] += 1
        if n % 2000000 == 0:
            print(f'  {n:,} rows', file=sys.stderr)
    print(f'  {n:,} rows, {len(intern):,} spike substitutions', file=sys.stderr)
    return counts, intern


def collect_pairs(counts, months, use, window, min_count):
    """Co-arrival pairs: for each new variant, an observed sequence exactly
    two substitutions back, and the two that were added."""
    first = {}
    for m in months:
        for v in counts[m]:
            if v not in first or m < first[v]:
                first[v] = m

    pairs, singles, n_new = [], [], 0
    for t in use:
        win = [m for m in months if t - window <= m < t]
        if not win:
            continue
        wset = set()
        for m in win:
            wset.update(counts[m])
        for y, c in counts[t].items():
            if first.get(y) != t or c < min_count:
                continue
            n_new += 1
            # exactly one substitution back
            got_single = False
            for s in y:
                if tuple(v for v in y if v != s) in wset:
                    singles.append(s)
                    got_single = True
                    break
            if got_single:
                continue
            # exactly two back -- a genuine co-arrival
            ys = list(y)
            found = None
            for i in range(len(ys)):
                for j in range(i + 1, len(ys)):
                    x = tuple(v for v in ys if v != ys[i] and v != ys[j])
                    if x in wset:
                        found = (ys[i], ys[j])
                        break
                if found:
                    break
            if found:
                pairs.append(tuple(sorted(found)))
        print(f'  {month_label(t)}: {len(pairs):,} pairs so far',
              file=sys.stderr)
    return pairs, singles, n_new


def concentration(pairs):
    """How concentrated the pair distribution is.

    Two statistics, because they answer slightly different questions:
      repeats  -- pairs seen more than once, as a fraction of all pairs.
                  Direct and easy to state.
      entropy  -- of the pair distribution; lower means more concentrated.
                  Sensitive to the whole shape, not just repeats."""
    c = defaultdict(int)
    for p in pairs:
        c[p] += 1
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0
    repeats = sum(v for v in c.values() if v > 1) / n
    ent = -sum((v / n) * math.log(v / n) for v in c.values())
    return repeats, ent, len(c)


def shuffle_pairs(pairs, rng):
    """Break the pairing, keep each substitution's own total.

    Pool every substitution from every pair, shuffle, redeal in twos.  A pair
    of two common substitutions stays likely; a pair that recurs because
    those two specifically travel together does not."""
    flat = [s for p in pairs for s in p]
    rng.shuffle(flat)
    out = []
    for i in range(0, len(flat) - 1, 2):
        a, b = flat[i], flat[i + 1]
        if a != b:
            out.append((a, b) if a < b else (b, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.')
    ap.add_argument('--window', type=int, default=2)
    ap.add_argument('--min-count', type=int, default=2)
    ap.add_argument('--perms', type=int, default=500)
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
    counts, intern = stream(sys.stdin, cols, args.qc)
    intern_rev = {v: k for k, v in intern.items()}

    months = sorted(counts)
    if args.maturity and args.maturity < len(months):
        months = months[:-args.maturity]
    if args.eval_from or args.eval_to:
        lo = month_index(args.eval_from + '-15') if args.eval_from else months[0]
        hi = month_index(args.eval_to + '-15') if args.eval_to else months[-1]
        use = [m for m in months if lo <= m <= hi]
    else:
        use = months

    print(f'\ncollecting co-arrivals over {month_label(use[0])}..'
          f'{month_label(use[-1])}', file=sys.stderr)
    pairs, singles, n_new = collect_pairs(counts, months, use, args.window,
                                          args.min_count)

    print(f'\n  new variants                {n_new:,}', file=sys.stderr)
    print(f'  reached by one substitution {len(singles):,}', file=sys.stderr)
    print(f'  reached by two (co-arrival) {len(pairs):,}', file=sys.stderr)

    if len(pairs) < 200:
        sys.exit('too few co-arrival pairs to test')

    rep, ent, uniq = concentration(pairs)
    rng = random.Random(args.seed)
    nulls_rep, nulls_ent = [], []
    for _ in range(args.perms):
        sp = shuffle_pairs(pairs, rng)
        r, e, _ = concentration(sp)
        nulls_rep.append(r)
        nulls_ent.append(e)

    def stats(v):
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        return m, sd

    mr, sr = stats(nulls_rep)
    me, se = stats(nulls_ent)
    z_rep = (rep - mr) / sr if sr > 0 else float('nan')
    z_ent = (me - ent) / se if se > 0 else float('nan')   # lower = concentrated
    p_rep = (sum(1 for x in nulls_rep if x >= rep) + 1) / (args.perms + 1)
    p_ent = (sum(1 for x in nulls_ent if x <= ent) + 1) / (args.perms + 1)

    print(f'\n  distinct pairs              {uniq:,}', file=sys.stderr)
    print(f'\n  fraction in repeated pairs  real {rep:.4f}   '
          f'null {mr:.4f} +/- {sr:.4f}   z {z_rep:.1f}   p {p_rep:.4f}',
          file=sys.stderr)
    print(f'  pair entropy (lower=conc.)  real {ent:.4f}   '
          f'null {me:.4f} +/- {se:.4f}   z {z_ent:.1f}   p {p_ent:.4f}',
          file=sys.stderr)

    top = sorted(((v, k) for k, v in
                  ((k, sum(1 for p in pairs if p == k))
                   for k in set(pairs))), reverse=True)[:15]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'pair_test.tsv')
    with open(path, 'w') as fh:
        fh.write('metric\treal\tnull_mean\tnull_sd\tz\tp\n')
        fh.write(f'repeat_fraction\t{rep:.5f}\t{mr:.5f}\t{sr:.5f}'
                 f'\t{z_rep:.2f}\t{p_rep:.4f}\n')
        fh.write(f'pair_entropy\t{ent:.5f}\t{me:.5f}\t{se:.5f}'
                 f'\t{z_ent:.2f}\t{p_ent:.4f}\n')
        fh.write('\nsubstitution_a\tsubstitution_b\tcount\n')
        for cnt, (a, b) in top:
            fh.write(f'{intern_rev[a]}\t{intern_rev[b]}\t{cnt}\n')

    print('\n  most frequent co-arriving pairs:', file=sys.stderr)
    for cnt, (a, b) in top[:8]:
        print(f'    {intern_rev[a]:<12} {intern_rev[b]:<12} {cnt}',
              file=sys.stderr)

    print('\nreading it:', file=sys.stderr)
    if z_rep > 3 and z_ent > 3:
        print('  Pairs recur beyond what the individual frequencies explain.\n'
              '  There is structure in which substitutions arrive together, and\n'
              '  the background-conditioned version is now worth the effort.',
              file=sys.stderr)
    else:
        print('  Co-arrival is explained by the individual frequencies alone.\n'
              '  No pair structure pooled means none can be hiding in the\n'
              '  background-conditioned version either, which has far fewer\n'
              '  events per cell. The sequence-only route closes here, and that\n'
              '  is a measurement rather than a failure to find the right model.',
              file=sys.stderr)

    print(f'\nwrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
