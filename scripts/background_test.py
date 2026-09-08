#!/usr/bin/env python3
"""
Background dependence: does which substitution attaches depend on WHICH
predecessor it attaches to, beyond how much of that predecessor is
circulating?

This is the distinction the ranking experiment could not make.  There,
`pred_max` -- predecessor frequency -- carried almost all the signal.  But
frequency is mass, not specificity: a heavily sequenced predecessor spawns
many observed children under either hypothesis.  Background dependence
predicts something stronger, that it spawns PARTICULAR children.

CONSTRUCTION
------------
From observed r=1 arrivals, build a contingency table
    A[x, s] = number of times (predecessor x + substitution s) was first
              observed as a new variant
Under a frequency-only null, attachment is independent of identity:
    E[x, s] = row_total(x) * col_total(s) / N
Mutual information between the row and column variables measures how far
the real table departs from that.

NULL
----
MI is positive in any finite table by chance, so the comparison is against
a permutation that preserves both margins and destroys identity: shuffle
which substitution attached to which predecessor, keeping each predecessor's
number of attachments and each substitution's total fixed.  Real MI above
the shuffled distribution is background dependence; equal MI is frequency
alone.

CONFOUND
--------
Sequencing depth.  A predecessor observed 100x more often has a denser,
less noisy row.  Margin-preserving permutation controls this at the margin,
not the noise level, so results are reported across several count floors.

Usage
-----
  zstdcat data/raw/metadata.tsv.zst | python3 background_test.py \
      --out results/background --window 2 --perms 200
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
    intern, counts = {}, defaultdict(lambda: defaultdict(int))
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
        if n % 1000000 == 0:
            print(f'  {n:,} rows', file=sys.stderr)
    print(f'  {n:,} rows, {len(intern):,} spike substitutions', file=sys.stderr)
    return counts, intern


def mutual_information(pairs):
    """pairs: list of (row, col).  Returns MI in nats."""
    n = len(pairs)
    if n == 0:
        return 0.0
    joint = defaultdict(int)
    rmarg = defaultdict(int)
    cmarg = defaultdict(int)
    for r, c in pairs:
        joint[(r, c)] += 1
        rmarg[r] += 1
        cmarg[c] += 1
    mi = 0.0
    for (r, c), v in joint.items():
        pxy = v / n
        px = rmarg[r] / n
        py = cmarg[c] / n
        mi += pxy * math.log(pxy / (px * py))
    return mi


def dense_core(ev, n_pred, n_sub, max_iter=8):
    """Restrict to the `n_pred` predecessors and `n_sub` substitutions with
    the most attachments, iterating until the selection is stable.

    Without this the contingency table is almost all 0s and 1s: MI saturates
    at its maximum for the real and permuted tables alike and the test has no
    power.  Selecting a dense subtable is what makes MI estimable.  It also
    restricts the claim to the well-sampled core, which must be stated."""
    for _ in range(max_iter):
        rc, cc = defaultdict(int), defaultdict(int)
        for r, c in ev:
            rc[r] += 1
            cc[c] += 1
        keep_r = {r for r, _ in sorted(rc.items(), key=lambda z: -z[1])[:n_pred]}
        keep_c = {c for c, _ in sorted(cc.items(), key=lambda z: -z[1])[:n_sub]}
        nxt = [(r, c) for r, c in ev if r in keep_r and c in keep_c]
        if len(nxt) == len(ev) or not nxt:
            return nxt
        ev = nxt
    return ev


def jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def background_similarity(events, min_attach=2):
    """Mean pairwise similarity among the predecessors that each substitution
    attached to.

    Two earlier statistics failed here and the reasons are worth recording.
    Recurrence is identically zero: a (predecessor, substitution) pair yields
    a NEW variant at most once, so pairs cannot repeat across months.  Mutual
    information has a small-sample bias of order (rows-1)(cols-1)/2n, which
    swamps the signal in tables this sparse.

    This statistic sidesteps both.  It asks the hypothesis directly: when a
    substitution attaches in several places, are those places alike?  If
    attachment is background-dependent, the predecessors sharing a
    substitution should resemble one another more than predecessors picked at
    random.  Similarity is Jaccard over spike substitution sets."""
    by_sub = defaultdict(list)
    for x, s, _ in events:
        by_sub[s].append(x)
    vals = []
    for s, preds in by_sub.items():
        preds = list(set(preds))
        if len(preds) < min_attach:
            continue
        tot = cnt = 0.0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                tot += jaccard(preds[i], preds[j])
                cnt += 1
        if cnt:
            vals.append(tot / cnt)
    return (sum(vals) / len(vals)) if vals else float('nan'), len(vals)


def permute_within_month(events, rng):
    """Shuffle substitution labels within each month, preserving both the
    per-month predecessor margin and the per-month substitution margin."""
    by_month = defaultdict(list)
    for x, s, t in events:
        by_month[t].append((x, s))
    out = []
    for t, evs in by_month.items():
        subs = [s for _, s in evs]
        rng.shuffle(subs)
        out.extend((x, subs[i], t) for i, (x, _) in enumerate(evs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.')
    ap.add_argument('--window', type=int, default=2)
    ap.add_argument('--perms', type=int, default=200)
    ap.add_argument('--min-count', type=int, default=2)
    ap.add_argument('--maturity', type=int, default=3)
    ap.add_argument('--floors', default='0,10,100,1000',
                    help='predecessor count floors to report')
    ap.add_argument('--core-preds', type=int, default=40,
                    help='keep this many predecessors with most attachments')
    ap.add_argument('--core-subs', type=int, default=60,
                    help='keep this many substitutions with most attachments')
    ap.add_argument('--eval-from', default=None)
    ap.add_argument('--eval-to', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--qc', action='store_true', default=True)
    ap.add_argument('--no-qc', dest='qc', action='store_false')
    args = ap.parse_args()

    header = sys.stdin.readline().rstrip('\n').split('\t')
    cols = {n: k for k, n in enumerate(header)}
    for need in ('date', 'aaSubstitutions'):
        if need not in cols:
            sys.exit(f'missing column: {need}')

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

    first = {}
    for mo in months:
        for v in counts[mo]:
            if v not in first or mo < first[v]:
                first[v] = mo

    # ---- collect observed r=1 attachment events
    # For each variant new in month t, find window predecessors exactly one
    # substitution away.  Ambiguous arrivals (several candidate predecessors)
    # are kept as a single event attributed to the most frequent predecessor;
    # `--floors` sensitivity covers the effect of that choice.
    events = []          # (predecessor, substitution, month)
    pred_freq_of = {}
    print(f'\ncollecting r=1 attachments over '
          f'{month_label(use[0])}..{month_label(use[-1])}', file=sys.stderr)

    for t in use:
        window = [m for m in months if t - args.window <= m < t]
        if not window:
            continue
        pf = defaultdict(int)
        for m in window:
            for v, c in counts[m].items():
                pf[v] += c
        # index window variants by their substitution set for O(1) lookup
        wset = set(pf)
        n_ev = 0
        for y, c in counts[t].items():
            if first.get(y) != t or c < args.min_count:
                continue
            best = None
            for s in y:                      # drop one substitution -> parent
                x = tuple(v for v in y if v != s)
                if x in wset:
                    if best is None or pf[x] > pf[best[0]]:
                        best = (x, s)
            if best is not None:
                events.append((best[0], best[1], t))
                pred_freq_of[best[0]] = pf[best[0]]
                n_ev += 1
        print(f'  {month_label(t)}: {n_ev} attachments', file=sys.stderr)

    if len(events) < 100:
        sys.exit(f'only {len(events)} attachment events; not enough to test')

    print(f'\ntotal attachments: {len(events):,}', file=sys.stderr)

    # ---- recurrence against a within-month permutation, at several floors
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    lines = ['floor\tn_events\tn_preds\tn_subs\tsimil_real\tsimil_null_mean'
             '\tsimil_null_sd\tz\tp_perm']
    print('\nfloor   events  preds  subs   simil   null     z        p',
          file=sys.stderr)

    for floor in [int(x) for x in args.floors.split(',')]:
        ev3 = [(x, s, t) for x, s, t in events if pred_freq_of[x] >= floor]
        core = set(dense_core([(x, s) for x, s, _ in ev3],
                              args.core_preds, args.core_subs))
        keep_r = {r for r, _ in core}
        keep_c = {c for _, c in core}
        ev3 = [(x, s, t) for x, s, t in ev3 if x in keep_r and s in keep_c]
        if len(ev3) < 60:
            print(f'{floor:<7} {len(ev3):<7} too few after core filter, skipped',
                  file=sys.stderr)
            continue

        real, n_sub_tested = background_similarity(ev3)
        if n_sub_tested < 5:
            print(f'{floor:<7} {len(ev3):<7} only {n_sub_tested} testable '
                  f'substitutions, skipped', file=sys.stderr)
            continue
        nulls = []
        for _ in range(args.perms):
            nulls.append(background_similarity(
                permute_within_month(ev3, rng))[0])
        m = sum(nulls) / len(nulls)
        sd = (sum((v - m) ** 2 for v in nulls) / len(nulls)) ** 0.5
        z = (real - m) / sd if sd > 0 else float('nan')
        p = (sum(1 for v in nulls if v >= real) + 1) / (len(nulls) + 1)

        lines.append(f'{floor}\t{len(ev3)}\t{len(keep_r)}\t{len(keep_c)}'
                     f'\t{real:.4f}\t{m:.4f}\t{sd:.4f}\t{z:.1f}\t{p:.4f}')
        print(f'{floor:<7} {len(ev3):<7} {len(keep_r):<6} {len(keep_c):<6} '
              f'{real:<7.4f} {m:<8.4f} {z:>6.1f}  {p:.4f}', file=sys.stderr)

    path = os.path.join(args.out, 'background_test.tsv')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    # ---- which substitutions are most background-specific
    ev = dense_core([(x, s) for x, s, _ in events],
                    args.core_preds, args.core_subs)
    joint, rm, cm = defaultdict(int), defaultdict(int), defaultdict(int)
    for r, c in ev:
        joint[(r, c)] += 1
        rm[r] += 1
        cm[c] += 1
    n = len(ev)
    contrib = defaultdict(float)
    for (r, c), v in joint.items():
        pxy = v / n
        contrib[c] += pxy * math.log(pxy / ((rm[r] / n) * (cm[c] / n)))
    top = sorted(contrib.items(), key=lambda z: -z[1])[:15]

    top_path = os.path.join(args.out, 'background_top_subs.tsv')
    with open(top_path, 'w') as fh:
        fh.write('substitution\tmi_contribution\tn_attachments\n')
        for s, v in top:
            fh.write(f'{intern_rev[s]}\t{v:.5f}\t{cm[s]}\n')

    print('\nmost background-specific substitutions:', file=sys.stderr)
    for s, v in top[:10]:
        print(f'  {intern_rev[s]:<14} MI {v:.5f}  n={cm[s]}', file=sys.stderr)

    print(f'\nwrote {path}\nwrote {top_path}', file=sys.stderr)
    print('\nreading it: simil >> null means substitutions attach to '
          'predecessors that\nresemble each other -- attachment depends on '
          'which background, not just how\nmuch of it is circulating. '
          'simil ~ null means frequency explains everything\nand the '
          'background hypothesis is not supported by this test.',
          file=sys.stderr)


if __name__ == '__main__':
    main()
