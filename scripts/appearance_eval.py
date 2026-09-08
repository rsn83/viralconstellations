#!/usr/bin/env python3
"""
Appearance ranking: does conditioning on the predecessor beat marginal
substitution frequency, and how far is either from the achievable ceiling?

THE TASK
--------
For evaluation month t, build a candidate set of variants that do not yet
exist: every pair (predecessor x observed in [t-W, t-1], substitution s)
gives a candidate child y = x + s.  Rank the candidates.  The positives are
the candidates that actually show up as new variants in month t.

Restricting to single-substitution arrivals (r = 1) keeps the task clean --
each positive has an unambiguous edit.  Per the coverage diagnostic that is
about 22% of new arrivals; the rest need r = 2 and are out of scope here.

WHAT IS COMPARED
----------------
  freq          score(y) = marginal frequency of s.  No predecessor at all.
  freq_x_prev   score(y) = freq(s) * frequency of the predecessor x.
                The frequency-weighted-neighbour baseline.
  model         logistic regression on a handful of features, trained on
                months strictly before t.  Conditions on the predecessor.
  CEILING       ranks by what actually happened in a random half of month t,
                scored on the other half.  Same month, so no forecasting
                difficulty whatsoever -- whatever this scores is an upper
                bound on any cross-month method.

The ceiling is the number that decides whether a gap between model and
baseline is worth chasing or whether both are already near the limit.

Usage
-----
  zstdcat data/raw/metadata.tsv.zst | python3 appearance_eval.py \
      --out results/appearance --window 2 --seeds 5
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict


# ------------------------------------------------------------------ parse

def month_index(datestr):
    if not datestr or len(datestr) < 7:
        return None
    y, m = datestr[:4], datestr[5:7]
    if not (y.isdigit() and m.isdigit()):
        return None
    m = int(m)
    return int(y) * 12 + (m - 1) if 1 <= m <= 12 else None


def month_label(mo):
    return f'{mo // 12}-{mo % 12 + 1:02d}'


def stream(fh, cols, args):
    """One pass.  Returns (counts[month][variant], intern) with variants as
    sorted int tuples of spike substitutions.

    With --stratify-region each sequence contributes 1/(sequences from its
    country that month) instead of 1.  Countries then carry equal weight
    regardless of how much they sequence.  This is the control for the
    confound in `pred_max`: a heavily sequenced predecessor spawns more
    OBSERVED children whether or not it spawns more real ones."""
    i_date, i_aa = cols['date'], cols['aaSubstitutions']
    i_qc = cols.get('QC_overall_status')
    i_reg = cols.get('country')
    ncol = len(cols)

    intern = {}
    counts = defaultdict(lambda: defaultdict(float))
    raw = [] if args.stratify_region else None
    region_n = defaultdict(int)
    n_rows = n_kept = 0

    for line in fh:
        n_rows += 1
        f = line.rstrip('\n').split('\t')
        if len(f) < ncol:
            continue
        if i_qc is not None and args.qc and f[i_qc] not in ('good', 'mediocre'):
            continue
        mo = month_index(f[i_date])
        if mo is None:
            continue
        aa = f[i_aa]
        if not aa:
            continue
        ids = []
        for tok in aa.split(','):
            tok = tok.strip()
            if not tok.startswith('S:'):
                continue
            v = intern.get(tok)
            if v is None:
                v = len(intern)
                intern[tok] = v
            ids.append(v)
        if not ids:
            continue
        key = tuple(sorted(set(ids)))
        if args.stratify_region:
            reg = f[i_reg] if i_reg is not None else '?'
            raw.append((mo, key, reg))
            region_n[(mo, reg)] += 1
        else:
            counts[mo][key] += 1
        n_kept += 1
        if n_rows % 1000000 == 0:
            print(f'  {n_rows:,} rows', file=sys.stderr)

    if args.stratify_region:
        # scale so the month's total weight matches its raw sequence count,
        # keeping min-count thresholds on a comparable scale
        tot = defaultdict(int)
        for mo, key, reg in raw:
            tot[mo] += 1
        wsum = defaultdict(float)
        for mo, key, reg in raw:
            wsum[mo] += 1.0 / region_n[(mo, reg)]
        for mo, key, reg in raw:
            counts[mo][key] += (1.0 / region_n[(mo, reg)]) * tot[mo] / wsum[mo]
        print(f'  region-stratified over {len({r for _, _, r in raw})} countries',
              file=sys.stderr)
    print(f'  {n_rows:,} rows, {n_kept:,} usable, {len(intern):,} spike substitutions',
          file=sys.stderr)
    return counts, intern


# -------------------------------------------------------------- candidates

def build_candidates(preds, subs, existing):
    """preds: [(variant, freq)].  subs: [(sub_id, freq)].
    Returns {child_variant: [(predecessor, pred_freq, sub_id, sub_freq), ...]}
    for children that do not already exist."""
    cand = defaultdict(list)
    for x, fx in preds:
        xs = set(x)
        for s, fs in subs:
            if s in xs:
                continue
            y = tuple(sorted(xs | {s}))
            if y in existing:
                continue
            cand[y].append((x, fx, s, fs))
    return cand


FEATURE_NAMES = [
    'pred_max',     # frequency of the most common predecessor offering this child
    'pred_total',   # total predecessor mass across all routes
    'sub_freq',     # substitution frequency over the window
    'sub_recent',   # substitution frequency in the last window month
    'pos_recent',   # any substitution at this position, last window month
    'n_routes',     # how many window predecessors are one edit from this child
    'pred_size',    # predecessor variant size
]


def features(entries, sub_recent, pos_recent, intern_rev, mask=None):
    """Aggregate a candidate's (predecessor, substitution) routes into a
    fixed feature vector.  Logs throughout: these quantities span orders of
    magnitude.  `mask` selects a subset of FEATURE_NAMES."""
    best_fx = max(e[1] for e in entries)
    tot_fx = sum(e[1] for e in entries)
    fs = entries[0][3]
    s = entries[0][2]
    n_routes = len(entries)
    rec = sub_recent.get(s, 0.0)
    tok = intern_rev[s]
    pos = tok.split(':')[1][1:-1] if ':' in tok else ''
    prec = pos_recent.get(pos, 0.0)
    size = len(entries[0][0])
    vals = [
        math.log1p(best_fx),
        math.log1p(tot_fx),
        math.log1p(fs),
        math.log1p(rec),
        math.log1p(prec),
        math.log1p(n_routes),
        size / 50.0,
    ]
    if mask is None:
        return vals
    return [vals[i] for i in mask]


# ---------------------------------------------------------------- learning

def fit_logistic(X, y, seed, epochs=60, lr=0.25, l2=1e-3):
    rng = random.Random(seed)
    d = len(X[0])
    w = [rng.gauss(0, 0.01) for _ in range(d)]
    b = 0.0
    idx = list(range(len(X)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            z = b + sum(w[k] * X[i][k] for k in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            g = p - y[i]
            b -= lr * g
            for k in range(d):
                w[k] -= lr * (g * X[i][k] + l2 * w[k])
    return w, b


def score_logistic(w, b, x):
    return b + sum(w[k] * x[k] for k in range(len(w)))


# ----------------------------------------------------------------- metrics

def recall_at(ranked, positives, ks):
    if not positives:
        return {k: float('nan') for k in ks}
    out = {}
    hits = 0
    seen = 0
    ks_sorted = sorted(ks)
    ptr = 0
    for y in ranked:
        seen += 1
        if y in positives:
            hits += 1
        while ptr < len(ks_sorted) and seen == ks_sorted[ptr]:
            out[ks_sorted[ptr]] = hits / len(positives)
            ptr += 1
        if ptr >= len(ks_sorted):
            break
    for k in ks_sorted:
        out.setdefault(k, hits / len(positives))
    return out


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.')
    ap.add_argument('--window', type=int, default=2,
                    help='months of predecessors')
    ap.add_argument('--top-preds', type=int, default=400)
    ap.add_argument('--top-subs', type=int, default=400)
    ap.add_argument('--min-count', type=int, default=2,
                    help='ignore variants seen fewer than this many times')
    ap.add_argument('--eval-months', type=int, default=8)
    ap.add_argument('--eval-from', default=None,
                    help='first evaluation month, e.g. 2022-01')
    ap.add_argument('--eval-to', default=None,
                    help='last evaluation month, e.g. 2023-12')
    ap.add_argument('--maturity', type=int, default=3,
                    help='drop this many trailing months as immature')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--stratify-region', action='store_true',
                    help='weight countries equally instead of by how much '
                         'they sequence; controls the detection-probability '
                         'confound in predecessor frequency')
    ap.add_argument('--ablate', action='store_true',
                    help='also train each-feature-alone and leave-one-out models')
    ap.add_argument('--features', default=None,
                    help='comma-separated subset of ' + ','.join(FEATURE_NAMES))
    ap.add_argument('--qc', action='store_true', default=True)
    ap.add_argument('--no-qc', dest='qc', action='store_false')
    args = ap.parse_args()

    header = sys.stdin.readline().rstrip('\n').split('\t')
    cols = {n: k for k, n in enumerate(header)}
    for need in ('date', 'aaSubstitutions'):
        if need not in cols:
            sys.exit(f'missing column: {need}')

    print('streaming metadata', file=sys.stderr)
    counts, intern = stream(sys.stdin, cols, args)
    intern_rev = {v: k for k, v in intern.items()}

    months = sorted(counts)
    if args.maturity:
        months = months[:-args.maturity] if args.maturity < len(months) else months
    if args.eval_from or args.eval_to:
        lo = month_index(args.eval_from + '-15') if args.eval_from else months[0]
        hi = month_index(args.eval_to + '-15') if args.eval_to else months[-1]
        eval_months = [m for m in months if lo <= m <= hi]
    else:
        eval_months = months[-args.eval_months:]
    print(f'\nmonths {month_label(months[0])}..{month_label(months[-1])}; '
          f'evaluating {[month_label(m) for m in eval_months]}', file=sys.stderr)

    # first appearance of every variant
    first = {}
    for mo in months:
        for v in counts[mo]:
            if v not in first or mo < first[v]:
                first[v] = mo

    KS = [20, 100, 500]
    results = defaultdict(lambda: defaultdict(list))
    per_month = defaultdict(lambda: defaultdict(dict))   # [method][k][month]

    def record(name, ranked_or_scores, pos, month):
        vals = recall_at(ranked_or_scores, pos, KS)
        for k, v in vals.items():
            results[name][k].append(v)
            per_month[name][k].setdefault(month, []).append(v)

    # which feature subsets to train.  Each-alone shows what a feature carries
    # by itself; leave-one-out shows what is lost without it.  They disagree
    # when features are correlated, which is exactly what we want to see.
    full = list(range(len(FEATURE_NAMES)))
    if args.features:
        sel = [FEATURE_NAMES.index(n.strip()) for n in args.features.split(',')]
        variants = [('model[' + args.features + ']', sel)]
    elif args.ablate:
        variants = [('model_full', full)]
        variants += [(f'only:{FEATURE_NAMES[i]}', [i]) for i in full]
        variants += [(f'drop:{FEATURE_NAMES[i]}',
                      [j for j in full if j != i]) for i in full]
    else:
        variants = [('model', full)]

    for t in eval_months:
        window = [m for m in months if t - args.window <= m < t]
        if not window:
            continue

        # predecessors: variants circulating in the window
        pf = defaultdict(int)
        for m in window:
            for v, c in counts[m].items():
                pf[v] += c
        preds = sorted(((v, c) for v, c in pf.items() if c >= args.min_count),
                       key=lambda z: -z[1])[:args.top_preds]

        # substitution frequencies, from the window only (no leakage)
        sf = defaultdict(int)
        for m in window:
            for v, c in counts[m].items():
                for s in v:
                    sf[s] += c
        subs = sorted(sf.items(), key=lambda z: -z[1])[:args.top_subs]

        # recency features, last window month only
        last = window[-1]
        sub_recent, pos_recent = defaultdict(float), defaultdict(float)
        for v, c in counts[last].items():
            for s in v:
                sub_recent[s] += c
                tok = intern_rev[s]
                pos_recent[tok.split(':')[1][1:-1]] += c

        existing = {v for v in first if first[v] < t}
        cand = build_candidates(preds, subs, existing)
        if not cand:
            continue

        # positives: candidates that really are new this month
        new_this = {v for v, c in counts[t].items()
                    if first.get(v) == t and c >= args.min_count}
        positives = set(cand) & new_this
        if len(positives) < 10:
            print(f'  {month_label(t)}: only {len(positives)} positives, skipped',
                  file=sys.stderr)
            continue

        keys = list(cand)
        feats = {y: features(cand[y], sub_recent, pos_recent, intern_rev)
                 for y in keys}

        # ---- baselines
        by_freq = sorted(keys, key=lambda y: -cand[y][0][3])
        by_freq_x = sorted(keys, key=lambda y: -(cand[y][0][3]
                                                 * max(e[1] for e in cand[y])))
        # the forward-equation weight: sum over ALL routes into y of
        # (predecessor frequency) x (substitution frequency).  This is the
        # baseline the model must actually beat -- unlike the two above it
        # sees route multiplicity, which is most of the predecessor signal.
        by_fwd = sorted(keys,
                        key=lambda y: -sum(e[1] * e[3] for e in cand[y]))
        record('freq', by_freq, positives, t)
        record('freq_x_pred', by_freq_x, positives, t)
        record('fwd_eq_sum', by_fwd, positives, t)

        # ---- null: same model, features shuffled across candidates.  Any
        # recall above this is structure rather than candidate-set geometry.
        for seed in range(args.seeds):
            rng = random.Random(500 + seed)
            shuffled = keys[:]
            rng.shuffle(shuffled)
            record('NULL_random', shuffled, positives, t)

        # ---- ceiling: split month t's OBSERVATIONS in half.  Rank candidates
        # by how often they were seen in half A; score against the positives
        # that show up in half B.  Same month, so there is no forecasting
        # difficulty at all -- this bounds any cross-month method.
        for seed in range(args.seeds):
            rng = random.Random(1000 + seed)
            cnt_a, cnt_b = {}, {}
            for v, c in counts[t].items():
                ci = int(round(c))
                a = sum(1 for _ in range(min(ci, 400)) if rng.random() < 0.5)
                if ci > 400:
                    a = int(round(a * ci / 400.0))
                cnt_a[v], cnt_b[v] = a, c - a
            pos_b = {y for y in positives if cnt_b.get(y, 0) >= 1}
            if len(pos_b) < 5:
                continue
            ranked = sorted(keys, key=lambda y: (-cnt_a.get(y, 0),
                                                 -cand[y][0][3]))
            record('CEILING', ranked, pos_b, t)

        # ---- model: trained on months strictly before t
        train_X, train_y = [], []
        for tt in [m for m in months if m < t][-6:]:
            w2 = [m for m in months if tt - args.window <= m < tt]
            if not w2:
                continue
            pf2 = defaultdict(int)
            for m in w2:
                for v, c in counts[m].items():
                    pf2[v] += c
            preds2 = sorted(((v, c) for v, c in pf2.items() if c >= args.min_count),
                            key=lambda z: -z[1])[:150]
            sf2 = defaultdict(int)
            for m in w2:
                for v, c in counts[m].items():
                    for s in v:
                        sf2[s] += c
            subs2 = sorted(sf2.items(), key=lambda z: -z[1])[:150]
            last2 = w2[-1]
            sr2, pr2 = defaultdict(float), defaultdict(float)
            for v, c in counts[last2].items():
                for s in v:
                    sr2[s] += c
                    pr2[intern_rev[s].split(':')[1][1:-1]] += c
            ex2 = {v for v in first if first[v] < tt}
            c2 = build_candidates(preds2, subs2, ex2)
            new2 = {v for v, c in counts[tt].items()
                    if first.get(v) == tt and c >= args.min_count}
            pos2 = set(c2) & new2
            if not pos2:
                continue
            neg2 = [y for y in c2 if y not in pos2]
            rng = random.Random(7)
            neg2 = rng.sample(neg2, min(len(neg2), 20 * len(pos2)))
            for y in list(pos2) + neg2:
                train_X.append(features(c2[y], sr2, pr2, intern_rev))
                train_y.append(1.0 if y in pos2 else 0.0)

        if len(train_X) < 50:
            print(f'  {month_label(t)}: too little training data, model skipped',
                  file=sys.stderr)
            continue

        # one model per ablation variant, all sharing the same full-width
        # feature vectors computed above
        for label, mask in variants:
            tX = [[r[i] for i in mask] for r in train_X]
            for seed in range(args.seeds):
                w, b = fit_logistic(tX, train_y, seed)
                ranked = sorted(keys, key=lambda y: -score_logistic(
                    w, b, [feats[y][i] for i in mask]))
                record(label, ranked, positives, t)

        cov = len(positives) / len(new_this) if new_this else float('nan')
        print(f'  {month_label(t)}: {len(keys):,} candidates, '
              f'{len(positives)}/{len(new_this)} new variants reachable '
              f'(cov {cov:.3f}), {len(train_X)} training rows',
              file=sys.stderr)

    # ------------------------------------------------------------- report
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'appearance_eval.tsv')

    def mean_sd(v):
        v = [x for x in v if x == x]
        if not v:
            return float('nan'), float('nan')
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        return m, sd

    order = (['NULL_random', 'freq', 'freq_x_pred', 'fwd_eq_sum']
             + [lbl for lbl, _ in variants] + ['CEILING'])
    lines = ['method\tk\trecall_mean\trecall_sd\tn']
    print('\nrecall@k on new single-substitution variants\n', file=sys.stderr)
    print(f'{"method":<14}' + ''.join(f'{"@" + str(k):>16}' for k in KS),
          file=sys.stderr)
    for name in order:
        if name not in results:
            continue
        cells = []
        for k in KS:
            m, sd = mean_sd(results[name][k])
            cells.append(f'{m:.3f}±{sd:.3f}')
            lines.append(f'{name}\t{k}\t{m:.4f}\t{sd:.4f}\t{len(results[name][k])}')
        print(f'{name:<14}' + ''.join(f'{c:>16}' for c in cells), file=sys.stderr)

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nwrote {path}', file=sys.stderr)

    # ---- per-month, at the largest k.  Pooled means hide two different
    # things: seed variance (model instability) and month variance (whether
    # the effect is consistent or carried by a few months).
    K = KS[-1]
    pm_path = os.path.join(args.out, 'appearance_eval_per_month.tsv')
    all_months = sorted({m for name in order if name in per_month
                         for m in per_month[name][K]})
    with open(pm_path, 'w') as fh:
        fh.write('month\t' + '\t'.join(order) + '\n')
        for m in all_months:
            cells = []
            for name in order:
                v = per_month.get(name, {}).get(K, {}).get(m, [])
                v = [x for x in v if x == x]
                cells.append(f'{sum(v)/len(v):.4f}' if v else '')
            fh.write(month_label(m) + '\t' + '\t'.join(cells) + '\n')

    ref = 'fwd_eq_sum'
    print(f'\nper-month recall@{K}, and months where each method beats '
          f'{ref}:\n', file=sys.stderr)
    for name in order:
        if name not in per_month or name == ref:
            continue
        wins = tot = 0
        for m in all_months:
            a = per_month[name][K].get(m, [])
            b = per_month.get(ref, {}).get(K, {}).get(m, [])
            if not a or not b:
                continue
            tot += 1
            if sum(a) / len(a) > sum(b) / len(b):
                wins += 1
        if tot:
            print(f'  {name:<22} {wins}/{tot} months', file=sys.stderr)
    print(f'\nwrote {pm_path}', file=sys.stderr)

    print('\nreading it: if model does not exceed freq_x_pred, conditioning on '
          '\nthe predecessor adds nothing. If both sit near CEILING, the task is '
          '\nnear its limit and no model will do much better.', file=sys.stderr)


if __name__ == '__main__':
    main()
