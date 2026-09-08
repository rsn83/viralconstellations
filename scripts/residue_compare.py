#!/usr/bin/env python3
"""
Multi-horizon forecast comparison on full residue sequences.

Trains once at an anchor month T on everything up to T, then forecasts
T+1 .. T+h.  This is a genuine multi-step forecast: the earlier
candidate-ranking experiment refit every month, so recall could not decay
with horizon and horizon was not actually being tested.

REPRESENTATION
--------------
Every model receives the same thing: the spike residue string, in genomic
order, restricted to positions that vary in the training window.  The
reference supplies the residue at every position, so this is a genuine
sequence -- ordering and "what residue is here" are both preserved -- and
only literally constant columns are dropped.  Of 1273 positions, typically a
few hundred vary; the constant ones carry no information and would consume
the whole model.

MODELS
------
lstm        Autoregressive LSTM over the residue string.  Conditions each
            position on all previous ones, so co-occurrence is representable
            with no bottleneck.
hmm         K regimes over months, emission factorised across positions given
            the regime.  Co-occurrence passes only through a K-dimensional
            bottleneck.  The contrast with the LSTM is the point.
ctmc        Rates over (position, residue) edits, fit by the predecessor-sum
            likelihood with real elapsed time, plus a replicator growth term
            fit by log-linear regression on frequency trajectories.  Growth is
            required for fairness: the LSTM and HMM are trained on observed
            frequencies and absorb growth implicitly.
ctmc_nogrow Same with growth disabled, to isolate what growth buys.
persist     Copy month T.  Scores zero on arrivals by construction, which is
            the point -- it is not a competitor on that target.

ASYMMETRY, STATED
-----------------
The CTMC additionally conditions on the observed population at T; it steps
forward from where things actually are.  The LSTM and HMM must have learned
that from training data.  This is a different information set.  Denying the
CTMC the current population would mean denying it the object the forward
equation is defined on, so the asymmetry is kept and reported rather than
removed.

METRICS
-------
pop_ll      mean log q(y) over whole sequences observed at T+h
pos_ll      mean per-position log-likelihood (models that never reproduce a
            whole sequence still score here, so a floored pop_ll is
            interpretable rather than uninformative)
recall@k    of the variants first seen at T+h, how many are in the model's
            top-k predicted novel variants
prec@k      of the top-k predicted novel variants, how many really are new

Usage
-----
  zstdcat data/raw/metadata.tsv.zst | python3 residue_compare.py \
      --reference data/raw/spike_reference.fasta \
      --out results/residue --anchor 2022-12 --horizons 6
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from ctmc_full import CTMCFull
    HAVE_FULL = True
except ImportError:
    HAVE_FULL = False

try:
    from ctmc_neural import CTMCNeural
    HAVE_NEURAL = True
except ImportError:
    HAVE_NEURAL = False

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


AA = 'ACDEFGHIKLMNPQRSTVWY*-X'
AA_IDX = {c: i for i, c in enumerate(AA)}


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


def load_reference(path):
    seq = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith('>'):
                seq.append(line.strip())
    return ''.join(seq)


def parse_subs(field):
    """S:D614G -> (614, 'G').  Positions are 1-based."""
    out = []
    for tok in field.split(','):
        tok = tok.strip()
        if not tok.startswith('S:'):
            continue
        body = tok[2:]
        if len(body) < 3:
            continue
        alt = body[-1]
        pos = body[1:-1]
        if pos.isdigit() and alt in AA_IDX:
            out.append((int(pos), alt))
    return out


def parse_nonspike(field):
    """Non-spike substitutions: the observed genome background.

    This is the context the corollary requires -- read off the same column,
    with no Pango label and nothing inferred.  The sufficiency test measured
    exactly this dependence (JS 0.366 vs null 0.235)."""
    return tuple(sorted(t.strip() for t in field.split(',')
                        if t.strip() and not t.strip().startswith('S:')))


def stream(fh, cols, qc):
    """Returns counts[month][variant], bg_of[variant] -> non-spike tuple."""
    i_date, i_aa = cols['date'], cols['aaSubstitutions']
    i_qc = cols.get('QC_overall_status')
    ncol = len(cols)
    counts = defaultdict(lambda: defaultdict(int))
    bg_of = {}
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
        subs = parse_subs(f[i_aa])
        if subs:
            key = tuple(sorted(set(subs)))
            counts[mo][key] += 1
            if key not in bg_of:
                bg_of[key] = parse_nonspike(f[i_aa])
        if n % 1000000 == 0:
            print(f'  {n:,} rows', file=sys.stderr)
    print(f'  {n:,} rows', file=sys.stderr)
    return counts, bg_of


def variable_positions(counts, train_months, min_count):
    """Positions that vary in the training window.  Computed from training
    data only -- using the test months here would leak."""
    seen = defaultdict(int)
    for m in train_months:
        for v, c in counts[m].items():
            for pos, _ in v:
                seen[pos] += c
    return sorted(p for p, c in seen.items() if c >= min_count)


def to_string(variant, positions, reference):
    """Residue string over the variable positions, in genomic order."""
    d = dict(variant)
    return ''.join(d.get(p, reference[p - 1] if p <= len(reference) else 'X')
                   for p in positions)


# ------------------------------------------------------------------ ctmc

class CTMC:
    def __init__(self, growth=True, top_preds=800, top_subs=800, prior=0.5):
        self.growth, self.top_preds = growth, top_preds
        self.top_subs, self.prior = top_subs, prior

    def fit(self, counts, train, window=2):
        first = {}
        for m in train:
            for v in counts[m]:
                if v not in first or m < first[v]:
                    first[v] = m

        num, denom = defaultdict(float), 0.0
        for t in train:
            win = [m for m in train if t - window <= m < t]
            if not win:
                continue
            pf = defaultdict(float)
            for m in win:
                for v, c in counts[m].items():
                    pf[v] += c
            wset, mass = set(pf), sum(pf.values())
            denom += mass
            for y in counts[t]:
                if first.get(y) != t:
                    continue
                for s in y:
                    if tuple(v for v in y if v != s) in wset:
                        num[s] += 1.0
                        break
        self.rate = {s: (v + self.prior) / max(denom, 1.0)
                     for s, v in num.items()}

        self.f = {}
        if self.growth:
            recent = train[-6:]
            tot = {m: sum(counts[m].values()) for m in recent}
            pts = defaultdict(list)
            for i, m in enumerate(recent):
                for v, c in counts[m].items():
                    if c >= 3:
                        pts[v].append((i, math.log(c / max(tot[m], 1))))
            for v, ps in pts.items():
                if len(ps) < 3:
                    continue
                sl = np.polyfit([p[0] for p in ps], [p[1] for p in ps], 1)[0]
                self.f[v] = float(sl) * len(ps) / (len(ps) + 4.0)

        self.train, self.window = train, window
        return self

    def predict(self, counts, T, h):
        win = [m for m in self.train if T - self.window < m <= T]
        p = defaultdict(float)
        for m in win:
            for v, c in counts[m].items():
                p[v] += c
        tot = sum(p.values())
        if not tot:
            return {}
        p = {v: c / tot for v, c in p.items()}
        preds = sorted(p.items(), key=lambda z: -z[1])[:self.top_preds]
        subs = sorted(self.rate.items(), key=lambda z: -z[1])[:self.top_subs]
        lam = sum(r for _, r in subs)
        out = defaultdict(float)
        for x, px in preds:
            g = math.exp(self.f.get(x, 0.0) * h) if self.growth else 1.0
            out[x] += px * g * math.exp(-lam * h)
            xs = set(x)
            for s, r in subs:
                if s in xs:
                    continue
                out[tuple(sorted(xs | {s}))] += px * g * (1 - math.exp(-r * h))
        z = sum(out.values())
        return {v: c / z for v, c in out.items()} if z else {}


# ------------------------------------------------------------------- hmm

class TemporalHMM:
    def __init__(self, k=4, seed=0):
        self.k, self.seed = k, seed

    def fit(self, counts, train, positions, reference):
        self.positions, self.reference = positions, reference
        L, A = len(positions), len(AA)
        prof = []
        for m in train:
            P = np.full((L, A), 0.1)
            for v, c in counts[m].items():
                s = to_string(v, positions, reference)
                for i, ch in enumerate(s):
                    P[i, AA_IDX.get(ch, AA_IDX['X'])] += c
            prof.append(P / P.sum(1, keepdims=True))
        P = np.array(prof)
        flat = P.reshape(len(P), -1)

        rng = np.random.default_rng(self.seed)
        k = min(self.k, len(flat))
        cen = flat[rng.choice(len(flat), k, replace=False)]
        for _ in range(30):
            a = ((flat[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
            for j in range(k):
                if (a == j).any():
                    cen[j] = flat[a == j].mean(0)
        self.cen = cen.reshape(k, L, A)
        Tm = np.ones((k, k)) * 0.1
        for i in range(len(a) - 1):
            Tm[a[i], a[i + 1]] += 1
        self.trans = Tm / Tm.sum(1, keepdims=True)
        self.state = np.zeros(k)
        self.state[a[-1]] = 1.0
        return self

    def profile(self, h):
        st = self.state.copy()
        for _ in range(h):
            st = st @ self.trans
        P = np.tensordot(st, self.cen, axes=(0, 0))
        return P / P.sum(1, keepdims=True)

    def sample(self, h, m, rng):
        P = self.profile(h)
        out = defaultdict(int)
        cum = P.cumsum(1)
        for _ in range(m):
            u = rng.random((len(P), 1))
            idx = (cum < u).sum(1)
            v = tuple(sorted(
                (self.positions[i], AA[idx[i]])
                for i in range(len(P))
                if AA[idx[i]] != self.reference[self.positions[i] - 1]))
            out[v] += 1
        return out


# ------------------------------------------------------------------ lstm

class LSTMModel:
    """Autoregressive over the residue string.  Each position is conditioned
    on every previous one, so co-occurrence has no bottleneck -- the property
    the HMM lacks."""

    def __init__(self, hidden=192, epochs=8, batch=128, lr=2e-3, seed=0,
                 max_train=20000):
        self.hidden, self.epochs = hidden, epochs
        self.batch, self.lr, self.seed = batch, lr, seed
        self.max_train = max_train

    def fit(self, counts, train, positions, reference):
        torch.manual_seed(self.seed)
        self.positions, self.reference = positions, reference
        L, A = len(positions), len(AA)
        self.L, self.A = L, A

        rows, w = [], []
        for m in train:
            for v, c in counts[m].items():
                rows.append([AA_IDX.get(ch, AA_IDX['X'])
                             for ch in to_string(v, positions, reference)])
                w.append(c)
        if not rows:
            raise RuntimeError('no training sequences')
        if len(rows) > self.max_train:
            # sample proportional to abundance: the model should see the
            # population, not the list of distinct variants
            pr = np.array(w, float)
            pr /= pr.sum()
            keep = np.random.default_rng(self.seed).choice(
                len(rows), self.max_train, p=pr, replace=True)
            rows = [rows[i] for i in keep]
        X = torch.tensor(rows, dtype=torch.long)

        self.emb = nn.Embedding(A, 48)
        self.rnn = nn.LSTM(48, self.hidden, batch_first=True)
        self.head = nn.Linear(self.hidden, A)
        self.bos = nn.Parameter(torch.zeros(1, 1, 48))
        params = (list(self.emb.parameters()) + list(self.rnn.parameters())
                  + list(self.head.parameters()) + [self.bos])
        opt = torch.optim.Adam(params, lr=self.lr)

        for ep in range(self.epochs):
            perm = torch.randperm(len(X))
            tot = 0.0
            for i in range(0, len(X), self.batch):
                b = X[perm[i:i + self.batch]]
                e = self.emb(b[:, :-1])
                inp = torch.cat([self.bos.expand(len(b), 1, -1), e], 1)
                out, _ = self.rnn(inp)
                logits = self.head(out)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, A), b.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            print(f'    lstm epoch {ep + 1}/{self.epochs} '
                  f'loss {tot / len(X):.4f}', file=sys.stderr)
        return self

    @torch.no_grad()
    def logprob_matrix(self, seqs):
        """Per-position log-probabilities for a batch of index sequences."""
        X = torch.tensor(seqs, dtype=torch.long)
        e = self.emb(X[:, :-1])
        inp = torch.cat([self.bos.expand(len(X), 1, -1), e], 1)
        out, _ = self.rnn(inp)
        return torch.log_softmax(self.head(out), -1), X

    @torch.no_grad()
    def sample(self, h, m, rng):
        # horizon enters only through how many samples are drawn from the
        # same fitted distribution: an autoregressive model over sequences
        # has no time argument.  This is a real limitation of the baseline
        # and is reported rather than engineered around.
        out = defaultdict(int)
        B = 512
        done = 0
        while done < m:
            n = min(B, m - done)
            hstate = None
            cur = self.bos.expand(n, 1, -1)
            idx = torch.zeros(n, self.L, dtype=torch.long)
            for i in range(self.L):
                o, hstate = self.rnn(cur, hstate)
                p = torch.softmax(self.head(o[:, -1]), -1)
                s = torch.multinomial(p, 1).squeeze(1)
                idx[:, i] = s
                cur = self.emb(s).unsqueeze(1)
            for r in idx.tolist():
                v = tuple(sorted(
                    (self.positions[i], AA[r[i]])
                    for i in range(self.L)
                    if AA[r[i]] != self.reference[self.positions[i] - 1]))
                out[v] += 1
            done += n
        return out


# ------------------------------------------------------------------ eval

def to_dist(sc):
    z = sum(sc.values())
    return {v: c / z for v, c in sc.items()} if z else {}


def pop_loglik(dist, observed, floor=1e-12):
    n = sum(observed.values())
    if not n:
        return float('nan')
    return sum(c * math.log(max(dist.get(v, 0.0), floor))
               for v, c in observed.items()) / n


def pos_loglik(dist, observed, positions, reference, floor=1e-9):
    """Per-position log-likelihood under the model's marginal profile.  A
    model that never reproduces a whole sequence still scores here, so a
    floored pop_ll stays interpretable."""
    L = len(positions)
    P = np.full((L, len(AA)), floor)
    for v, p in dist.items():
        s = to_string(v, positions, reference)
        for i, ch in enumerate(s):
            P[i, AA_IDX.get(ch, AA_IDX['X'])] += p
    P /= P.sum(1, keepdims=True)
    tot = n = 0.0
    for v, c in observed.items():
        s = to_string(v, positions, reference)
        for i, ch in enumerate(s):
            tot += c * math.log(P[i, AA_IDX.get(ch, AA_IDX['X'])])
        n += c * L
    return tot / n if n else float('nan')


def recall_prec(dist, observed_new, seen_before, k, observed=None):
    """Unweighted and mass-weighted recall, plus precision.

    Unweighted counts a singleton and a future dominant variant the same, so a
    model can score well by finding many trivial arrivals and missing every
    one that mattered.  The weighted version fixes that by weighting each hit
    by the variant's abundance at T+h, which is closer to the population-level
    question.  Where the two diverge sharply, that divergence is itself the
    result."""
    novel = sorted(((v, p) for v, p in dist.items() if v not in seen_before),
                   key=lambda z: -z[1])[:k]
    top = {v for v, _ in novel}
    if not top or not observed_new:
        return 0.0, 0.0, 0.0
    hit = top & observed_new
    rec = len(hit) / len(observed_new)
    prec = len(hit) / len(top)

    wrec = float('nan')
    if observed is not None:
        mass = sum(observed.get(v, 0) for v in observed_new)
        if mass:
            wrec = sum(observed.get(v, 0) for v in hit) / mass
    return rec, prec, wrec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', default='.')
    ap.add_argument('--anchor', default='2022-12')
    ap.add_argument('--horizons', type=int, default=6)
    ap.add_argument('--train-months', type=int, default=12)
    ap.add_argument('--window', type=int, default=2)
    ap.add_argument('--min-pos-count', type=int, default=50)
    ap.add_argument('--sample-m', type=int, default=20000)
    ap.add_argument('--topk', type=int, default=500)
    ap.add_argument('--hmm-k', type=int, default=4)
    ap.add_argument('--lstm-epochs', type=int, default=8)
    ap.add_argument('--lstm-max-train', type=int, default=20000)
    ap.add_argument('--no-lstm', action='store_true')
    ap.add_argument('--em', action='store_true',
                    help='also run the tabular EM variants (off by default: '
                         'the M-step step size is not stable at large arrival '
                         'counts and em_spike diverged on real data)')
    ap.add_argument('--full-radius', type=int, default=1,
                    help='predecessor radius for the EM model (1 or 2)')
    ap.add_argument('--em-iters', type=int, default=5)
    ap.add_argument('--steps-per-month', type=int, default=2)
    ap.add_argument('--only-ctmc', action='store_true',
                    help='skip hmm and lstm; run only the ctmc family')
    ap.add_argument('--neural', action='store_true',
                    help='add the neural rate model (needs torch)')
    ap.add_argument('--neural-epochs', type=int, default=25)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--qc', action='store_true', default=True)
    ap.add_argument('--no-qc', dest='qc', action='store_false')
    args = ap.parse_args()

    reference = load_reference(args.reference)
    print(f'reference: {len(reference)} residues', file=sys.stderr)

    header = sys.stdin.readline().rstrip('\n').split('\t')
    cols = {n: k for k, n in enumerate(header)}
    if 'date' not in cols or 'aaSubstitutions' not in cols:
        sys.exit('missing required columns')

    print('streaming metadata', file=sys.stderr)
    counts, bg_of = stream(sys.stdin, cols, args.qc)
    months = sorted(counts)

    T = month_index(args.anchor + '-15')
    if T not in counts:
        sys.exit(f'anchor {args.anchor} not in data')
    train = [m for m in months if m <= T][-args.train_months:]

    positions = variable_positions(counts, train, args.min_pos_count)
    print(f'\nanchor {month_label(T)}; train '
          f'{month_label(train[0])}..{month_label(train[-1])}', file=sys.stderr)
    print(f'variable positions: {len(positions)} of {len(reference)}',
          file=sys.stderr)

    seen_before = set()
    for m in [x for x in months if x <= T]:
        seen_before.update(counts[m])

    print('\nfitting ctmc (+growth)', file=sys.stderr)
    ctmc = CTMC(growth=True).fit(counts, train, args.window)
    print('fitting ctmc (no growth)', file=sys.stderr)
    ctmc0 = CTMC(growth=False).fit(counts, train, args.window)
    hmm = None
    if not args.only_ctmc:
        print('fitting hmm', file=sys.stderr)
        hmm = TemporalHMM(k=args.hmm_k, seed=args.seed).fit(
            counts, train, positions, reference)

    lstm = None
    if not args.no_lstm and not args.only_ctmc:
        if not HAVE_TORCH:
            print('torch missing; skipping lstm', file=sys.stderr)
        else:
            print('fitting lstm', file=sys.stderr)
            lstm = LSTMModel(epochs=args.lstm_epochs, seed=args.seed,
                             max_train=args.lstm_max_train).fit(
                counts, train, positions, reference)

    # EM-fitted CTMC, three nested conditionings.  The gap between them is
    # the corollary as a number: `spike` minus `marginal` is what the
    # predecessor marginalization buys, `genome` minus `spike` is what the
    # non-spike background is worth.
    sub_vocab = defaultdict(int)
    for m in train[-6:]:
        for v, c in counts[m].items():
            for s in v:
                sub_vocab[s] += c
    self_sv = [s for s, _ in sorted(sub_vocab.items(),
                                    key=lambda z: -z[1])[:400]]

    full = {}
    if HAVE_FULL and args.em:
        sv = self_sv
        for cond in ('marginal', 'spike', 'genome'):
            print(f'fitting ctmc_em[{cond}]', file=sys.stderr)
            try:
                mdl = CTMCFull(conditioning=cond, radius=args.full_radius,
                               em_iters=args.em_iters,
                               steps_per_month=args.steps_per_month,
                               seed=args.seed)
                mdl.set_vocab(sv)
                mdl.fit(counts, bg_of, train)
                full['em_' + cond] = mdl
            except Exception as e:
                print(f'  ctmc_em[{cond}] failed: {e}', file=sys.stderr)

    if args.neural and HAVE_NEURAL:
        print('fitting ctmc_neural', file=sys.stderr)
        try:
            nm = CTMCNeural(radius=args.full_radius,
                            epochs=args.neural_epochs,
                            steps_per_month=args.steps_per_month,
                            device=args.device, seed=args.seed)
            nm.fit(counts, bg_of, train, positions, reference, self_sv)
            full['neural'] = nm
        except Exception as e:
            print(f'  ctmc_neural failed: {e}', file=sys.stderr)
    elif args.neural:
        print('torch missing; skipping neural', file=sys.stderr)

    persist = to_dist(dict(counts[T]))
    rng = np.random.default_rng(args.seed)

    rows = []
    for h in range(1, args.horizons + 1):
        if T + h not in counts:
            continue
        observed = dict(counts[T + h])
        observed_new = {v for v in observed if v not in seen_before}

        dists = {
            'persist': persist,
            'ctmc_nogrow': ctmc0.predict(counts, T, h),
            'ctmc': ctmc.predict(counts, T, h),
        }
        if hmm is not None:
            dists['hmm'] = to_dist(hmm.sample(h, args.sample_m, rng))
        if lstm is not None:
            dists['lstm'] = to_dist(lstm.sample(h, args.sample_m, rng))
        for nm, mdl in full.items():
            d, disc = mdl.predict(counts, T, h)
            dists[nm] = d
            if disc > 1e-6:
                print(f'    {nm} h={h}: discarded mass {disc:.4f}',
                      file=sys.stderr)

        for nm, d in dists.items():
            pl = pop_loglik(d, observed)
            ps = pos_loglik(d, observed, positions, reference)
            rc, pr, wrc = recall_prec(d, observed_new, seen_before,
                                      args.topk, observed)
            rows.append((h, nm, pl, ps, rc, wrc, pr, len(observed_new)))
        print(f'  h={h}: {len(observed_new)} novel variants observed',
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'residue_compare.tsv')
    with open(path, 'w') as fh:
        fh.write('horizon\tmodel\tpop_ll\tpos_ll\trecall\trecall_weighted'
                 '\tprecision\tn_novel\n')
        for r in rows:
            fh.write('%d\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % r)

    order = ['persist', 'hmm', 'lstm', 'ctmc_nogrow', 'ctmc',
             'em_marginal', 'em_spike', 'em_genome', 'neural']
    hs = sorted({r[0] for r in rows})
    tbl = {(r[1], r[0]): r for r in rows}

    for lbl, col in [('population log-likelihood (whole sequence)', 2),
                     ('per-position log-likelihood', 3),
                     (f'novel recall@{args.topk} (unweighted)', 4),
                     (f'novel recall@{args.topk} (mass-weighted)', 5),
                     (f'novel precision@{args.topk}', 6)]:
        print(f'\n{lbl}\n', file=sys.stderr)
        print(f'{"model":<14}' + ''.join(f'{"h=" + str(h):>11}' for h in hs),
              file=sys.stderr)
        for nm in order:
            if not any((nm, h) in tbl for h in hs):
                continue
            cells = ''.join(
                f'{tbl[(nm, h)][col]:>11.4f}' if (nm, h) in tbl else f'{"":>11}'
                for h in hs)
            print(f'{nm:<14}{cells}', file=sys.stderr)

    print(f'\nwrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
