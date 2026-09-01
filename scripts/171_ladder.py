#!/usr/bin/env python3
"""
Predictability ladder for new-constellation forecasting.

Measures how much each order of structure contributes to predicting which NEW
mutation constellations appear at T+h, given the population at T.

    RUNG 0  marginal      p(S u {D}) ~ mass(S) * freq(D)
    RUNG 1  pairwise      + lift(D, members of S)
    RUNG 2  background    + attachment rate of D to backgrounds similar to S

Reported as held-out log-loss over observed new constellations, ranked against
the full radius-1 candidate pool. Rung differences are the contributions.

    gap(0->1) = pairwise co-occurrence structure
    gap(1->2) = BACKGROUND STRUCTURE   <- the project hypothesis

Every estimate is fit on the training window only and scored on the test
window. Plug-in conditional entropy is deliberately NOT used: it is biased
downward on sparse conditionals and would manufacture a background effect.

Usage:
    python ladder.py --events data/processed/events_v3.tsv --seed 0
    for s in 0 1 2 3 4; do python ladder.py --seed $s --out res_$s.json; done
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG -- edit TRAIN_END / TEST_END and the parser to match your file
# ----------------------------------------------------------------------------

TRAIN_END = "2024-06-17"
TEST_END = "2026-05-14"

# Minimum count for a constellation to be considered "present" in a month.
# Guards against singleton sequencing artefacts being called new variants.
MIN_COUNT = 5

# Jaccard threshold for "similar background" backoff in rung 2.
JACCARD_TAU = 0.5

# Laplace smoothing added to every count-based probability.
ALPHA = 0.5


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------

def parse_variant(field):
    """Parse the variant field into a frozenset of mutation tokens.

    ASSUMPTION: mutations are comma- or pipe-separated in one column.
    If events_v3.tsv stores variants some other way (an ID that indexes into
    vocab_v3.tsv, say), replace this function -- it is the only place the
    file format is assumed.
    """
    field = field.strip()
    for sep in (",", "|", ";", " "):
        if sep in field:
            toks = [t.strip() for t in field.split(sep) if t.strip()]
            if len(toks) > 1:
                return frozenset(toks)
    return frozenset([field]) if field else frozenset()


def load_events(path):
    """Return {month: {constellation: count}} keyed by 'YYYY-MM'."""
    monthly = defaultdict(Counter)
    n_rows = 0
    with open(path) as fh:
        first = fh.readline()
        # skip a header row if present
        if not first[:4].isdigit():
            pass
        else:
            fh.seek(0)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            date_s, var_s, cnt_s = parts[0], parts[1], parts[2]
            try:
                cnt = int(float(cnt_s))
            except ValueError:
                continue
            S = parse_variant(var_s)
            if not S:
                continue
            monthly[date_s[:7]][S] += cnt
            n_rows += 1
    print(f"  loaded {n_rows:,} rows over {len(monthly)} months")
    return monthly


def population(month_counter, min_count=MIN_COUNT):
    """Normalised mass over constellations present this month."""
    kept = {S: c for S, c in month_counter.items() if c >= min_count}
    tot = sum(kept.values())
    if tot == 0:
        return {}
    return {S: c / tot for S, c in kept.items()}


# ----------------------------------------------------------------------------
# TEST 1 -- does exclusion / reinforcement structure exist at all?
# ----------------------------------------------------------------------------

def test_pairwise_structure(pops, months, top_k=200):
    """Compare observed pair co-occurrence to product of marginals.

    If lift ~ 1 everywhere, marginals determine the joint and autoregression
    buys nothing. This is the cheapest possible check and it gates everything
    downstream.
    """
    marg = Counter()
    pair = Counter()
    for m in months:
        for S, w in pops.get(m, {}).items():
            Sl = sorted(S)
            for i in Sl:
                marg[i] += w
            for a in range(len(Sl)):
                for b in range(a + 1, len(Sl)):
                    pair[(Sl[a], Sl[b])] += w
    n = len(months)
    if n == 0:
        return {}
    for d in (marg, pair):
        for k in d:
            d[k] /= n

    common = [m for m, _ in marg.most_common(top_k)]
    lifts, excl, reinf = [], 0, 0
    for a in range(len(common)):
        for b in range(a + 1, len(common)):
            i, j = sorted((common[a], common[b]))
            pi, pj = marg[i], marg[j]
            if pi < 1e-4 or pj < 1e-4:
                continue
            obs = pair.get((i, j), 0.0)
            exp = pi * pj
            lift = (obs + 1e-9) / (exp + 1e-9)
            lifts.append(lift)
            if lift < 0.1:
                excl += 1
            elif lift > 10:
                reinf += 1
    lifts = np.array(lifts)
    return {
        "n_pairs": int(len(lifts)),
        "median_lift": float(np.median(lifts)) if len(lifts) else None,
        "frac_near_independent": float(np.mean((lifts > 0.5) & (lifts < 2.0)))
        if len(lifts) else None,
        "n_strong_exclusion": int(excl),
        "n_strong_reinforcement": int(reinf),
    }


# ----------------------------------------------------------------------------
# CANDIDATES AND OBSERVED POSITIVES
# ----------------------------------------------------------------------------

def radius1_candidates(pop_T, vocab, max_backgrounds=300):
    """All (background, added mutation) pairs at Hamming radius 1.

    This is the 163 support: covered ~ 0.999 at h=1. Candidates are PAIRS,
    not sets, because the same set may be reachable from several backgrounds
    and ancestry is latent -- we never claim which parent produced it.
    """
    bgs = sorted(pop_T.items(), key=lambda kv: -kv[1])[:max_backgrounds]
    cands = []
    for S, w in bgs:
        for D in vocab:
            if D not in S:
                cands.append((S, D, w))
    return cands


def new_constellations(pop_T, pop_next, seen_ever):
    """Constellations present at T+h that were never present at or before T."""
    return {S: w for S, w in pop_next.items()
            if S not in seen_ever and S not in pop_T}


# ----------------------------------------------------------------------------
# THE THREE RUNGS
# ----------------------------------------------------------------------------

class Rungs:
    """Each rung scores a (background S, addition D) pair.

    Fit on train months only. Scored on test months only.
    """

    def __init__(self, pops, train_months):
        self.marg = Counter()          # freq(D)
        self.pair = Counter()          # co-occurrence(D, i)
        self.attach = defaultdict(Counter)   # background -> Counter(D added)
        self.bg_seen = Counter()       # how often each background was a parent
        self._fit(pops, train_months)

    def _fit(self, pops, train_months):
        n = max(len(train_months), 1)
        for m in train_months:
            for S, w in pops.get(m, {}).items():
                Sl = sorted(S)
                for i in Sl:
                    self.marg[i] += w
                for a in range(len(Sl)):
                    for b in range(len(Sl)):
                        if a != b:
                            self.pair[(Sl[a], Sl[b])] += w
        for k in self.marg:
            self.marg[k] /= n
        for k in self.pair:
            self.pair[k] /= n

        # attachment events: a set at month t+1 that is a radius-1 extension
        # of a set present at month t. NOTE: this is a REACHABILITY relation,
        # not an ancestry claim -- several backgrounds may qualify and we
        # credit all of them.
        for a in range(len(train_months) - 1):
            pT, pN = pops.get(train_months[a], {}), pops.get(train_months[a + 1], {})
            if not pT or not pN:
                continue
            for Sn in pN:
                for S in pT:
                    if len(Sn) == len(S) + 1 and S < Sn:
                        D = next(iter(Sn - S))
                        self.attach[S][D] += 1
                        self.bg_seen[S] += 1

    # -- rung 0 : marginal / frequency-matched null ---------------------------
    def score0(self, S, D, w):
        return math.log(w + 1e-12) + math.log(self.marg.get(D, 0.0) + 1e-9)

    # -- rung 1 : + pairwise lift --------------------------------------------
    def score1(self, S, D, w):
        pD = self.marg.get(D, 0.0) + 1e-9
        ll = 0.0
        for i in S:
            pi = self.marg.get(i, 0.0) + 1e-9
            obs = self.pair.get((D, i), 0.0)
            ll += math.log((obs + 1e-9) / (pD * pi) + 1e-9)
        # mean, so long constellations are not penalised by size alone
        ll = ll / max(len(S), 1)
        return math.log(w + 1e-12) + math.log(pD) + ll

    # -- rung 2 : + background-specific attachment ---------------------------
    def score2(self, S, D, w, tau=JACCARD_TAU):
        """Attachment rate of D to backgrounds SIMILAR to S.

        Backoff by Jaccard similarity, because most test backgrounds were
        never seen in training. If this rung does not beat rung 1, background
        identity carries no information beyond its members' marginals -- which
        is the hypothesis failing.
        """
        num = den = 0.0
        for Sbg, cnts in self.attach.items():
            inter = len(S & Sbg)
            if inter == 0:
                continue
            j = inter / len(S | Sbg)
            if j < tau:
                continue
            num += j * cnts.get(D, 0)
            den += j * self.bg_seen.get(Sbg, 0)
        if den < 1.0:
            return self.score1(S, D, w)     # back off cleanly
        rate = (num + ALPHA) / (den + ALPHA * len(self.marg))
        return math.log(w + 1e-12) + math.log(rate + 1e-12)


def logloss(scores, positive_idx):
    """Normalised negative log-likelihood of the true items under the scores."""
    if not positive_idx:
        return None
    s = np.array(scores, dtype=np.float64)
    s -= s.max()
    logZ = math.log(np.exp(s).sum() + 1e-300)
    return float(np.mean([-(s[i] - logZ) for i in positive_idx]))


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--horizon", type=int, default=1, help="months ahead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("loading ...")
    monthly = load_events(args.events)
    months = sorted(monthly)
    pops = {m: population(monthly[m]) for m in months}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)

    tr_end, te_end = TRAIN_END[:7], TEST_END[:7]
    train_months = [m for m in months if m <= tr_end]
    test_months = [m for m in months if tr_end < m <= te_end]
    print(f"  train {len(train_months)} months | test {len(test_months)} months")

    vocab = sorted({i for m in train_months for S in pops[m] for i in S})
    print(f"  vocab {len(vocab):,} mutations")

    # ---- TEST 1 -------------------------------------------------------------
    print("\n[test 1] pairwise structure (train window)")
    t1 = test_pairwise_structure(pops, train_months)
    for k, v in t1.items():
        print(f"    {k:28s} {v}")
    print("    -> if frac_near_independent ~ 1.0, marginals determine the")
    print("       joint and autoregression buys nothing. STOP HERE if so.")

    # ---- FIT ----------------------------------------------------------------
    print("\nfitting rungs on train ...")
    R = Rungs(pops, train_months)
    print(f"  {len(R.attach):,} backgrounds with observed attachments")

    # ---- LADDER ON TEST -----------------------------------------------------
    print(f"\n[ladder] held-out, horizon={args.horizon}m")
    seen_ever = set()
    for m in train_months:
        seen_ever |= set(pops[m])

    rows = []
    for a, m in enumerate(test_months):
        nxt_i = months.index(m) + args.horizon
        if nxt_i >= len(months):
            break
        pT, pN = pops[m], pops[months[nxt_i]]
        new = new_constellations(pT, pN, seen_ever)
        seen_ever |= set(pT)
        if not new:
            continue

        cands = radius1_candidates(pT, vocab, args.max_bg)
        if not cands:
            continue
        index = {(S, D): k for k, (S, D, _) in enumerate(cands)}

        pos = []
        for Sn in new:
            for (S, D, _) in cands:
                if len(Sn) == len(S) + 1 and S < Sn and D in Sn:
                    pos.append(index[(S, D)])
        if not pos:
            continue

        s0 = [R.score0(S, D, w) for S, D, w in cands]
        s1 = [R.score1(S, D, w) for S, D, w in cands]
        s2 = [R.score2(S, D, w) for S, D, w in cands]

        rows.append({
            "month": m,
            "n_new": len(new),
            "n_covered": len(set(pos)),
            "n_cand": len(cands),
            "r0": logloss(s0, pos),
            "r1": logloss(s1, pos),
            "r2": logloss(s2, pos),
        })

    if not rows:
        print("  NO EVALUABLE MONTHS -- check MIN_COUNT and the variant parser")
        return

    r0 = float(np.mean([r["r0"] for r in rows]))
    r1 = float(np.mean([r["r1"] for r in rows]))
    r2 = float(np.mean([r["r2"] for r in rows]))
    cov = float(np.mean([r["n_covered"] / max(r["n_new"], 1) for r in rows]))

    print(f"\n  months evaluated     {len(rows)}")
    print(f"  radius-1 coverage    {cov:.3f}")
    print(f"\n  RUNG 0  marginal     {r0:.4f} nats")
    print(f"  RUNG 1  pairwise     {r1:.4f} nats   gap {r0 - r1:+.4f}")
    print(f"  RUNG 2  background   {r2:.4f} nats   gap {r1 - r2:+.4f}  <-- HYPOTHESIS")
    print(f"\n  ceiling (all mass on covered truth): {-math.log(cov + 1e-12):.4f}")
    print("\n  gap(1->2) <= 0  =>  background structure adds nothing beyond")
    print("  its members' pairwise marginals, in this representation.")

    out = {
        "seed": args.seed, "horizon": args.horizon,
        "test1_pairwise": t1,
        "rung0": r0, "rung1": r1, "rung2": r2,
        "gap_0_1": r0 - r1, "gap_1_2": r1 - r2,
        "coverage": cov, "n_months": len(rows), "per_month": rows,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
