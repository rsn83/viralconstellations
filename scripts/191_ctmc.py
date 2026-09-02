#!/usr/bin/env python3
"""
191 -- CTMC BASELINE: IS DIRECTIONAL CHANGE PREDICTABLE?

THE LADDER
----------
    null        freq(i changes)            no direction, no horizon
    CTMC        exp(Q_i * h)               direction + horizon, no temporal dynamics
    GRU+CTMC    exp(MLP(h_t) * h)          direction + horizon + temporal dynamics

This script establishes the CTMC rung. If it doesn't beat the null,
neither will the GRU, because there is no directional signal to learn.

WHAT IS A CTMC HERE
-------------------
For each variable position i, fit a rate matrix Q_i ∈ R^{21x21} from
observed residue transitions in training months:

    Q_i[r, r'] = count(i: r -> r') / time_in_r     r != r'
    Q_i[r, r]  = -sum_{r'!=r} Q_i[r, r']

Then the probability of transitioning from current residue r to r' in
h months is:

    P_i(h) = expm(Q_i * h)

The score for position i changing away from its current dominant residue
is 1 - P_i(h)[r_current, r_current].

WHAT THIS ADDS OVER THE NULL
-----------------------------
The null scores position i by how often it changed in training,
regardless of direction. The CTMC also knows WHICH residue it will
change to, and uses horizon h explicitly. If CTMC > null, directional
information and horizon conditioning help.

EVALUATION: identical to 189/190
    truth  = positions where dominant residue changed by >= thresh
    metric = recall@K: fraction of truth in top-K by predicted change prob

USAGE
    python scripts/191_ctmc.py \
        --events data/processed/events_v3.tsv \
        --vocab  data/processed/vocab_v3.tsv \
        --ladder scripts/171_ladder.py \
        --test-end 2025-02 --out results/ctmc.json

    # sweep horizons
    for h in 1 2 3; do
      python scripts/191_ctmc.py ... --horizon $h
    done

GIT
    git add scripts/191_ctmc.py
    git commit -m "191: CTMC baseline -- is directional residue change predictable"
    git push
"""

import argparse
import json
import importlib.util

import numpy as np
from scipy.linalg import expm


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def recall_at_k(scores, truth, Ks, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=np.float64)
    s = s + rng.uniform(-1e-12, 1e-12, size=s.shape)
    order = np.argsort(-s)
    rank = np.empty(len(s), dtype=np.int64)
    rank[order] = np.arange(len(s))
    hits = np.asarray(sorted(truth))
    return {K: float(np.mean(rank[hits] < K)) for K in Ks}


def fit_ctmc(emb, months, var_ix, n_aa=21, pseudo=0.1):
    """Fit per-position rate matrices from consecutive monthly transitions.

    emb:      {month: array(n_pos, 21)}  full position embeddings
    months:   ordered list of training months
    var_ix:   indices of variable positions

    Returns Q: array(P, 21, 21) where P = len(var_ix)
    """
    P = len(var_ix)
    # count matrix: C[i, r, r'] = weighted transitions r->r' at position i
    C = np.zeros((P, n_aa, n_aa), dtype=np.float64)
    T = np.zeros((P, n_aa),       dtype=np.float64)  # time in state r

    for a in range(len(months) - 1):
        E1 = emb.get(months[a])
        E2 = emb.get(months[a + 1])
        if E1 is None or E2 is None:
            continue
        dt = 1.0   # one month per step
        for j, ix in enumerate(var_ix):
            p1 = E1[ix]   # (21,) residue distribution at month a
            p2 = E2[ix]   # (21,) residue distribution at month a+1
            # treat as soft counts: transition mass r->r' = p1[r] * p2[r']
            # (assumes independence within the month, consistent with the
            # mean-field approximation used throughout 189/190)
            C[j] += np.outer(p1, p2)
            T[j] += p1 * dt

    # build rate matrix per position
    Q = np.zeros((P, n_aa, n_aa), dtype=np.float64)
    for j in range(P):
        for r in range(n_aa):
            denom = T[j, r] + pseudo
            for rp in range(n_aa):
                if rp != r:
                    Q[j, r, rp] = (C[j, r, rp] + pseudo / n_aa) / denom
            Q[j, r, r] = -Q[j, r, :].sum()
    return Q


def ctmc_change_prob(Q, E_now, h):
    """For each position, probability of changing away from current residue.

    Q:     (P, 21, 21) rate matrices
    E_now: (P, 21)     current residue distribution
    h:     horizon in months

    Returns scores: (P,) probability of change at each position
    """
    P = Q.shape[0]
    scores = np.zeros(P)
    for j in range(P):
        Ph = expm(Q[j] * h)           # (21, 21) transition matrix
        cur_dom = int(np.argmax(E_now[j]))
        # prob of staying at current dominant residue
        p_stay = Ph[cur_dom, cur_dom]
        scores[j] = 1.0 - p_stay
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",       default="data/processed/events_v3.tsv")
    ap.add_argument("--vocab",        default="data/processed/vocab_v3.tsv")
    ap.add_argument("--ladder",       default="scripts/171_ladder.py")
    ap.add_argument("--horizon",      type=int,   default=1)
    ap.add_argument("--change-thresh",type=float, default=0.02,
                    dest="change_thresh")
    ap.add_argument("--test-end",     default="2025-02")
    ap.add_argument("--pseudo",       type=float, default=0.1)
    ap.add_argument("--out",          default=None)
    a = ap.parse_args()

    KS = [5, 10, 20]

    L189 = load_mod("scripts/189_gru_residue.py", "gru189")
    L    = load_mod("scripts/171_ladder.py",       "ladder171")

    print("loading ...")
    monthly = L.load_events(a.events)
    months  = sorted(monthly)
    pos_res = L189.load_vocab(a.vocab)

    tr_end      = L.TRAIN_END[:7]
    all_train   = [m for m in months if m <= tr_end]
    test_months = [m for m in months if tr_end < m <= a.test_end]

    all_pos, wuhan, emb = L189.build_embeddings(
        monthly, pos_res, all_train + test_months)
    var_ix = L189.variable_positions(emb, all_train, a.change_thresh)
    P = len(var_ix)
    print(f"  {P} variable positions | h={a.horizon}")

    # historical change frequency (the null)
    hist_change = np.zeros(P)
    for i in range(len(all_train) - 1):
        E1 = emb.get(all_train[i])
        E2 = emb.get(all_train[i + 1])
        if E1 is not None and E2 is not None:
            hist_change += np.abs(E2 - E1)[var_ix].max(axis=1)
    hist_change /= max(len(all_train) - 1, 1)

    # fit CTMC on training months
    print("fitting CTMC ...")
    Q = fit_ctmc(emb, all_train, var_ix, pseudo=a.pseudo)
    print(f"  rate matrices fitted: {P} positions x 21x21")

    def get_E_var(m):
        e = emb.get(m)
        return e[var_ix] if e is not None else None

    def changed(E1, E2):
        return list(np.where(
            np.abs(E2 - E1).max(axis=1) >= a.change_thresh)[0])

    print(f"\n[eval] CTMC vs null  h={a.horizon}")
    print(f"  {'month':9s} {'n_ch':>5s} "
          + "".join(f"null@{K:2d} " for K in KS)
          + "".join(f"ctmc@{K:2d} " for K in KS))

    rows = []
    for m in test_months:
        t_ix = months.index(m)
        if t_ix + a.horizon >= len(months):
            break
        E_now = get_E_var(m)
        E_nxt = get_E_var(months[t_ix + a.horizon])
        if E_now is None or E_nxt is None:
            continue
        truth = changed(E_now, E_nxt)
        if not truth:
            continue

        ctmc_scores = ctmc_change_prob(Q, E_now, float(a.horizon))
        r_null = recall_at_k(hist_change,  truth, KS)
        r_ctmc = recall_at_k(ctmc_scores,  truth, KS)

        row = {"month": m, "n_changed": len(truth),
               "null": r_null, "ctmc": r_ctmc}
        rows.append(row)
        print(f"  {m:9s} {len(truth):5d} "
              + "".join(f"{r_null[K]:7.3f} " for K in KS)
              + "".join(f"{r_ctmc[K]:7.3f} " for K in KS))

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    avg_null = {K: float(np.mean([r["null"][K] for r in rows])) for K in KS}
    avg_ctmc = {K: float(np.mean([r["ctmc"][K] for r in rows])) for K in KS}
    print(f"\n  {'MEAN':9s} {'':5s} "
          + "".join(f"{avg_null[K]:7.3f} " for K in KS)
          + "".join(f"{avg_ctmc[K]:7.3f} " for K in KS))
    print(f"\n  CTMC over null:")
    for K in KS:
        g = avg_ctmc[K] - avg_null[K]
        print(f"    @{K:2d}  {g:+.4f}")
    print("\n  > 0  -> directional change is predictable; "
          "GRU+CTMC warranted.")
    print("  ~ 0  -> no directional signal; "
          "GRU will not improve on the null.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"horizon": a.horizon,
                       "change_thresh": a.change_thresh,
                       "P": P, "n_months": len(rows),
                       "avg_null": avg_null, "avg_ctmc": avg_ctmc,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
