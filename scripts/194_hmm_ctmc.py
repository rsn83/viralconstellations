#!/usr/bin/env python3
"""
194 -- HMM WITH SWITCHING CTMC FOR RESIDUE SUBSTITUTION

THE MODEL
---------
A Hidden Markov Model where:
    hidden state z_t ∈ {0, 1, ..., K-1}    regime (e.g. heterogeneous/clonal)
    observation E_t ∈ R^{P x 21}            monthly residue distributions

Transition: z_{t+1} | z_t ~ Categorical(A[z_t])     K x K transition matrix
Emission:   E_t | z_t ~ per-regime CTMC rate matrix Q_{z_t}

The CTMC emission means: given regime z, position i changes at rate Q_{z}[i, r, r'].
The predicted change probability at horizon h:

    p(position i changes | z_t, h) = 1 - exp(Q_{z_t}[i] * h)[r_current, r_current]

TRAINING
--------
Forward-backward (Baum-Welch EM) on the sequence of monthly embeddings.
Fit K=2 regimes (heterogeneous and clonal, per 180's finding of two regimes).

EVALUATION
----------
Same as 191/193: recall@K on changed positions.
At each test month, infer the most likely regime (Viterbi), use that
regime's Q matrix to predict which positions change.

COMPARISON TABLE
----------------
    null      0.276 @20    historical frequency
    CTMC      0.448 @20    single-regime per-position rates
    GRU       0.437 @20    temporal, independent positions
    DFM       0.671 @20    joint rates, no temporal
    DFM+GRU   0.810 @20    joint rates + temporal
    HMM-CTMC  ?     @20    switching regimes + per-position rates

If HMM-CTMC > CTMC: regime switching adds over a single set of rates.
If HMM-CTMC ~ CTMC: one regime is sufficient (consistent with 180's
  finding that the clonal regime dominates the test window).

USAGE
    python scripts/194_hmm_ctmc.py \
        --events data/processed/events_v3.tsv \
        --vocab  data/processed/vocab_v3.tsv \
        --ladder scripts/171_ladder.py \
        --n-regimes 2 --test-end 2025-02 \
        --out results/hmm_ctmc.json

GIT
    git add scripts/194_hmm_ctmc.py
    git commit -m "194: HMM with switching CTMC regimes vs DFM+GRU"
    git push
"""

import argparse
import importlib.util
import json

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp


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


# ----------------------------------------------------------------------------
# CTMC PER REGIME
# ----------------------------------------------------------------------------

def fit_regime_ctmc(emb_list, n_aa=21, pseudo=0.1):
    """Fit a per-position rate matrix Q from a list of monthly embeddings.

    emb_list: list of (P, 21) arrays assigned to this regime
    Returns Q: (P, 21, 21)
    """
    if len(emb_list) < 2:
        return None
    P = emb_list[0].shape[0]
    C = np.zeros((P, n_aa, n_aa))
    T = np.zeros((P, n_aa))
    for a in range(len(emb_list) - 1):
        E0, E1 = emb_list[a], emb_list[a + 1]
        for j in range(P):
            C[j] += np.outer(E0[j], E1[j])
            T[j] += E0[j]
    Q = np.zeros((P, n_aa, n_aa))
    for j in range(P):
        for r in range(n_aa):
            denom = T[j, r] + pseudo
            for rp in range(n_aa):
                if rp != r:
                    Q[j, r, rp] = (C[j, r, rp] + pseudo / n_aa) / denom
            Q[j, r, r] = -Q[j, r, :].sum()
    return Q


def ctmc_change_scores(Q, E_now, h=1.0):
    """Probability of changing away from current dominant residue."""
    P = Q.shape[0]
    scores = np.zeros(P)
    for j in range(P):
        Ph = expm(Q[j] * h)
        cur = int(np.argmax(E_now[j]))
        scores[j] = 1.0 - Ph[cur, cur]
    return scores


# ----------------------------------------------------------------------------
# HMM: FORWARD-BACKWARD + VITERBI
# ----------------------------------------------------------------------------

class SwitchingCTMC:
    """HMM where each hidden state is a CTMC regime.

    Parameters:
        pi:   (K,) initial state distribution
        A:    (K, K) transition matrix
        Q:    (K, P, 21, 21) per-regime rate matrices (learned by EM)
    """

    def __init__(self, K, P, n_aa=21):
        self.K = K
        self.P = P
        self.n_aa = n_aa
        # initialise randomly
        rng = np.random.default_rng(0)
        self.pi = np.ones(K) / K
        A = rng.dirichlet(np.ones(K) * 5, size=K)
        self.A = A
        self.Q = np.zeros((K, P, n_aa, n_aa))

    def emission_logp(self, E_prev, E_curr, z, h=1.0):
        """Log p(E_curr | E_prev, regime z, h) under the CTMC.

        Approximation: treat each position's dominant residue transition
        as a multinomial draw from exp(Q_z[j] * h)[r_prev, :].
        """
        if self.Q is None or np.all(self.Q[z] == 0):
            return 0.0
        ll = 0.0
        for j in range(self.P):
            Ph = expm(self.Q[z, j] * h)
            r_prev = int(np.argmax(E_prev[j]))
            # soft target: weighted by current distribution
            p_next = Ph[r_prev, :]
            p_next = np.clip(p_next, 1e-9, 1.0)
            p_next /= p_next.sum()
            ll += float(E_curr[j] @ np.log(p_next))
        return ll

    def forward(self, emb_seq):
        """Forward algorithm. Returns log alpha (T, K)."""
        T = len(emb_seq)
        log_alpha = np.full((T, self.K), -np.inf)
        log_alpha[0] = np.log(self.pi + 1e-12)
        for t in range(1, T):
            for k in range(self.K):
                emit = self.emission_logp(emb_seq[t-1], emb_seq[t], k)
                log_alpha[t, k] = (logsumexp(log_alpha[t-1] + np.log(self.A[:, k] + 1e-12))
                                   + emit)
        return log_alpha

    def backward(self, emb_seq):
        """Backward algorithm. Returns log beta (T, K)."""
        T = len(emb_seq)
        log_beta = np.zeros((T, self.K))
        for t in range(T - 2, -1, -1):
            for k in range(self.K):
                vals = []
                for kp in range(self.K):
                    emit = self.emission_logp(emb_seq[t], emb_seq[t+1], kp)
                    vals.append(np.log(self.A[k, kp] + 1e-12) + emit
                                + log_beta[t+1, kp])
                log_beta[t, k] = logsumexp(vals)
        return log_beta

    def viterbi(self, emb_seq):
        """Viterbi decoding. Returns most likely state sequence."""
        T = len(emb_seq)
        delta = np.full((T, self.K), -np.inf)
        psi = np.zeros((T, self.K), dtype=int)
        delta[0] = np.log(self.pi + 1e-12)
        for t in range(1, T):
            for k in range(self.K):
                emit = self.emission_logp(emb_seq[t-1], emb_seq[t], k)
                vals = delta[t-1] + np.log(self.A[:, k] + 1e-12)
                psi[t, k] = np.argmax(vals)
                delta[t, k] = vals[psi[t, k]] + emit
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def em_step(self, emb_seq):
        """One Baum-Welch EM step. Updates pi, A; returns regime assignments."""
        T = len(emb_seq)
        log_alpha = self.forward(emb_seq)
        log_beta  = self.backward(emb_seq)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # update pi
        self.pi = gamma[0] / gamma[0].sum()

        # update A
        xi_sum = np.zeros((self.K, self.K))
        for t in range(T - 1):
            for k in range(self.K):
                for kp in range(self.K):
                    emit = self.emission_logp(emb_seq[t], emb_seq[t+1], kp)
                    xi_sum[k, kp] += np.exp(
                        log_alpha[t, k]
                        + np.log(self.A[k, kp] + 1e-12)
                        + emit
                        + log_beta[t+1, kp]
                        - logsumexp(log_alpha[-1])
                    )
        self.A = xi_sum / xi_sum.sum(axis=1, keepdims=True).clip(min=1e-12)

        # assign each month to its most likely regime
        assignments = np.argmax(gamma, axis=1)
        return assignments

    def fit(self, emb_seq, var_ix, n_iter=10, pseudo=0.1):
        """Run EM, updating Q after each assignment."""
        for it in range(n_iter):
            assignments = self.em_step(emb_seq)
            # refit Q per regime
            for k in range(self.K):
                idx = np.where(assignments == k)[0]
                regime_embs = [emb_seq[i] for i in idx if i + 1 < len(emb_seq)]
                next_embs   = [emb_seq[i+1] for i in idx if i + 1 < len(emb_seq)]
                if not regime_embs:
                    continue
                Q_k = fit_regime_ctmc(regime_embs + next_embs, pseudo=pseudo)
                if Q_k is not None:
                    self.Q[k] = Q_k
        return assignments


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",        default="data/processed/events_v3.tsv")
    ap.add_argument("--vocab",         default="data/processed/vocab_v3.tsv")
    ap.add_argument("--ladder",        default="scripts/171_ladder.py")
    ap.add_argument("--n-regimes",     type=int,   default=2, dest="n_regimes")
    ap.add_argument("--em-iters",      type=int,   default=10, dest="em_iters")
    ap.add_argument("--horizon",       type=int,   default=1)
    ap.add_argument("--change-thresh", type=float, default=0.02,
                    dest="change_thresh")
    ap.add_argument("--test-end",      default="2025-02")
    ap.add_argument("--out",           default=None)
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
    print(f"  {P} variable positions | K={a.n_regimes} regimes")

    # historical change frequency null
    hist_change = np.zeros(P)
    for i in range(len(all_train) - 1):
        E1, E2 = emb.get(all_train[i]), emb.get(all_train[i+1])
        if E1 is not None and E2 is not None:
            hist_change += np.abs(E2 - E1)[var_ix].max(axis=1)
    hist_change /= max(len(all_train) - 1, 1)

    # build embedding sequence for training
    emb_seq_tr = [emb[m][var_ix] for m in all_train if m in emb]
    print(f"  {len(emb_seq_tr)} training months for HMM")

    # fit HMM
    print(f"fitting HMM ({a.n_regimes} regimes, {a.em_iters} EM iters) ...")
    hmm = SwitchingCTMC(a.n_regimes, P)
    assignments = hmm.fit(emb_seq_tr, var_ix, a.em_iters)
    counts = np.bincount(assignments, minlength=a.n_regimes)
    print(f"  regime assignments: {counts}")
    for k in range(a.n_regimes):
        ms = [all_train[i] for i, z in enumerate(assignments) if z == k]
        print(f"  regime {k}: {ms[:5]}{'...' if len(ms)>5 else ''}")

    def changed(E1, E2):
        return list(np.where(
            np.abs(E2-E1).max(axis=1) >= a.change_thresh)[0])

    # run Viterbi on full sequence to get regime at each test month
    emb_seq_all = [emb[m][var_ix]
                   for m in all_train + test_months if m in emb]
    all_months_used = [m for m in all_train + test_months if m in emb]
    viterbi_path = hmm.viterbi(emb_seq_all)
    regime_at = {m: viterbi_path[i]
                 for i, m in enumerate(all_months_used)}

    print(f"\n[eval] HMM-CTMC vs null")
    print(f"  for reference: CTMC @20=0.448  DFM+GRU @20=0.810")
    print(f"  {'month':9s} {'n_ch':>5s} {'regime':>7s} "
          + "".join(f"null@{K:2d} " for K in KS)
          + "".join(f" hmm@{K:2d} " for K in KS))

    rows = []
    for m in test_months:
        t_ix = months.index(m)
        if t_ix + a.horizon >= len(months): break
        E_now = emb.get(m)
        E_nxt = emb.get(months[t_ix + a.horizon])
        if E_now is None or E_nxt is None: continue
        truth = changed(E_now[var_ix], E_nxt[var_ix])
        if not truth: continue

        z = regime_at.get(m, 0)
        hmm_scores = ctmc_change_scores(hmm.Q[z], E_now[var_ix], a.horizon)
        r_null = recall_at_k(hist_change, truth, KS)
        r_hmm  = recall_at_k(hmm_scores,  truth, KS)

        row = {"month": m, "regime": int(z), "n_changed": len(truth),
               "null": r_null, "hmm": r_hmm}
        rows.append(row)
        print(f"  {m:9s} {len(truth):5d} {z:7d} "
              + "".join(f"{r_null[K]:7.3f} " for K in KS)
              + "".join(f"{r_hmm[K]:7.3f}  " for K in KS))

    if not rows:
        print("  NO EVALUABLE MONTHS"); return

    avg_null = {K: float(np.mean([r["null"][K] for r in rows])) for K in KS}
    avg_hmm  = {K: float(np.mean([r["hmm"][K]  for r in rows])) for K in KS}
    print(f"\n  {'MEAN':9s} {'':13s} "
          + "".join(f"{avg_null[K]:7.3f} " for K in KS)
          + "".join(f"{avg_hmm[K]:7.3f}  " for K in KS))
    print(f"\n  HMM-CTMC over null:")
    for K in KS:
        print(f"    @{K:2d}  {avg_hmm[K]-avg_null[K]:+.4f}")
    print(f"\n  COMPARISON TABLE")
    print(f"    null      0.276 @20")
    print(f"    CTMC      0.448 @20")
    print(f"    GRU       0.437 @20")
    print(f"    DFM       0.671 @20")
    print(f"    DFM+GRU   0.810 @20")
    print(f"    HMM-CTMC  {avg_hmm[20]:.3f} @20")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"n_regimes": a.n_regimes, "em_iters": a.em_iters,
                       "avg_null": avg_null, "avg_hmm": avg_hmm,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
