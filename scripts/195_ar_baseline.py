#!/usr/bin/env python3
"""
195 -- AUTOREGRESSIVE BASELINE FOR RESIDUE SUBSTITUTION

THE MODEL
---------
Predict each position's next residue conditioned on all previous positions
in frequency order (most-to-least variable, so the most informative
positions are conditioned on first):

    p(r'_i | r'_1, ..., r'_{i-1}, h_T)

where r'_j is the predicted (sampled) next residue at position j and
h_T is the GRU temporal context.

This is the standard autoregressive (AR) language model applied to
residue sequences. It captures within-sequence dependencies through
the chain rule, unlike the independent models (CTMC, GRU, HMM-CTMC).

WHY THIS COMPARISON MATTERS
---------------------------
DFM+GRU captures joint rates through shared conditioning on x_t.
AR captures joint rates through the chain rule.
Both are valid approaches to modelling position dependencies.

If DFM+GRU > AR:  non-autoregressive CTMC-based modeling beats
                   sequential conditioning. The CTMC path integral
                   captures dependencies that the AR chain misses.
If DFM+GRU ~ AR:  both capture the same dependencies. The choice
                   between them is computational, not scientific.

ARCHITECTURE
------------
Same RateNet as 193, but at evaluation time:
    1. For each position i in order:
       - feed x_t[:i] (already predicted positions) and h_T to network
       - sample or take argmax of predicted distribution for position i
    2. Score each position by its predicted change probability GIVEN
       the preceding positions' predictions.

Training is identical to 193 (DFM cross-entropy on pairs) -- only the
evaluation differs. This isolates whether the AR evaluation protocol
changes results vs the DFM sampling protocol.

COMPARISON TABLE
----------------
    null        0.276 @20
    CTMC        0.448 @20    independent, stationary
    GRU         0.437 @20    independent, temporal
    HMM-CTMC    0.351 @20    independent, regime-switching
    DFM         0.671 @20    joint, no temporal
    DFM+GRU     0.810 @20    joint, temporal
    AR+GRU      ?     @20    autoregressive joint, temporal

USAGE
    python scripts/195_ar_baseline.py \\
        --events data/processed/events_v3.tsv \\
        --vocab  data/processed/vocab_v3.tsv \\
        --ladder scripts/171_ladder.py \\
        --train-window 12 --epochs 100 \\
        --test-end 2025-02 --out results/ar_baseline.json

GIT
    git add scripts/195_ar_baseline.py
    git commit -m "195: autoregressive baseline vs DFM+GRU"
    git push
"""

import argparse
import importlib.util
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AA = list("ACDEFGHIKLMNPQRSTVWY-")
N_AA = len(AA)


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
# AUTOREGRESSIVE MODEL
# Identical architecture to 193's RateNet+GRU, but evaluated autoregressively
# ----------------------------------------------------------------------------

class TemporalGRU(nn.Module):
    def __init__(self, P, d_gru=32):
        super().__init__()
        self.gru = nn.GRU(P * N_AA, d_gru, batch_first=True)
        self.d_gru = d_gru

    def forward(self, E_seq):
        x = torch.tensor(E_seq, dtype=torch.float32).view(1, len(E_seq), -1)
        _, h = self.gru(x)
        return h.squeeze()


class ARNet(nn.Module):
    """Autoregressive residue predictor.

    For position i, conditions on:
        - embeddings of all positions 0..i-1 (causal mask)
        - time t (flow interpolation step)
        - h_T (GRU temporal context)

    Uses a causal transformer-style aggregation: position i attends
    to positions 0..i-1 through a lower-triangular mask.
    """

    def __init__(self, P, d=32, hidden=128, n_fourier=4, d_gru=32):
        super().__init__()
        self.P = P
        self.d_gru = d_gru
        self.emb = nn.Embedding(N_AA, d)
        # positional encoding
        pe = torch.zeros(P, d)
        pos = torch.arange(P).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float()
                        * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d // 2])
        self.register_buffer("pe", pe)
        self.n_fourier = n_fourier
        t_dim = 2 * n_fourier
        # causal context: weighted sum of previous position embeddings
        self.causal_agg = nn.Linear(d, d)
        # output MLP per position
        self.mlp = nn.Sequential(
            nn.Linear(d + d + t_dim + d_gru, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, N_AA),
        )

    def fourier_t(self, t, device):
        freqs = torch.arange(1, self.n_fourier + 1,
                             dtype=torch.float32, device=device)
        return torch.cat([torch.sin(freqs * t * np.pi),
                          torch.cos(freqs * t * np.pi)])

    def forward(self, x_t, t_val, h_T=None):
        """Non-causal forward for TRAINING (same as DFM, all positions).

        x_t: (P,) int64; t_val: scalar; h_T: (d_gru,) or None
        Returns logits: (P, 21)
        """
        emb = self.emb(x_t) + self.pe            # (P, d)
        psi = self.fourier_t(t_val, x_t.device)  # (t_dim,)
        if h_T is None:
            h_T = torch.zeros(self.d_gru, device=x_t.device)

        # causal context: for position i, aggregate emb[0..i-1]
        logits = []
        ctx = torch.zeros(self.pe.shape[1], device=x_t.device)  # running sum
        for i in range(self.P):
            causal = torch.relu(self.causal_agg(ctx / max(i, 1)))
            inp = torch.cat([emb[i], causal, psi, h_T])
            logits.append(self.mlp(inp))
            ctx = ctx + emb[i]                   # accumulate for next pos
        return torch.stack(logits)               # (P, 21)

    def ar_scores(self, x_now, h_T, n_samples=10):
        """Autoregressive scoring: sample predicted next sequence,
        score each position by p(change) given preceding positions.

        Runs the chain n_samples times and averages for stability.
        """
        P = self.P
        scores = np.zeros(P)
        if h_T is None:
            h_T_use = torch.zeros(self.d_gru)
        else:
            h_T_use = h_T

        with torch.no_grad():
            for _ in range(n_samples):
                x_pred = x_now.copy()
                for i in range(P):
                    x_in = torch.tensor(x_pred, dtype=torch.long)
                    logits = self.forward(x_in, 0.5, h_T_use)
                    probs = torch.softmax(logits[i], dim=0).numpy()
                    # score = prob of NOT staying at current residue
                    scores[i] += 1.0 - probs[x_now[i]]
                    # sample next residue for autoregressive chain
                    x_pred[i] = int(np.random.choice(N_AA, p=probs))
        return scores / n_samples


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",         default="data/processed/events_v3.tsv")
    ap.add_argument("--vocab",          default="data/processed/vocab_v3.tsv")
    ap.add_argument("--ladder",         default="scripts/171_ladder.py")
    ap.add_argument("--train-window",   type=int,   default=12)
    ap.add_argument("--epochs",         type=int,   default=100)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--d",              type=int,   default=32)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--pairs-per-step", type=int,   default=50)
    ap.add_argument("--max-per-month",  type=int,   default=200)
    ap.add_argument("--n-ar-samples",   type=int,   default=10,
                    dest="n_ar_samples")
    ap.add_argument("--change-thresh",  type=float, default=0.02,
                    dest="change_thresh")
    ap.add_argument("--weight-changed", type=float, default=10.0,
                    dest="weight_changed")
    ap.add_argument("--test-end",       default="2025-02")
    ap.add_argument("--seed",           type=int,   default=0)
    ap.add_argument("--out",            default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    # load helpers from 193 and 189
    L193 = load_mod("scripts/193_dfm_seqs.py", "dfm193")
    L189 = load_mod("scripts/189_gru_residue.py", "gru189")
    L    = load_mod("scripts/171_ladder.py", "ladder171")

    print("loading ...")
    monthly_agg = L.load_events(a.events)
    months = sorted(monthly_agg)
    pos_res = L193.load_vocab(a.vocab)

    tr_end      = L.TRAIN_END[:7]
    all_train   = [m for m in months if m <= tr_end]
    train_months = all_train[-a.train_window:] if a.train_window > 0 \
        else all_train
    test_months = [m for m in months if tr_end < m <= a.test_end]

    all_pos, wuhan, emb = L189.build_embeddings(
        monthly_agg, pos_res, all_train + test_months)
    var_ix = L189.variable_positions(emb, all_train, a.change_thresh)
    P = len(var_ix)
    print(f"  {P} variable positions")
    if P < 5:
        print("  TOO FEW -- lower --change-thresh"); return

    # historical change frequency null
    hist_change = np.zeros(P)
    for i in range(len(all_train) - 1):
        E1, E2 = emb.get(all_train[i]), emb.get(all_train[i+1])
        if E1 is not None and E2 is not None:
            hist_change += np.abs(E2-E1)[var_ix].max(axis=1)
    hist_change /= max(len(all_train)-1, 1)

    # build training data (same as 193)
    print("building sequence pools ...")
    monthly_raw = L193.build_monthly(
        a.events, pos_res, train_months + test_months)
    seq_pool = L193.build_seq_pool(
        monthly_raw, train_months, pos_res, wuhan, all_pos, var_ix,
        a.max_per_month)

    # build labeled pairs with month context
    rng_pair = random.Random(a.seed)
    labeled_pairs = []
    for a_idx in range(len(train_months) - 1):
        m0, m1 = train_months[a_idx], train_months[a_idx+1]
        s0, s1 = seq_pool.get(m0, []), seq_pool.get(m1, [])
        if not s0 or not s1: continue
        w0, w1 = [c for _, c in s0], [c for _, c in s1]
        for _ in range(a.pairs_per_step):
            x0 = rng_pair.choices(s0, weights=w0, k=1)[0][0]
            x1 = rng_pair.choices(s1, weights=w1, k=1)[0][0]
            if not np.array_equal(x0, x1):
                labeled_pairs.append((x0.copy(), x1.copy(), m0))
    print(f"  {len(labeled_pairs):,} training pairs")
    if not labeled_pairs:
        print("  NO PAIRS"); return

    # train
    gru   = TemporalGRU(P, d_gru=a.d)
    model = ARNet(P, a.d, a.hidden, d_gru=a.d)
    params = list(model.parameters()) + list(gru.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    print(f"\ntraining AR+GRU  epochs={a.epochs} "
          f"params={sum(p.numel() for p in params):,} ...")

    def get_E_ctx(up_to_month):
        ctx = [emb[m][var_ix] for m in train_months
               if m <= up_to_month and m in emb]
        return np.stack(ctx) if ctx else None

    def sample_xt(x0, x1, t):
        mask = (torch.rand(x0.shape) < t)
        return torch.where(mask, x1, x0)

    for ep in range(a.epochs):
        model.train(); gru.train()
        random.shuffle(labeled_pairs)
        tot = 0.0
        for x0, x1, m0 in labeled_pairs:
            x0t = torch.tensor(x0, dtype=torch.long)
            x1t = torch.tensor(x1, dtype=torch.long)
            E_ctx = get_E_ctx(m0)
            h_T = gru(E_ctx) if E_ctx is not None else None
            t = random.uniform(0.01, 0.99)
            xt = sample_xt(x0t, x1t, t)
            logits = model(xt, t, h_T)
            changed = (x0t != x1t).float()
            w = changed * a.weight_changed + 1.0
            loss = (F.cross_entropy(logits, x1t, reduction="none") * w
                    ).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep+1) % 20 == 0:
            print(f"  ep {ep+1:3d}  loss {tot/len(labeled_pairs):.4f}")

    # evaluate
    model.eval(); gru.eval()
    KS = [5, 10, 20]

    def changed(E1, E2):
        return list(np.where(
            np.abs(E2-E1)[var_ix].max(axis=1) >= a.change_thresh)[0])

    print(f"\n[eval] AR+GRU vs null")
    print(f"  for reference: CTMC=0.448  DFM+GRU=0.810")
    print(f"  {'month':9s} {'n_ch':>5s} "
          + "".join(f"null@{K:2d} " for K in KS)
          + "".join(f"  ar@{K:2d} " for K in KS))

    rows = []
    for t_ix_t, m in enumerate(test_months):
        t_ix = months.index(m)
        if t_ix + 1 >= len(months): break
        E_now = emb.get(m)
        E_nxt = emb.get(months[t_ix+1])
        if E_now is None or E_nxt is None: continue
        truth = changed(E_now, E_nxt)
        if not truth: continue

        # GRU context up to test month
        E_ctx_test = np.stack([emb[x][var_ix] for x in
                               train_months + test_months[:t_ix_t+1]
                               if x <= m and x in emb]) \
            if any(x in emb for x in train_months) else None
        with torch.no_grad():
            h_T = gru(E_ctx_test) if E_ctx_test is not None else None

        # sample x_0 from current month sequences
        test_raw = monthly_raw.get(m, {})
        if not test_raw: continue
        items = list(test_raw.items())
        weights = [c for _, c in items]
        sampled = random.choices(items, weights=weights,
                                 k=min(10, len(items)))
        x0_arr = np.array([
            L193.seq_from_muts(muts, wuhan, all_pos, pos_res, var_ix)
            for muts, _ in sampled
        ])

        # AR scoring: average over sampled x_0 sequences
        ar_scores = np.zeros(P)
        for x0 in x0_arr:
            ar_scores += model.ar_scores(x0, h_T, a.n_ar_samples)
        ar_scores /= len(x0_arr)

        r_null = recall_at_k(hist_change, truth, KS)
        r_ar   = recall_at_k(ar_scores,   truth, KS)

        row = {"month": m, "n_changed": len(truth),
               "null": r_null, "ar": r_ar}
        rows.append(row)
        print(f"  {m:9s} {len(truth):5d} "
              + "".join(f"{r_null[K]:7.3f} " for K in KS)
              + "".join(f"{r_ar[K]:7.3f}  " for K in KS))

    if not rows:
        print("  NO EVALUABLE MONTHS"); return

    avg_null = {K: float(np.mean([r["null"][K] for r in rows])) for K in KS}
    avg_ar   = {K: float(np.mean([r["ar"][K]   for r in rows])) for K in KS}
    print(f"\n  {'MEAN':9s} {'':5s} "
          + "".join(f"{avg_null[K]:7.3f} " for K in KS)
          + "".join(f"{avg_ar[K]:7.3f}  " for K in KS))
    print(f"\n  AR+GRU over null:")
    for K in KS:
        print(f"    @{K:2d}  {avg_ar[K]-avg_null[K]:+.4f}")
    print(f"\n  FINAL COMPARISON TABLE")
    print(f"    null        0.276 @20")
    print(f"    HMM-CTMC    0.351 @20")
    print(f"    CTMC        0.448 @20")
    print(f"    GRU         0.437 @20")
    print(f"    DFM         0.671 @20")
    print(f"    AR+GRU      {avg_ar[20]:.3f} @20")
    print(f"    DFM+GRU     0.810 @20")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"train_window": a.train_window, "epochs": a.epochs,
                       "P": P, "avg_null": avg_null, "avg_ar": avg_ar,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
