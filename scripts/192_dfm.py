#!/usr/bin/env python3
"""
192 -- DISCRETE FLOW MATCHING OVER SPIKE SEQUENCES

WHAT THIS TESTS
---------------
Does conditioning the substitution rate on the FULL CURRENT SEQUENCE
beat independent per-position rates (CTMC, 191)?

Comparison table:
    null   0.276 @20   historical change frequency, no direction
    CTMC   0.448 @20   direction, static rates, independent positions
    GRU    0.437 @20   dynamics, no direction, independent positions
    DFM    ?     @20   direction + within-sequence context, learned rates

THE MODEL
---------
Discrete Flow Matching (Campbell et al. 2024, Gat et al. 2024), fixed-
length substitution-only variant (section 2.2 of Edit Flows paper).

For a pair (x0, x1) of consecutive month dominant sequences, the
token-wise mixture path is:

    p_t(x^i | x0^i, x1^i) = (1-kappa_t) delta_{x0^i} + kappa_t delta_{x1^i}

with kappa_t = t (linear schedule). At time t, position i is at its
source residue with prob (1-t) and at its target with prob t.

The conditional rate is (eq 8, Edit Flows):

    u_t(x^i=a | x_t, x0^i, x1^i) = 1/(1-t) * (delta_{x1^i}(a) - delta_{x_t^i}(a))

i.e. jump from x_t^i to x1^i at rate 1/(1-t).

The model learns the MARGINAL rate (eq 7):

    u_theta(x^i=a | x_t) ~ neural network

Training: sample t ~ Uniform(0,1), sample x_t from the mixture path,
predict which positions should jump (cross-entropy against x1).

KEY DIFFERENCE FROM CTMC (191)
-------------------------------
The rate network sees the FULL x_t (all 1273 positions) not just
position i in isolation. So position 501's rate conditions on 498,
505, 417, etc. That's the within-sequence context.

FIXED LENGTH, SUBSTITUTION ONLY
---------------------------------
Deletions in spike are encoded as the '-' token in vocab_v3, so they
are substitutions. No insertion/deletion machinery needed. This is the
simplest DFM case -- no auxiliary Markov process (Theorem 3.1 of Edit
Flows is not needed).

EVALUATION: identical to 191
    truth  = positions where dominant residue changed >= thresh
    metric = recall@K: fraction of truth in top-K by predicted jump rate

TRAINING PAIRS
--------------
(x0, x1) = consecutive months' population-weighted dominant sequences.
x_t is sampled from the mixture path at random t.

The model sees x_t as a 1273-dimensional integer sequence and predicts
which positions will jump. Loss is cross-entropy at positions that
actually change (x0^i != x1^i), with the weight on changed positions
upweighted to prevent the model from learning to predict no change.

USAGE
    python scripts/192_dfm.py \
        --events data/processed/events_v3.tsv \
        --vocab  data/processed/vocab_v3.tsv \
        --ladder scripts/171_ladder.py \
        --epochs 100 --test-end 2025-02 \
        --out results/dfm.json

GIT
    git add scripts/192_dfm.py
    git commit -m "192: DFM over spike sequences -- within-sequence context vs independent CTMC"
    git push
"""

import argparse
import json
import importlib.util
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


# ----------------------------------------------------------------------------
# SEQUENCE CONSTRUCTION
# ----------------------------------------------------------------------------

def dominant_sequence(E, wuhan_res, all_pos, AA_IX, n_aa=21):
    """From a population embedding E (n_pos x 21), return the dominant
    residue index at each position. Positions not in E get the Wuhan residue.

    Returns: array of shape (1273,) with residue indices 0..20
    """
    seq = np.zeros(max(all_pos) + 2, dtype=np.int64)
    # fill with wuhan
    for pos, wt in wuhan_res.items():
        if pos < len(seq):
            seq[pos] = AA_IX.get(wt, 0)
    # fill with dominant observed residue
    for j, pos in enumerate(all_pos):
        if pos < len(seq):
            seq[pos] = int(np.argmax(E[j]))
    return seq


def build_pairs(emb, months, all_pos, wuhan_res, AA_IX):
    """Build (x0, x1) pairs from consecutive months."""
    pairs = []
    for a in range(len(months) - 1):
        E0 = emb.get(months[a])
        E1 = emb.get(months[a + 1])
        if E0 is None or E1 is None:
            continue
        x0 = dominant_sequence(E0, wuhan_res, all_pos, AA_IX)
        x1 = dominant_sequence(E1, wuhan_res, all_pos, AA_IX)
        if (x0 != x1).sum() == 0:
            continue
        pairs.append((x0, x1))
    return pairs


# ----------------------------------------------------------------------------
# MODEL: rate network
# ----------------------------------------------------------------------------

class RateNet(nn.Module):
    """Predicts substitution rates for each position given x_t.

    Input:  x_t ∈ {0..20}^L  (L = sequence length, padded/truncated to max_L)
    Output: logits ∈ R^{L x 21}  -- rate of each position jumping to each residue

    Architecture: embedding + 2-layer MLP with residual.
    No attention for laptop-friendliness. Could be replaced with a
    transformer for larger compute.
    """

    def __init__(self, max_L, n_aa=21, d=32, hidden=128):
        super().__init__()
        self.max_L = max_L
        self.emb = nn.Embedding(n_aa, d)
        # positional encoding: fixed sinusoidal
        pe = torch.zeros(max_L, d)
        pos = torch.arange(max_L).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d//2])
        self.register_buffer("pe", pe)
        # time embedding
        self.t_emb = nn.Linear(1, d)
        # per-position MLP: takes (token_emb + pos_enc + t_emb) -> 21
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_aa),
        )

    def forward(self, x_t, t_val):
        """x_t: (B, L) int64; t_val: scalar"""
        B, L = x_t.shape
        tok = self.emb(x_t)                          # (B, L, d)
        pe  = self.pe[:L].unsqueeze(0)               # (1, L, d)
        te  = self.t_emb(torch.tensor([[t_val]],
                         dtype=torch.float32,
                         device=x_t.device))         # (1, 1, d)
        h   = tok + pe + te                          # (B, L, d)
        return self.mlp(h)                           # (B, L, 21)


# ----------------------------------------------------------------------------
# DFM TRAINING
# ----------------------------------------------------------------------------

def sample_xt(x0, x1, t, n_aa=21):
    """Sample x_t from the token-wise mixture path (eq 8, Edit Flows).

    At each position independently:
        x_t^i = x1^i with prob t, x0^i with prob (1-t)
    """
    mask = (torch.rand(x0.shape, device=x0.device) < t)
    return torch.where(mask, x1, x0)


def dfm_loss(model, x0, x1, n_aa=21, weight_changed=10.0):
    """Flow matching cross-entropy loss (eq 7/8, fixed-length substitution).

    Sample t ~ Uniform(0,1), sample x_t, predict x1 at changed positions.
    Changed positions get upweighted to prevent predicting no-change.
    """
    t = random.uniform(0.0, 0.999)
    x_t = sample_xt(x0, x1, t)
    logits = model(x_t.unsqueeze(0), t).squeeze(0)  # (L, 21)

    changed = (x0 != x1)                            # (L,) bool
    weights = torch.where(changed,
                          torch.full_like(x0, weight_changed, dtype=torch.float),
                          torch.ones_like(x0, dtype=torch.float))
    loss = F.cross_entropy(logits, x1, reduction="none")
    return (loss * weights).sum() / weights.sum()


# ----------------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------------

def recall_at_k(scores, truth, Ks, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=np.float64)
    s = s + rng.uniform(-1e-12, 1e-12, size=s.shape)
    order = np.argsort(-s)
    rank = np.empty(len(s), dtype=np.int64)
    rank[order] = np.arange(len(s))
    hits = np.asarray(sorted(truth))
    return {K: float(np.mean(rank[hits] < K)) for K in Ks}


def predict_change_scores(model, x_t, var_ix, n_aa=21, n_t=10):
    """Score each variable position by predicted probability of changing.

    Average over multiple t values for a stable estimate.
    score[j] = mean_t (1 - p(x_t^i stays | x_t))
             = mean_t p(x_t^i changes to any other residue | x_t)
    """
    scores = np.zeros(len(var_ix))
    x_in = torch.tensor(x_t, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for t in np.linspace(0.1, 0.9, n_t):
            logits = model(x_in, float(t)).squeeze(0)  # (L, 21)
            probs = torch.softmax(logits, dim=-1).numpy()
            for k, ix in enumerate(var_ix):
                cur = x_t[ix]
                scores[k] += (1.0 - probs[ix, cur])
    return scores / n_t


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",        default="data/processed/events_v3.tsv")
    ap.add_argument("--vocab",         default="data/processed/vocab_v3.tsv")
    ap.add_argument("--ladder",        default="scripts/171_ladder.py")
    ap.add_argument("--epochs",        type=int,   default=100)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--d",             type=int,   default=32)
    ap.add_argument("--hidden",        type=int,   default=128)
    ap.add_argument("--change-thresh", type=float, default=0.02,
                    dest="change_thresh")
    ap.add_argument("--weight-changed",type=float, default=10.0,
                    dest="weight_changed")
    ap.add_argument("--test-end",      default="2025-02")
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--out",           default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    L189 = load_mod("scripts/189_gru_residue.py", "gru189")
    L    = load_mod("scripts/171_ladder.py",       "ladder171")

    # make AA_IX accessible
    AA_IX = L189.AA_IX
    AA    = L189.AA

    print("loading ...")
    monthly = L.load_events(a.events)
    months  = sorted(monthly)
    pos_res = L189.load_vocab(a.vocab)

    tr_end      = L.TRAIN_END[:7]
    all_train   = [m for m in months if m <= tr_end]
    test_months = [m for m in months if tr_end < m <= a.test_end]

    all_pos, wuhan_res, emb = L189.build_embeddings(
        monthly, pos_res, all_train + test_months)

    var_ix = L189.variable_positions(emb, all_train, a.change_thresh)
    P = len(var_ix)
    L_seq = max(all_pos) + 2
    print(f"  {P} variable positions | L_seq={L_seq}")

    # historical change frequency (the null)
    hist_change = np.zeros(P)
    for i in range(len(all_train) - 1):
        E1 = emb.get(all_train[i])
        E2 = emb.get(all_train[i+1])
        if E1 is not None and E2 is not None:
            hist_change += np.abs(E2-E1)[var_ix].max(axis=1)
    hist_change /= max(len(all_train)-1, 1)

    # build training pairs
    print("building training pairs ...")
    pairs = build_pairs(emb, all_train, all_pos, wuhan_res, AA_IX)
    print(f"  {len(pairs)} consecutive month pairs")
    if not pairs:
        print("  NO PAIRS"); return

    # convert to tensors
    def to_tensor(seq):
        return torch.tensor(seq[:L_seq], dtype=torch.long)

    train_pairs = [(to_tensor(x0), to_tensor(x1)) for x0, x1 in pairs]

    # train
    model = RateNet(L_seq, d=a.d, hidden=a.hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"\ntraining DFM  epochs={a.epochs} "
          f"params={sum(p.numel() for p in model.parameters()):,} ...")

    for ep in range(a.epochs):
        model.train()
        random.shuffle(train_pairs)
        tot = 0.0
        for x0, x1 in train_pairs:
            loss = dfm_loss(model, x0, x1,
                            weight_changed=a.weight_changed)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep+1) % 20 == 0:
            print(f"  ep {ep+1:3d}  loss {tot/len(train_pairs):.4f}")

    # evaluate
    model.eval()
    KS = [5, 10, 20]

    def changed(E1, E2):
        return list(np.where(
            np.abs(E2-E1)[var_ix].max(axis=1) >= a.change_thresh)[0])

    print(f"\n[eval] DFM vs null vs CTMC  h=1")
    print(f"  for CTMC baseline see results/ctmc_h1.json")
    print(f"  {'month':9s} {'n_ch':>5s} "
          + "".join(f"null@{K:2d} " for K in KS)
          + "".join(f" dfm@{K:2d} " for K in KS))

    rows = []
    for m in test_months:
        t_ix = months.index(m)
        if t_ix + 1 >= len(months): break
        E_now = emb.get(m)
        E_nxt = emb.get(months[t_ix+1])
        if E_now is None or E_nxt is None: continue
        truth = changed(E_now, E_nxt)
        if not truth: continue

        x_now = dominant_sequence(E_now, wuhan_res, all_pos, AA_IX)
        dfm_scores = predict_change_scores(model, x_now, var_ix)
        r_null = recall_at_k(hist_change, truth, KS)
        r_dfm  = recall_at_k(dfm_scores,  truth, KS)

        row = {"month": m, "n_changed": len(truth),
               "null": r_null, "dfm": r_dfm}
        rows.append(row)
        print(f"  {m:9s} {len(truth):5d} "
              + "".join(f"{r_null[K]:7.3f} " for K in KS)
              + "".join(f"{r_dfm[K]:7.3f}  " for K in KS))

    if not rows:
        print("  NO EVALUABLE MONTHS"); return

    avg_null = {K: float(np.mean([r["null"][K] for r in rows])) for K in KS}
    avg_dfm  = {K: float(np.mean([r["dfm"][K]  for r in rows])) for K in KS}
    print(f"\n  {'MEAN':9s} {'':5s} "
          + "".join(f"{avg_null[K]:7.3f} " for K in KS)
          + "".join(f"{avg_dfm[K]:7.3f}  " for K in KS))
    print(f"\n  DFM over null:")
    for K in KS:
        print(f"    @{K:2d}  {avg_dfm[K]-avg_null[K]:+.4f}")
    print(f"\n  CTMC @20 = 0.448  (from 191)")
    print(f"  DFM  @20 = {avg_dfm[20]:.3f}")
    print(f"  DFM > CTMC -> within-sequence context adds over "
          f"independent rates.")
    print(f"  DFM ~ CTMC -> context does not help; "
          f"independent rates are sufficient.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"epochs": a.epochs, "d": a.d,
                       "change_thresh": a.change_thresh,
                       "P": P, "L_seq": L_seq,
                       "avg_null": avg_null, "avg_dfm": avg_dfm,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
