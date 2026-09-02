#!/usr/bin/env python3
"""
193 -- DFM ON INDIVIDUAL SEQUENCES

WHAT CHANGED FROM 192
---------------------
192 trained on 16 dominant-sequence pairs -- too few for a neural network.
193 trains on INDIVIDUAL OBSERVED SEQUENCES:
    for each month t, each observed constellation is a training x_0
    paired with sequences from month t+1 as x_1
    (random pairing, since ancestry is latent)
This gives thousands of training pairs instead of 16.

EVALUATION
----------
Same as 191 (CTMC): recall@K on changed positions.
Changed position = variable position where the population-weighted
dominant residue shifted by >= change_thresh between month T and T+1.

Score per position = mean over sampled x_0 of predicted change probability.

COMPARISON
----------
    null     0.276 @20    historical frequency
    CTMC     0.448 @20    per-position rates, independent
    GRU      0.437 @20    temporal dynamics, independent
    DFM      ?     @20    joint rates via full-sequence conditioning

DFM > CTMC -> within-sequence context helps
DFM ~ CTMC -> independence sufficient (consistent with 174)

TRAINING PAIRS
--------------
For each consecutive month pair (t, t+1) in the training window:
    sample min(n_pairs_per_month, n_seqs_t * n_seqs_t1) pairs
    x_0 ~ uniform over sequences in month t (weighted by count)
    x_1 ~ uniform over sequences in month t+1 (weighted by count)
    ancestry is latent; pairing is approximate but data-sufficient

RUNTIME ESTIMATE (MacBook CPU)
    P=114 variable positions, d=32, hidden=128
    ~1000 training pairs x 100 epochs = ~100k forward passes
    ~10-20 minutes

USAGE
    python scripts/193_dfm_seqs.py \
        --events data/processed/events_v3.tsv \
        --vocab  data/processed/vocab_v3.tsv \
        --ladder scripts/171_ladder.py \
        --train-window 12 --epochs 100 \
        --test-end 2025-02 --out results/dfm_seqs.json

GIT
    git add scripts/193_dfm_seqs.py
    git commit -m "193: DFM on individual sequences -- proper training and recall@K eval"
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
AA_IX = {a: i for i, a in enumerate(AA)}
N_AA = len(AA)


# ----------------------------------------------------------------------------
# BUILD PER-SEQUENCE REPRESENTATIONS OVER VARIABLE POSITIONS
# ----------------------------------------------------------------------------

def load_vocab(path):
    import re
    pos_res = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            mid, name = parts[0].strip(), parts[1].strip()
            m = re.match(r"S:([A-Z-])(\d+)([A-Z-])", name)
            if m:
                pos_res[mid] = (int(m.group(2)), m.group(1), m.group(3))
    return pos_res


def build_monthly(events_path, pos_res, months):
    """For each month, build {frozenset_of_mut_ids: count}."""
    from collections import defaultdict, Counter
    monthly_raw = defaultdict(Counter)
    with open(events_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            date, var, cnt = parts[0].strip(), parts[1].strip(), parts[2].strip()
            m = date[:7]
            if m not in set(months):
                continue
            try:
                c = int(float(cnt))
            except ValueError:
                continue
            muts = frozenset(var.split(",")) if "," in var else frozenset([var])
            monthly_raw[m][muts] += c
    return monthly_raw


def seq_from_muts(muts, wuhan, all_pos, pos_res, var_ix):
    """Convert a frozenset of mutation IDs to a variable-position sequence.

    Returns array of shape (P,) with residue indices 0..20.
    Unmutated positions get the Wuhan residue index.
    """
    # start from wuhan residue index at each position
    seq = {}
    for pos, wt in wuhan.items():
        seq[pos] = AA_IX.get(wt, 0)
    # apply mutations
    for mid in muts:
        if mid in pos_res:
            pos, wt, mt = pos_res[mid]
            seq[pos] = AA_IX.get(mt, 0)
    # extract variable positions; default to 0 (Alanine) if position unknown
    return np.array([seq.get(all_pos[i], 0) for i in var_ix], dtype=np.int64)


def build_seq_pool(monthly_raw, months, pos_res, wuhan, all_pos, var_ix,
                   max_per_month=500):
    """For each month, build a list of (sequence, count) pairs."""
    pool = {}
    for m in months:
        seqs = []
        for muts, cnt in monthly_raw.get(m, {}).items():
            if not any(mid in pos_res for mid in muts) and len(muts) > 1:
                continue
            s = seq_from_muts(muts, {all_pos[i]: AA[np.argmax(
                np.zeros(N_AA))] for i in var_ix}, wuhan, all_pos, var_ix)
            seqs.append((s, cnt))
        if len(seqs) > max_per_month:
            seqs = random.choices(seqs, weights=[c for _, c in seqs],
                                  k=max_per_month)
        pool[m] = seqs
    return pool


def sample_pairs(pool, months, n_pairs_per_step=50, seed=0):
    """Sample (x_0, x_1) pairs from consecutive months."""
    rng = random.Random(seed)
    pairs = []
    for a in range(len(months) - 1):
        m0, m1 = months[a], months[a + 1]
        s0 = pool.get(m0, [])
        s1 = pool.get(m1, [])
        if not s0 or not s1:
            continue
        w0 = [c for _, c in s0]
        w1 = [c for _, c in s1]
        for _ in range(n_pairs_per_step):
            x0 = rng.choices(s0, weights=w0, k=1)[0][0]
            x1 = rng.choices(s1, weights=w1, k=1)[0][0]
            if not np.array_equal(x0, x1):
                pairs.append((x0.copy(), x1.copy()))
    return pairs


# ----------------------------------------------------------------------------
# MODEL: rate network over variable positions
# ----------------------------------------------------------------------------

class RateNet(nn.Module):
    """Rate network: given x_t (P variable positions) and t,
    predict substitution rates at each position.

    Input:  x_t ∈ {0..20}^P, t ∈ [0,1]
    Output: logits ∈ R^{P x 21}
    """

    def __init__(self, P, d=32, hidden=128, n_fourier=4):
        super().__init__()
        self.P = P
        self.emb = nn.Embedding(N_AA, d)
        # positional encoding
        pe = torch.zeros(P, d)
        pos = torch.arange(P).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float()
                        * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d // 2])
        self.register_buffer("pe", pe)
        # time conditioning via Fourier features
        self.n_fourier = n_fourier
        t_dim = 2 * n_fourier
        # MLP: per-position, but sees all positions through embedding sum
        self.context = nn.Linear(P * d, d)   # global context from all positions
        self.mlp = nn.Sequential(
            nn.Linear(d + d + t_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),        nn.ReLU(),
            nn.Linear(hidden, N_AA),
        )

    def fourier_t(self, t, device):
        freqs = torch.arange(1, self.n_fourier + 1,
                             dtype=torch.float32, device=device)
        return torch.cat([torch.sin(freqs * t * np.pi),
                          torch.cos(freqs * t * np.pi)])

    def forward(self, x_t, t_val):
        """x_t: (P,) int64; t_val: scalar float"""
        emb = self.emb(x_t)                          # (P, d)
        emb = emb + self.pe                           # (P, d)
        ctx = torch.relu(self.context(emb.flatten())) # (d,) global context
        psi = self.fourier_t(t_val, x_t.device)      # (t_dim,)
        logits = []
        for i in range(self.P):
            inp = torch.cat([emb[i], ctx, psi])       # (d + d + t_dim,)
            logits.append(self.mlp(inp))
        return torch.stack(logits)                    # (P, 21)


# ----------------------------------------------------------------------------
# DFM TRAINING AND EVALUATION
# ----------------------------------------------------------------------------

def sample_xt(x0, x1, t):
    """Token-wise mixture path: each position is at x1 with prob t."""
    mask = (torch.rand(x0.shape) < t)
    return torch.where(mask, x1, x0)


def dfm_step(model, x0, x1, weight_changed=10.0):
    t = random.uniform(0.01, 0.99)
    xt = sample_xt(x0, x1, t)
    logits = model(xt, t)                            # (P, 21)
    changed = (x0 != x1).float()
    w = changed * weight_changed + 1.0
    loss = F.cross_entropy(logits, x1, reduction="none")
    return (loss * w).sum() / w.sum()


def recall_at_k(scores, truth, Ks, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=np.float64)
    s = s + rng.uniform(-1e-12, 1e-12, size=s.shape)
    order = np.argsort(-s)
    rank = np.empty(len(s), dtype=np.int64)
    rank[order] = np.arange(len(s))
    hits = np.asarray(sorted(truth))
    return {K: float(np.mean(rank[hits] < K)) for K in Ks}


def predict_scores(model, x0_arr, n_t=5):
    """Score each variable position by mean predicted change probability.

    Average over multiple sampled x_0 sequences from current month
    and multiple interpolation times t.
    """
    P = x0_arr.shape[1] if x0_arr.ndim == 2 else len(x0_arr)
    scores = np.zeros(P if x0_arr.ndim == 1 else x0_arr.shape[1])
    n = 0
    seqs = x0_arr if x0_arr.ndim == 2 else x0_arr[None]
    with torch.no_grad():
        for x0 in seqs:
            x0t = torch.tensor(x0, dtype=torch.long)
            for t in np.linspace(0.1, 0.9, n_t):
                logits = model(x0t, float(t))        # (P, 21)
                probs = torch.softmax(logits, dim=1).numpy()
                cur = x0
                scores += 1.0 - probs[np.arange(P), cur]
                n += 1
    return scores / max(n, 1)


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

    L189 = load_mod("scripts/189_gru_residue.py", "gru189")
    L    = load_mod("scripts/171_ladder.py",       "ladder171")

    print("loading events ...")
    monthly_agg = L.load_events(a.events)
    months = sorted(monthly_agg)

    print("loading vocab ...")
    pos_res = load_vocab(a.vocab)

    tr_end      = L.TRAIN_END[:7]
    all_train   = [m for m in months if m <= tr_end]
    train_months = all_train[-a.train_window:] if a.train_window > 0 \
        else all_train
    test_months = [m for m in months if tr_end < m <= a.test_end]

    # build population embeddings for evaluation
    print("building residue embeddings ...")
    all_pos, wuhan, emb = L189.build_embeddings(
        monthly_agg, pos_res, all_train + test_months)
    var_ix = L189.variable_positions(emb, all_train, a.change_thresh)
    P = len(var_ix)
    print(f"  {P} variable positions")
    if P < 5:
        print("  TOO FEW -- lower --change-thresh"); return

    # wuhan residues at variable positions
    wuhan_at_var = {all_pos[i]: wuhan.get(all_pos[i], "A") for i in var_ix}

    # historical change frequency null
    hist_change = np.zeros(P)
    for i in range(len(all_train) - 1):
        E1, E2 = emb.get(all_train[i]), emb.get(all_train[i + 1])
        if E1 is not None and E2 is not None:
            hist_change += np.abs(E2 - E1)[var_ix].max(axis=1)
    hist_change /= max(len(all_train) - 1, 1)

    # build individual sequence pool
    print("building individual sequence pools ...")
    monthly_raw = build_monthly(a.events, pos_res, train_months + test_months)
    seq_pool = build_seq_pool(monthly_raw, train_months, pos_res,
                              wuhan, all_pos, var_ix,
                              a.max_per_month)
    pairs = sample_pairs(seq_pool, train_months, a.pairs_per_step, a.seed)
    print(f"  {len(pairs):,} training pairs from individual sequences")
    if not pairs:
        print("  NO PAIRS -- check vocab and events format"); return

    # train
    model = RateNet(P, a.d, a.hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"\ntraining DFM  epochs={a.epochs} "
          f"params={sum(p.numel() for p in model.parameters()):,} ...")

    for ep in range(a.epochs):
        model.train()
        random.shuffle(pairs)
        tot = 0.0
        for x0, x1 in pairs:
            x0t = torch.tensor(x0, dtype=torch.long)
            x1t = torch.tensor(x1, dtype=torch.long)
            loss = dfm_step(model, x0t, x1t, a.weight_changed)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d}  loss {tot/len(pairs):.4f}")

    # evaluate: same protocol as 191 (CTMC)
    model.eval()
    KS = [5, 10, 20]

    def changed(E1, E2):
        return list(np.where(
            np.abs(E2 - E1)[var_ix].max(axis=1) >= a.change_thresh)[0])

    print(f"\n[eval] DFM vs null  (CTMC @20 = 0.448 for reference)")
    print(f"  {'month':9s} {'n_ch':>5s} "
          + "".join(f"null@{K:2d} " for K in KS)
          + "".join(f" dfm@{K:2d} " for K in KS))

    rows = []
    for m in test_months:
        t_ix = months.index(m)
        if t_ix + 1 >= len(months): break
        E_now = emb.get(m)
        E_nxt = emb.get(months[t_ix + 1])
        if E_now is None or E_nxt is None: continue
        truth = changed(E_now, E_nxt)
        if not truth: continue

        # sample sequences from current month for scoring
        test_seqs_raw = monthly_raw.get(m, {})
        if not test_seqs_raw:
            continue
        items = list(test_seqs_raw.items())
        weights = [c for _, c in items]
        sampled = random.choices(items, weights=weights,
                                 k=min(20, len(items)))
        x0_arr = np.array([
            seq_from_muts(muts, wuhan, all_pos, pos_res, var_ix)
            for muts, _ in sampled
        ])

        dfm_scores = predict_scores(model, x0_arr)
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
    print(f"\n  CTMC @20 = 0.448  GRU @20 = 0.437  null @20 = 0.276")
    print(f"  DFM  @20 = {avg_dfm[20]:.3f}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"train_window": a.train_window, "epochs": a.epochs,
                       "P": P, "n_pairs": len(pairs),
                       "avg_null": avg_null, "avg_dfm": avg_dfm,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
