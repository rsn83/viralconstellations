#!/usr/bin/env python3
"""
189 -- GRU OVER MONTHLY RESIDUE VECTORS

WHAT IS MODELLED
----------------
At each month t the population has a distribution over residues at each
spike position. Represent it as a mean embedding:

    E_t[i, r] = fraction of sequences in month t carrying residue r
                at position i, weighted by sequence count

That's a matrix in R^{P x 21} where P = number of variable positions and
21 = amino acid alphabet + deletion.

The GRU takes E_t as input and predicts E_{t+h}.

WHY THIS AVOIDS THE HIGHER-ORDER PROBLEM
-----------------------------------------
With sets, a new constellation is a new atomic object in 2^V space.
With sequences, position i is always position i -- fixed dimension P.
Predicting the next residue distribution is P independent categoricals.
No combinatorial explosion.

EVALUATION: RECALL@K ON CHANGED POSITIONS
------------------------------------------
A model predicting "no change" gets ~99.6% accuracy (5 positions change
out of 1273). So accuracy is meaningless. Instead:

    For each test month pair (t, t+h):
    - truth  = positions where the population-modal residue changed
    - model  = ranks positions by predicted probability of change
    - metric = recall@K: fraction of changed positions in top K

Null: rank positions by historical change frequency (how often they
changed in training). If model recall > null recall, the model learned
something beyond counting.

WHAT THE MODEL LEARNS
---------------------
To predict which positions' residue distributions will shift in the next
h months. A position moving from 0% Y to 30% Y in the prediction means
the model expects that mutation to become common -- a new constellation
signal without enumerating sets.

SCOPE
-----
Train on last N months (default 6), predict month N+1.
Variable positions only: those that changed at least once in training.
Laptop-runnable: P ~ 100-300, d = 32, 77 monthly snapshots.

USAGE
    python scripts/189_gru_residue.py \
        --events data/processed/events_v3.tsv \
        --vocab  data/processed/vocab_v3.tsv \
        --train-window 6 --horizon 1 \
        --out results/gru_residue.json

GIT
    git add scripts/189_gru_residue.py
    git commit -m "189: GRU over monthly residue vectors, recall@K on changed positions"
    git push
"""

import argparse
import importlib.util
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# BUILD RESIDUE VOCAB AND POPULATION EMBEDDINGS
# ----------------------------------------------------------------------------

def load_vocab(path):
    """mut_id -> (position, residue_char).
    Lines: id <TAB> S:A501Y <TAB> date
    """
    pos_res = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            mid, name = parts[0].strip(), parts[1].strip()
            # parse S:X{pos}{res} e.g. S:D614G -> pos=614, wt=D, mut=G
            import re
            m = re.match(r"S:([A-Z-])(\d+)([A-Z-])", name)
            if m:
                pos_res[mid] = (int(m.group(2)), m.group(1), m.group(3))
    return pos_res


AA = list("ACDEFGHIKLMNPQRSTVWY-")
AA_IX = {a: i for i, a in enumerate(AA)}
N_AA = len(AA)    # 21


def build_embeddings(monthly, pos_res, months):
    """For each month, build population-weighted residue distribution.

    Returns:
        positions: sorted list of variable positions
        wuhan_res: {pos: wt_residue} from vocab
        emb: dict {month: np.array shape (P, 21)}
    """
    # find all positions ever mutated
    all_pos = sorted({pos for mid in pos_res for pos, wt, mt in [pos_res[mid]]})

    # wuhan residue per position (the wild-type)
    wuhan = {}
    for mid, (pos, wt, mt) in pos_res.items():
        wuhan[pos] = wt

    # per-month residue counts per position
    # default: wuhan residue weighted by total sequences
    emb = {}
    for m in months:
        counts = monthly.get(m, {})
        total = sum(counts.values())
        if total == 0:
            continue
        # start from wuhan: all mass on wt residue
        pos_counts = {pos: defaultdict(float) for pos in all_pos}
        for pos in all_pos:
            wt = wuhan.get(pos, "A")
            pos_counts[pos][wt] += total

        # subtract wuhan contribution for mutated positions and add mutant
        for mut_set, cnt in counts.items():
            for mid in mut_set:
                if mid not in pos_res:
                    continue
                pos, wt, mt = pos_res[mid]
                pos_counts[pos][wt] -= cnt
                pos_counts[pos][mt] += cnt

        arr = np.zeros((len(all_pos), N_AA), dtype=np.float32)
        for j, pos in enumerate(all_pos):
            s = sum(pos_counts[pos].values())
            if s <= 0:
                wt = wuhan.get(pos, "A")
                arr[j, AA_IX.get(wt, 0)] = 1.0
            else:
                for res, c in pos_counts[pos].items():
                    if res in AA_IX:
                        arr[j, AA_IX[res]] += max(c, 0) / s
        emb[m] = arr
    return all_pos, wuhan, emb


def variable_positions(emb, months, min_change=0.05):
    """Positions where the dominant residue changed by >= min_change in
    at least one consecutive month pair."""
    if len(months) < 2:
        return list(range(len(next(iter(emb.values())))))
    var = set()
    arrays = [emb[m] for m in months if m in emb]
    for a in range(len(arrays) - 1):
        diff = np.abs(arrays[a + 1] - arrays[a]).max(axis=1)
        var |= set(np.where(diff >= min_change)[0])
    return sorted(var)


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------

class ResidueGRU(nn.Module):
    """GRU over monthly population residue embeddings.

    Input per step: flattened (P x 21) residue distribution
    Output: predicted (P x 21) distribution h months ahead
    h enters as a Fourier encoding concatenated to the MLP input.
    """

    def __init__(self, P, d=32, n_fourier=4):
        super().__init__()
        self.P = P
        self.d = d
        self.n_fourier = n_fourier
        self.gru = nn.GRU(P * N_AA, d, batch_first=True)
        # h-conditioning: Fourier features of horizon
        h_dim = 2 * n_fourier
        self.mlp = nn.Sequential(
            nn.Linear(d + h_dim, 128), nn.ReLU(),
            nn.Linear(128, P * N_AA),
        )

    def fourier_h(self, h, device):
        freqs = torch.arange(1, self.n_fourier + 1, dtype=torch.float32,
                             device=device)
        return torch.cat([torch.sin(freqs * h), torch.cos(freqs * h)])

    def forward(self, seq, h_val):
        """seq: (T, P*21) sequence of monthly embeddings
           h_val: scalar horizon in months
        """
        out, _ = self.gru(seq.unsqueeze(0))      # (1, T, d)
        psi = self.fourier_h(h_val, seq.device)
        logits = []
        for t in range(out.shape[1]):
            inp = torch.cat([out[0, t], psi])
            logits.append(self.mlp(inp))          # (P*21,)
        logits = torch.stack(logits)              # (T, P*21)
        return logits.view(out.shape[1], self.P, N_AA)


# ----------------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------------

def recall_at_k(scores, truth_idx, Ks):
    """Fraction of truth positions in top-K by score."""
    order = np.argsort(-np.asarray(scores))
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(len(scores))
    hits = np.asarray(sorted(truth_idx))
    return {K: float(np.mean(rank[hits] < K)) for K in Ks}


def changed_positions(E_now, E_next, threshold=0.1):
    """Positions where dominant residue probability shifted by >= threshold."""
    diff = np.abs(E_next - E_now).max(axis=1)
    return list(np.where(diff >= threshold)[0])


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--vocab",  default="data/processed/vocab_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--train-window", type=int, default=6,
                    help="months used as training context")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--change-thresh", type=float, default=0.05)
    ap.add_argument("--test-end", default="2025-02")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cpu"

    L = load_ladder(a.ladder)
    print("loading events ...")
    monthly = L.load_events(a.events)
    months = sorted(monthly)

    print("loading vocab ...")
    pos_res = load_vocab(a.vocab)
    print(f"  {len(pos_res):,} mutations with position/residue info")

    tr_end = L.TRAIN_END[:7]
    te_end = a.test_end
    all_train = [m for m in months if m <= tr_end]
    test_months = [m for m in months if tr_end < m <= te_end]

    print("building residue embeddings ...")
    all_pos, wuhan, emb = build_embeddings(monthly, pos_res,
                                           all_train + test_months)
    print(f"  {len(all_pos)} positions total")

    # variable positions from training data only
    var_ix = variable_positions(emb, all_train, a.change_thresh)
    P = len(var_ix)
    print(f"  {P} variable positions (changed >= {a.change_thresh} in train)")
    if P < 5:
        print("  TOO FEW -- lower --change-thresh")
        return

    def get_E(m):
        return emb[m][var_ix, :] if m in emb else None

    # historical change frequency per position (the null)
    hist_change = np.zeros(P)
    for a_ in range(len(all_train) - 1):
        E1, E2 = get_E(all_train[a_]), get_E(all_train[a_ + 1])
        if E1 is None or E2 is None:
            continue
        hist_change += np.abs(E2 - E1).max(axis=1)
    hist_change /= max(len(all_train) - 1, 1)

    # build training sequences: sliding windows of length train_window
    model = ResidueGRU(P, a.d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    print(f"\ntraining  (window={a.train_window}m  h={a.horizon}  "
          f"epochs={a.epochs}) ...")
    for ep in range(a.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for start in range(len(all_train) - a.train_window - a.horizon + 1):
            ctx_months = all_train[start:start + a.train_window]
            tgt_month = all_train[start + a.train_window + a.horizon - 1]
            Es = [get_E(m) for m in ctx_months]
            Et = get_E(tgt_month)
            if any(e is None for e in Es) or Et is None:
                continue

            seq = torch.tensor(np.stack(Es), dtype=torch.float32,
                               device=device).view(a.train_window, -1)
            tgt = torch.tensor(Et, dtype=torch.float32, device=device)
            tgt_ix = torch.argmax(tgt, dim=1)   # dominant residue per pos

            logits = model(seq, float(a.horizon))[-1]  # last step pred
            # weight: upweight positions that changed from context end
            E_ctx_end = torch.tensor(Es[-1], dtype=torch.float32)
            change_w = (torch.abs(tgt - E_ctx_end).max(dim=1).values
                        * 10 + 1.0)
            loss = (loss_fn(logits, tgt_ix) * change_w.to(device)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            n_batches += 1

        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}  loss {total_loss/max(n_batches,1):.4f}")

    # evaluation
    model.eval()
    KS = [5, 10, 20, 50]
    print(f"\n[eval] h={a.horizon}")
    print(f"  {'month':9s} {'n_changed':>10s} "
          + " ".join(f"{'null@'+str(K):>8s}" for K in KS)
          + " ".join(f"{'gru@'+str(K):>8s}" for K in KS))
    print("  " + "-" * (12 + 9 * len(KS) * 2))

    rows = []
    for t_ix, m in enumerate(test_months):
        nxt_ix = months.index(m) + a.horizon
        if nxt_ix >= len(months):
            break
        nxt = months[nxt_ix]
        E_now = get_E(m)
        E_nxt = get_E(nxt)
        if E_now is None or E_nxt is None:
            continue

        truth = changed_positions(E_now, E_nxt, a.change_thresh)
        if not truth:
            continue

        # GRU prediction: use last train_window months ending at m
        ctx_ms = [x for x in (all_train + test_months[:t_ix + 1])
                  if x <= m][-a.train_window:]
        Es_ctx = [get_E(x) for x in ctx_ms if get_E(x) is not None]
        if len(Es_ctx) < 2:
            continue
        seq = torch.tensor(np.stack(Es_ctx), dtype=torch.float32,
                           device=device).view(len(Es_ctx), -1)
        with torch.no_grad():
            logits = model(seq, float(a.horizon))[-1]   # (P, 21)
            pred = torch.softmax(logits, dim=1).cpu().numpy()

        # score per position: probability mass NOT on current dominant residue
        cur_dom = np.argmax(E_now, axis=1)
        gru_scores = 1.0 - pred[np.arange(P), cur_dom]
        null_scores = hist_change

        r_null = recall_at_k(null_scores, truth, KS)
        r_gru = recall_at_k(gru_scores, truth, KS)

        row = {"month": m, "n_changed": len(truth),
               "null": r_null, "gru": r_gru}
        rows.append(row)
        print(f"  {m:9s} {len(truth):10d} "
              + " ".join(f"{r_null[K]:8.3f}" for K in KS)
              + " ".join(f"{r_gru[K]:8.3f}" for K in KS))

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    avg_null = {K: float(np.mean([r["null"][K] for r in rows])) for K in KS}
    avg_gru  = {K: float(np.mean([r["gru"][K]  for r in rows])) for K in KS}
    print(f"\n  {'MEAN':9s} {'':10s} "
          + " ".join(f"{avg_null[K]:8.3f}" for K in KS)
          + " ".join(f"{avg_gru[K]:8.3f}" for K in KS))
    print(f"\n  GRU over null:")
    for K in KS:
        print(f"    @{K:2d}  {avg_gru[K] - avg_null[K]:+.3f}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"train_window": a.train_window, "horizon": a.horizon,
                       "P": P, "d": a.d, "seed": a.seed,
                       "avg_null": avg_null, "avg_gru": avg_gru,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
