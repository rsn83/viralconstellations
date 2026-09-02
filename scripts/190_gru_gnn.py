#!/usr/bin/env python3
"""
190 -- GRU + GNN EMISSION OVER 3D CONTACT GRAPH

WHAT THIS ADDS OVER 189
-----------------------
189 predicts each position's next residue independently from h_t:

    p(s_{t+1}[i] | h_t) = softmax(MLP(h_t))_i

This script replaces the independent MLP with a GNN over the 6VXX
structural contact graph:

    p(s_{t+1}[i] | h_t, neighbours) = softmax(GNN(h_t, A))_i

where A is the position-level contact adjacency derived from the 6VXX
crystal structure (CA-CA distance <= 8 Angstroms).

WHY THIS IS BIOLOGICALLY JUSTIFIED
------------------------------------
Epistasis is spatially local. Whether position 501 mutates to Y depends
on what's at positions 498, 505, 417 -- its actual 3D neighbours. That
constraint is invisible to a position-independent emission but captured
by the GNN. 174 measured conditional independence at the SET level; the
GNN captures finer RESIDUE-LEVEL dependencies within the same position
neighbourhood.

CONTACT GRAPH ALIGNMENT
------------------------
The structural prior (outputs/structural_prior.pt) is over 1180
(aa_pos, residue) posres nodes. The model works on P variable spike
positions. Alignment:

    for each variable position p:
        posres_nodes(p) = {node_idx : aa_pos == p}
        structural_neighbours(p) = {q : any posres node in posres_nodes(p)
                                       has a contact with any posres node
                                       in posres_nodes(q)}

This collapses the (position, residue) graph to a position-level graph,
which aligns with the model's variable-position representation.

ABLATION
--------
--no-gnn runs the independent MLP emission (= 189 exactly).
Compare GRU+GNN vs GRU-only to test whether the contact graph adds signal.

USAGE
    # with GNN
    python scripts/190_gru_gnn.py \
        --events data/processed/events_v3.tsv \
        --vocab  data/processed/vocab_v3.tsv \
        --posres data/processed/full_data_graphs_posres/posres_vocab.tsv \
        --struct outputs/structural_prior.pt \
        --train-window 6 --horizon 1 --epochs 50 \
        --out results/gru_gnn.json

    # ablation (= 189)
    python scripts/190_gru_gnn.py ... --no-gnn --out results/gru_nognn.json

GIT
    git add scripts/190_gru_gnn.py
    git commit -m "190: GRU + GNN emission over 3D contact graph vs independent baseline"
    git push
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse helpers from 189
import importlib.util, sys


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------
# CONTACT GRAPH: posres (1180x1180) -> position-level (PxP)
# ----------------------------------------------------------------------------

def build_position_contact(struct_pt, posres_tsv, var_positions):
    """Return a binary adjacency matrix A of shape (P, P) where P =
    len(var_positions), using only resolved structural contacts.

    Two variable positions p, q are adjacent if ANY posres node with
    aa_pos == p has a structural contact with ANY posres node with
    aa_pos == q in the 6VXX crystal structure.
    """
    prior = torch.load(struct_pt, map_location="cpu")
    struct = prior["struct"].numpy()          # (1180, 1180) binary
    resolved = prior["resolved_mask"].numpy() # (1180,) bool

    vocab = pd.read_csv(posres_tsv, sep="\t")
    # map aa_pos -> list of node_idx
    pos_to_nodes = defaultdict(list)
    for _, row in vocab.iterrows():
        pos_to_nodes[int(row["aa_pos"])].append(int(row["node_idx"]))

    P = len(var_positions)
    pos_ix = {p: i for i, p in enumerate(var_positions)}
    A = np.zeros((P, P), dtype=np.float32)

    for i, pi in enumerate(var_positions):
        for j, pj in enumerate(var_positions):
            if i == j:
                continue
            ni_list = [n for n in pos_to_nodes[pi] if resolved[n]]
            nj_list = [n for n in pos_to_nodes[pj] if resolved[n]]
            if not ni_list or not nj_list:
                continue
            # contact if any pair is in contact
            block = struct[np.ix_(ni_list, nj_list)]
            if block.max() > 0:
                A[i, j] = 1.0

    n_edges = int(A.sum()) // 2
    n_resolved = sum(1 for p in var_positions
                     if any(resolved[n] for n in pos_to_nodes[p]))
    print(f"  contact graph: {P} positions, {n_resolved} with structure, "
          f"{n_edges} undirected edges")
    return torch.tensor(A, dtype=torch.float32)


# ----------------------------------------------------------------------------
# GNN LAYER: one-hop message passing with self-loops
# ----------------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X, A_norm):
        """X: (P, in_dim)  A_norm: (P, P) row-normalised + self-loop"""
        return F.relu(self.lin(A_norm @ X))


def normalise_adj(A):
    """Add self-loops, symmetric row normalisation."""
    I = torch.eye(A.shape[0], device=A.device)
    Ah = A + I
    deg = Ah.sum(1, keepdim=True).clamp_min(1.0)
    return Ah / deg


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------

class ResidueGRU_GNN(nn.Module):
    def __init__(self, P, d=32, n_fourier=4, no_gnn=False, gnn_hidden=32):
        super().__init__()
        self.P = P
        self.d = d
        self.n_fourier = n_fourier
        self.no_gnn = no_gnn
        self.gru = nn.GRU(P * 21, d, batch_first=True)
        h_dim = 2 * n_fourier

        if no_gnn:
            self.mlp = nn.Sequential(
                nn.Linear(d + h_dim, 128), nn.ReLU(),
                nn.Linear(128, P * 21),
            )
        else:
            # project h_t to per-position features, then GNN
            self.h_proj = nn.Linear(d + h_dim, P * gnn_hidden)
            self.gcn1 = GCNLayer(gnn_hidden, gnn_hidden)
            self.gcn2 = GCNLayer(gnn_hidden, 21)

    def fourier_h(self, h, device):
        freqs = torch.arange(1, self.n_fourier + 1,
                             dtype=torch.float32, device=device)
        return torch.cat([torch.sin(freqs * h), torch.cos(freqs * h)])

    def forward(self, seq, h_val, A_norm=None):
        out, _ = self.gru(seq.unsqueeze(0))     # (1, T, d)
        psi = self.fourier_h(h_val, seq.device)
        logits = []
        for t in range(out.shape[1]):
            inp = torch.cat([out[0, t], psi])   # (d + h_dim,)
            if self.no_gnn:
                logits.append(self.mlp(inp).view(self.P, 21))
            else:
                X = self.h_proj(inp).view(self.P, -1)  # (P, gnn_hidden)
                X = self.gcn1(X, A_norm)
                X = self.gcn2(X, A_norm)               # (P, 21)
                logits.append(X)
        return torch.stack(logits)  # (T, P, 21)


# ----------------------------------------------------------------------------
# MAIN (identical to 189 except for GNN wiring)
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",  default="data/processed/events_v3.tsv")
    ap.add_argument("--vocab",   default="data/processed/vocab_v3.tsv")
    ap.add_argument("--posres",
                    default="data/processed/full_data_graphs_posres/"
                            "posres_vocab.tsv")
    ap.add_argument("--struct",  default="outputs/structural_prior.pt")
    ap.add_argument("--ladder",  default="scripts/171_ladder.py")
    ap.add_argument("--no-gnn",  action="store_true", dest="no_gnn")
    ap.add_argument("--train-window", type=int, default=6)
    ap.add_argument("--horizon",      type=int, default=1)
    ap.add_argument("--d",            type=int, default=32)
    ap.add_argument("--gnn-hidden",   type=int, default=32)
    ap.add_argument("--epochs",       type=int, default=50)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--change-thresh",type=float, default=0.02,
                    dest="change_thresh")
    ap.add_argument("--test-end",     default="2025-02")
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--out",          default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # load 189 helpers
    L189 = load_mod("scripts/189_gru_residue.py", "gru189")
    L    = load_mod("scripts/171_ladder.py",       "ladder171")

    print("loading events ...")
    monthly = L.load_events(a.events)
    months  = sorted(monthly)

    print("loading vocab ...")
    pos_res = L189.load_vocab(a.vocab)

    tr_end      = L.TRAIN_END[:7]
    all_train   = [m for m in months if m <= tr_end]
    test_months = [m for m in months if tr_end < m <= a.test_end]

    print("building residue embeddings ...")
    all_pos, wuhan, emb = L189.build_embeddings(
        monthly, pos_res, all_train + test_months)
    var_ix = L189.variable_positions(emb, all_train, a.change_thresh)
    P = len(var_ix)
    print(f"  {P} variable positions")
    if P < 5:
        print("  TOO FEW -- lower --change-thresh"); return

    var_aa_pos = [all_pos[i] for i in var_ix]  # actual spike positions

    # contact graph
    A_norm = None
    if not a.no_gnn:
        print("building position-level contact graph ...")
        A = build_position_contact(a.struct, a.posres, var_aa_pos)
        A_norm = normalise_adj(A)

    def get_E(m):
        return emb[m][var_ix, :] if m in emb else None

    hist_change = np.zeros(P)
    for i in range(len(all_train) - 1):
        E1, E2 = get_E(all_train[i]), get_E(all_train[i + 1])
        if E1 is not None and E2 is not None:
            hist_change += np.abs(E2 - E1).max(axis=1)
    hist_change /= max(len(all_train) - 1, 1)

    model = ResidueGRU_GNN(P, a.d, no_gnn=a.no_gnn,
                           gnn_hidden=a.gnn_hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss(reduction="none")

    print(f"\ntraining ({'no-GNN baseline' if a.no_gnn else 'GNN emission'}, "
          f"window={a.train_window}m h={a.horizon} epochs={a.epochs}) ...")
    for ep in range(a.epochs):
        model.train()
        tot, nb = 0.0, 0
        for start in range(len(all_train) - a.train_window - a.horizon + 1):
            ctx_m = all_train[start:start + a.train_window]
            tgt_m = all_train[start + a.train_window + a.horizon - 1]
            Es = [get_E(m) for m in ctx_m]
            Et = get_E(tgt_m)
            if any(e is None for e in Es) or Et is None:
                continue
            seq = torch.tensor(
                np.stack(Es), dtype=torch.float32).view(a.train_window, -1)
            tgt    = torch.tensor(Et, dtype=torch.float32)
            tgt_ix = torch.argmax(tgt, dim=1)
            logits = model(seq, float(a.horizon), A_norm)[-1]  # (P,21)
            E_ctx  = torch.tensor(Es[-1], dtype=torch.float32)
            w = torch.abs(tgt - E_ctx).max(dim=1).values * 10 + 1.0
            loss = (lossf(logits, tgt_ix) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}  loss {tot/max(nb,1):.4f}")

    # evaluation -- identical to 189
    model.eval()
    KS = [5, 10, 20]

    def recall_at_k(scores, truth, Ks):
        order = np.argsort(-np.asarray(scores))
        rank  = np.empty(len(scores), dtype=np.int64)
        rank[order] = np.arange(len(scores))
        hits = np.asarray(sorted(truth))
        return {K: float(np.mean(rank[hits] < K)) for K in Ks}

    def changed(E1, E2, thresh):
        return list(np.where(np.abs(E2-E1).max(axis=1) >= thresh)[0])

    print(f"\n[eval] {'GNN' if not a.no_gnn else 'no-GNN'} h={a.horizon}")
    print(f"  {'month':9s} {'n_ch':>5s} "
          + "".join(f"null@{K:2d} " for K in KS)
          + "".join(f" gnn@{K:2d} " for K in KS))

    rows = []
    for t_ix, m in enumerate(test_months):
        nxt_ix = months.index(m) + a.horizon
        if nxt_ix >= len(months): break
        E_now = get_E(m)
        E_nxt = get_E(months[nxt_ix])
        if E_now is None or E_nxt is None: continue
        truth = changed(E_now, E_nxt, a.change_thresh)
        if not truth: continue

        ctx_ms = [x for x in (all_train + test_months[:t_ix+1])
                  if x <= m][-a.train_window:]
        Es_ctx = [get_E(x) for x in ctx_ms if get_E(x) is not None]
        if len(Es_ctx) < 2: continue

        seq = torch.tensor(np.stack(Es_ctx),
                           dtype=torch.float32).view(len(Es_ctx), -1)
        with torch.no_grad():
            logits = model(seq, float(a.horizon), A_norm)[-1]
            pred   = torch.softmax(logits, dim=1).cpu().numpy()

        cur_dom   = np.argmax(E_now, axis=1)
        gnn_scores = 1.0 - pred[np.arange(P), cur_dom]
        r_null = recall_at_k(hist_change, truth, KS)
        r_gnn  = recall_at_k(gnn_scores,  truth, KS)

        row = {"month": m, "n_changed": len(truth),
               "null": r_null, "gnn": r_gnn}
        rows.append(row)
        print(f"  {m:9s} {len(truth):5d} "
              + "".join(f"{r_null[K]:7.3f} " for K in KS)
              + "".join(f"{r_gnn[K]:7.3f}  " for K in KS))

    if not rows:
        print("  NO EVALUABLE MONTHS"); return

    avg_null = {K: float(np.mean([r["null"][K] for r in rows])) for K in KS}
    avg_gnn  = {K: float(np.mean([r["gnn"][K]  for r in rows])) for K in KS}
    print(f"\n  {'MEAN':9s} {'':5s} "
          + "".join(f"{avg_null[K]:7.3f} " for K in KS)
          + "".join(f"{avg_gnn[K]:7.3f}  " for K in KS))
    print(f"\n  GNN over null:")
    for K in KS:
        print(f"    @{K:2d}  {avg_gnn[K]-avg_null[K]:+.4f}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"no_gnn": a.no_gnn, "train_window": a.train_window,
                       "horizon": a.horizon, "P": P, "d": a.d,
                       "avg_null": avg_null, "avg_gnn": avg_gnn,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
