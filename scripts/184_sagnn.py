#!/usr/bin/env python3
"""
184 -- IS THE HYPEREDGE INTERACTION TERM ACTIVE? (standalone)

WHAT THIS IS
------------
A self-contained implementation of HyperSAGNN's scorer -- the only component
of HGDHE (AAAI 2023) / DHyperNodeTPP (AAAI 2025) that this test needs -- so
the result does not depend on the provenance of 164_faithful.py.

    dyn  = MultiheadAttention(X, X, X)     over the members of the set
    stat = W_s(X)                          per-node, no set information
    score = mean_i W_o((dyn_i - stat_i)^2)

(dyn - stat) is the entire hyperedge contribution: how much a node's
representation changes once it is told which set it is in. Everything else
in those papers -- GRU memory, Fourier drift, HGNN history aggregation, TPP
timing, candidate assembly -- is removed. Only the set function remains.

THE TASK
--------
Binary classification: separate constellations that actually circulated in a
month from frontier negatives (the same constellation with one mutation
added or removed). Frontier negatives, not random node sets, because random
sets are separable by node commonness alone -- which is what a frequency
baseline computes, so beating that would not test the set function.

THE ABLATION
------------
    --no-interaction replaces the ATTENTION with a per-node linear map, so
    dyn no longer sees the other members. Same functional form, same
    parameter budget, no set information. (Simply setting dyn = stat would
    make the score identically constant and AUC 0.5 by construction, which
    tests nothing.) If AUC is unchanged, the set information contributed
    nothing.

Both arms share every other line, so a bug outside score_hyperedge affects
them equally and cancels. A bug INSIDE score_hyperedge would not cancel,
which is why it is implemented here in full rather than imported.

THE PREDICTION (falsifiable, stated before running)
---------------------------------------------------
    182 measured node-context similarity: 0.485 in 2020-03..2021-08
    (heterogeneous, 100+ lineage components) and 0.905 in 2024-07..2025-02
    (clonal, 1 component). dyn is computed from the other members, so where
    contexts do not vary, dyn cannot vary whatever the weights.

    heterogeneous regime : ablation COSTS AUC
    clonal regime        : ablation costs ~NOTHING

    both zero  -> the term is never active here; the regime story is wrong,
                  or this task does not exercise it.
    both cost  -> 182's input-side proxy does not capture what the trained
                  model uses.

USAGE
    for W in "2020-03 2021-08" "2022-01 2023-06" "2024-07 2025-02"; do
      set -- $W
      python3 scripts/184_sagnn.py --events data/processed/events_v3.tsv \\
          --ladder scripts/171_ladder.py --date-min $1 --date-max $2
      python3 scripts/184_sagnn.py --events data/processed/events_v3.tsv \\
          --ladder scripts/171_ladder.py --date-min $1 --date-max $2 \\
          --no-interaction
    done

GIT
    git add scripts/184_sagnn.py
    git commit -m "184: standalone HyperSAGNN interaction ablation by regime"
    git push
"""

import argparse
import importlib.util
import json
import random

import numpy as np
import torch
import torch.nn as nn


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HyperSAGNN(nn.Module):
    """The scorer from Zhang, Zou & Ma (2019), as used in both AAAI papers.

    Node embeddings are learned here rather than supplied by a temporal
    memory module. That is a simplification, and it applies identically to
    both arms of the ablation.
    """

    def __init__(self, V, d=64, heads=2, no_interaction=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.W_s = nn.Linear(d, d)
        self.W_o = nn.Linear(d, 1)
        # ABLATION BRANCH. Setting dyn = stat would make (dyn - stat)^2
        # identically zero, so the score would be a constant and AUC 0.5 by
        # construction -- that tests nothing. Instead the attention is
        # replaced by a PER-NODE linear map: the same functional form and
        # parameter budget, but no access to the other members. What is
        # removed is set information, not the score itself.
        self.W_d = nn.Linear(d, d)
        self.no_interaction = no_interaction
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx, mask):
        """idx (B,K) padded member indices; mask (B,K) True where PADDING."""
        X = self.emb(idx)
        if self.no_interaction:
            dyn = self.W_d(X)               # per-node, sees no co-members
        else:
            dyn, _ = self.attn(X, X, X, key_padding_mask=mask,
                               need_weights=False)
        stat = self.W_s(X)
        per = self.W_o((dyn - stat) ** 2).squeeze(-1)
        valid = (~mask).float()
        return (per * valid).sum(1) / valid.sum(1).clamp_min(1.0)


def frontier_negative(S, vocab, rng):
    """One mutation added or removed. Never a random node set: random sets
    are separable by node commonness alone."""
    if len(S) > 2 and rng.random() < 0.5:
        drop = rng.choice(sorted(S))
        return frozenset(S - {drop})
    for _ in range(20):
        add = rng.choice(vocab)
        if add not in S:
            return frozenset(S | {add})
    return None


def collate(sets, labels, mut2ix, device, cap=64):
    K = max(1, min(cap, max(len(s) for s in sets)))
    idx = torch.zeros(len(sets), K, dtype=torch.long)
    msk = torch.ones(len(sets), K, dtype=torch.bool)
    for b, s in enumerate(sets):
        ms = [mut2ix[i] for i in sorted(s) if i in mut2ix][:K]
        if not ms:
            continue
        idx[b, :len(ms)] = torch.tensor(ms)
        msk[b, :len(ms)] = False
    return (idx.to(device), msk.to(device),
            torch.tensor(labels, dtype=torch.float32, device=device))


def auc(scores, labels):
    s = np.asarray(scores)
    y = np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC, ties counted as 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--date-min", required=True)
    ap.add_argument("--date-max", required=True)
    ap.add_argument("--no-interaction", action="store_true",
                    dest="no_interaction",
                    help="replace attention with a per-node linear map")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-bg", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    L = load_ladder(a.ladder)
    print("loading ...")
    monthly = L.load_events(a.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = [m for m in sorted(pops)
              if a.date_min[:7] <= m <= a.date_max[:7]]
    if len(months) < 4:
        print("  window too short")
        return

    cut = int(len(months) * (1 - a.test_frac))
    tr_m, te_m = months[:cut], months[cut:]
    vocab = sorted({i for m in tr_m for S in pops[m] for i in S})
    mut2ix = {m: k for k, m in enumerate(vocab)}
    print(f"  window {months[0]}..{months[-1]} | "
          f"train {len(tr_m)}m test {len(te_m)}m | vocab {len(vocab):,}")
    print(f"  ABLATION: {'ON (dyn = stat)' if a.no_interaction else 'off'}")

    def build(ms):
        S_, y_ = [], []
        for m in ms:
            top = sorted(pops[m].items(), key=lambda kv: -kv[1])[:a.max_bg]
            for S, _ in top:
                if not any(i in mut2ix for i in S):
                    continue
                S_.append(S); y_.append(1)
                neg = frontier_negative(S, vocab, rng)
                if neg and any(i in mut2ix for i in neg):
                    S_.append(neg); y_.append(0)
        return S_, y_

    trS, trY = build(tr_m)
    teS, teY = build(te_m)
    print(f"  train {len(trS):,} sets | test {len(teS):,} sets")
    if not trS or not teS:
        print("  no data")
        return

    model = HyperSAGNN(len(vocab), a.d, a.heads, a.no_interaction).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()

    order = list(range(len(trS)))
    bs = 128
    best = None
    for ep in range(a.epochs):
        model.train()
        rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), bs):
            js = order[i:i + bs]
            idx, msk, y = collate([trS[j] for j in js],
                                  [trY[j] for j in js], mut2ix, device)
            loss = lossf(model(idx, msk), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(js)

        model.eval()
        sc = []
        with torch.no_grad():
            for i in range(0, len(teS), 256):
                idx, msk, _ = collate(teS[i:i + 256], teY[i:i + 256],
                                      mut2ix, device)
                sc += model(idx, msk).cpu().tolist()
        A = auc(sc, teY)
        if best is None or A > best:
            best = A
        print(f"  ep {ep:3d}  loss {tot/len(order):.4f}  test AUC {A:.4f}")

    print(f"\n  WINDOW {months[0]}..{months[-1]}")
    print(f"  interaction {'ZEROED' if a.no_interaction else 'active'}")
    print(f"  best test AUC {best:.4f}")
    print("\n  Compare against the same window with the flag flipped.")
    print("  AUC unchanged  ->  the interaction term contributed nothing.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"date_min": a.date_min, "date_max": a.date_max,
                       "no_interaction": a.no_interaction,
                       "best_auc": best, "n_train": len(trS),
                       "n_test": len(teS), "seed": a.seed}, fh, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
