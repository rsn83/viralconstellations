#!/usr/bin/env python3
"""
STEP 1 of the incremental build.

ONE question, nothing else:

    do LEARNED mutation embeddings generalise better than Jaccard backoff?

Rung 2 of the ladder (171) estimates p(D | S) by finding training backgrounds
with Jaccard >= tau to S and averaging their attachment counts. That fails
when a test background has no similar neighbour in training.

This replaces that estimator with

    z_S    = mean({ e_i : i in S })            e_i learned, R^d
    p(D|S) = softmax(MLP(z_S))                 over the full vocabulary

Same input information as rung 2 -- the background S and nothing else.
NO population context, NO momentum, NO horizon conditioning, NO timing.
Those are steps 2-6. Adding any of them here would make the comparison
uninterpretable.

EVALUATION IS DELIBERATELY IDENTICAL TO 171: same months, same candidate
pool, same MIN_COUNT, same log-loss. The only thing that changes is where
the score comes from. The model is wired in as `score3` on the Rungs object
so every other line of the evaluation is literally the same code.

    TARGET: beat rung 2.

If it does not, that is informative -- it means Jaccard backoff already
extracts what is available from the background alone, and the bottleneck
is elsewhere.

USAGE
    python scripts/172_step1_mlp.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --epochs 30 --out results/step1.json

GIT
    git add scripts/172_step1_mlp.py
    git commit -m "172: step 1 -- learned embeddings vs Jaccard backoff for p(D|S)"
    git push
"""

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# load 171 (numeric module name -> importlib, plain import is a syntax error)
# ----------------------------------------------------------------------------

def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------

class AttachMLP(nn.Module):
    """p(D | S), where S is summarised by the mean of its member embeddings.

    Mean pooling is permutation-invariant and -- the point of the whole
    set-model framing -- DEFINED FOR SETS NEVER SEEN IN TRAINING. An
    ID-keyed lookup would be undefined exactly on the backgrounds we need
    to generalise to.
    """

    def __init__(self, n_vocab, dim=64, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_vocab),
        )
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, members, mask):
        """members (B,L) padded mutation indices; mask (B,L) 1 for real."""
        e = self.emb(members) * mask.unsqueeze(-1)
        z = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.net(z)                      # (B, V) logits


# ----------------------------------------------------------------------------
# TRAINING PAIRS
# ----------------------------------------------------------------------------

def build_pairs(pops, months, mut2ix):
    """(background S at t, mutation D added at t+1) for consecutive months.

    NOTE ON ANCESTRY: a set at t+1 that is a radius-1 extension of several
    sets at t generates one pair PER qualifying background. This is a
    reachability relation, not a claim that any particular S was the parent.
    Parents are latent and are never asserted.
    """
    pairs = []
    for a in range(len(months) - 1):
        pT, pN = pops.get(months[a], {}), pops.get(months[a + 1], {})
        if not pT or not pN:
            continue
        by_size = defaultdict(list)
        for S in pT:
            by_size[len(S)].append(S)
        for Sn in pN:
            for S in by_size.get(len(Sn) - 1, ()):
                if S < Sn:
                    D = next(iter(Sn - S))
                    if D in mut2ix and all(i in mut2ix for i in S):
                        pairs.append(([mut2ix[i] for i in S], mut2ix[D]))
    return pairs


def collate(batch, device):
    L = max(len(m) for m, _ in batch)
    mem = torch.zeros(len(batch), L, dtype=torch.long)
    msk = torch.zeros(len(batch), L)
    tgt = torch.zeros(len(batch), dtype=torch.long)
    for k, (m, d) in enumerate(batch):
        mem[k, :len(m)] = torch.tensor(m)
        msk[k, :len(m)] = 1.0
        tgt[k] = d
    return mem.to(device), msk.to(device), tgt.to(device)


def run_epoch(model, pairs, opt, device, bs=256, train=True):
    model.train(train)
    idx = list(range(len(pairs)))
    if train:
        random.shuffle(idx)
    tot = n = 0
    lossf = nn.CrossEntropyLoss()
    for a in range(0, len(idx), bs):
        batch = [pairs[j] for j in idx[a:a + bs]]
        mem, msk, tgt = collate(batch, device)
        with torch.set_grad_enabled(train):
            loss = lossf(model(mem, msk), tgt)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot += loss.item() * len(batch)
        n += len(batch)
    return tot / max(n, 1)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-months", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-end", default=None,
                    help="truncate test window, e.g. 2025-02 for a "
                         "maturity-safe evaluation given the GISAID lag")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    L = load_ladder(args.ladder)

    print("loading ...")
    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)

    tr_end = L.TRAIN_END[:7]
    te_end = args.test_end or L.TEST_END[:7]
    train_months = [m for m in months if m <= tr_end]
    test_months = [m for m in months if tr_end < m <= te_end]

    vocab = sorted({i for m in train_months for S in pops[m] for i in S})
    mut2ix = {m: k for k, m in enumerate(vocab)}
    print(f"  train {len(train_months)}m | test {len(test_months)}m "
          f"| vocab {len(vocab):,}")

    # -- pairs, split by TIME so validation is a forecast, not a shuffle ------
    cut = args.val_months
    fit_months, val_months = train_months[:-cut], train_months[-cut - 1:]
    tr_pairs = build_pairs(pops, fit_months, mut2ix)
    va_pairs = build_pairs(pops, val_months, mut2ix)
    print(f"  pairs: train {len(tr_pairs):,} | val {len(va_pairs):,}")
    if not tr_pairs:
        print("  NO TRAINING PAIRS -- check MIN_COUNT in 171")
        return

    # -- train ---------------------------------------------------------------
    model = AttachMLP(len(vocab), args.dim, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        tl = run_epoch(model, tr_pairs, opt, device, train=True)
        vl = run_epoch(model, va_pairs, opt, device, train=False) \
            if va_pairs else float("nan")
        flag = ""
        if va_pairs and vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            flag = " *"
        else:
            bad += 1
        print(f"  ep {ep:3d}  train {tl:.4f}  val {vl:.4f}{flag}")
        if bad >= 5:
            print("  early stop")
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # -- wire the model in as score3, leave 171's evaluation untouched -------
    R = L.Rungs(pops, train_months)
    print(f"  {len(R.attach):,} backgrounds with attachments (rung 2)")

    logp_cache = {}

    def logp_for(S):
        if S in logp_cache:
            return logp_cache[S]
        ix = [mut2ix[i] for i in S if i in mut2ix]
        if not ix:
            return None
        mem = torch.tensor([ix], dtype=torch.long, device=device)
        msk = torch.ones(1, len(ix), device=device)
        with torch.no_grad():
            lp = torch.log_softmax(model(mem, msk), dim=-1)[0].cpu().numpy()
        logp_cache[S] = lp
        return lp

    def score3(S, D, w):
        lp = logp_for(S)
        if lp is None or D not in mut2ix:
            return R.score2(S, D, w)            # back off, same as rung 2
        return math.log(w + 1e-12) + float(lp[mut2ix[D]])

    R.score3 = score3

    # -- evaluate: IDENTICAL protocol to 171 ---------------------------------
    print(f"\n[eval] horizon={args.horizon}m  test<= {te_end}")
    seen_ever = set()
    for m in train_months:
        seen_ever |= set(pops[m])

    rows = []
    for m in test_months:
        nxt = months.index(m) + args.horizon
        if nxt >= len(months):
            break
        pT, pN = pops[m], pops[months[nxt]]
        new = L.new_constellations(pT, pN, seen_ever)
        seen_ever |= set(pT)
        if not new:
            continue
        cands = L.radius1_candidates(pT, vocab, args.max_bg)
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
        r2 = L.logloss([R.score2(S, D, w) for S, D, w in cands], pos)
        r3 = L.logloss([score3(S, D, w) for S, D, w in cands], pos)
        # rare-background split: does the gain sit where backoff has nothing?
        rare = np.mean([R.attach_profile(S)[1] < 5.0
                        for S, _, _ in cands[:200]])
        rows.append({"month": m, "n_new": len(new), "n_cand": len(cands),
                     "rung2": r2, "step1": r3, "frac_rare_bg": float(rare)})

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    r2 = float(np.mean([r["rung2"] for r in rows]))
    r3 = float(np.mean([r["step1"] for r in rows]))
    per = [r["rung2"] - r["step1"] for r in rows]

    print(f"\n  months            {len(rows)}")
    print(f"  RUNG 2  backoff   {r2:.4f} nats")
    print(f"  STEP 1  learned   {r3:.4f} nats   gain {r2 - r3:+.4f}")
    print(f"  per-month gain    median {np.median(per):+.3f}  "
          f"min {min(per):+.3f}  max {max(per):+.3f}  "
          f"n_pos {sum(x > 0 for x in per)}/{len(per)}")
    print("\n  gain <= 0  =>  Jaccard backoff already extracts what the")
    print("  background alone provides; the bottleneck is not the estimator.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rung2": r2, "step1": r3, "gain": r2 - r3,
                       "n_months": len(rows), "seed": args.seed,
                       "test_end": te_end, "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
