#!/usr/bin/env python3
"""
182 -- CAN HYPERSAGNN'S INTERACTION TERM CARRY INFORMATION HERE?

THE MECHANISM BEING TESTED
--------------------------
164_faithful.py, score_hyperedge:

    dyn  = SAGNNattn(X, X, X)          attention over the members of the set
    stat = W_sasgnn(X)                 per-node linear map, no set information
    per  = W_out((dyn - stat) ** 2)

(dyn - stat) is the entire hyperedge contribution. It measures how much a
node's representation changes once it is told which set it is in. In HGDHE
(AAAI 2023) this term is worth ~38% MRR over the pairwise equivalent, on
email and congress-bill data.

    dyn is computed from the OTHER MEMBERS of the set.

So if a node's context -- the other members -- is nearly identical in every
set that node appears in, then dyn is nearly constant for that node, and
(dyn - stat) cannot distinguish one set from another. This holds for ANY
learned attention weights, so it can be tested WITHOUT TRAINING ANYTHING.

WHY THIS IS EXPECTED TO FAIL ON THIS DATA
    180: from 2021-09 onward, mean pairwise Jaccard among circulating
         constellations is ~0.90, n_components = 1, frac_dissim = 0.000.
    174: given the background, additions are conditionally independent
         (-0.003 nats), which is the same absence measured from the data
         side.

WHAT IS MEASURED, per month
    ctx_jaccard   for each node, the mean pairwise Jaccard between the
                  contexts (S \\ {i}) of the sets containing it, averaged
                  over nodes. 1.0 means every context is identical and the
                  attention has nothing to attend differently to.
    ctx_sets      mean number of distinct sets a node appears in
    ctx_distinct  mean number of DISTINCT contexts per node

READING IT
    ctx_jaccard near 1.0    the interaction term is uninformative in this
                            regime, whatever weights are learned. The
                            hypergraph machinery is inactive, not badly
                            trained.
    ctx_jaccard well below  the term has variation to exploit; a null result
                            from 164 would be a training or evaluation
                            problem rather than a structural one.

The regime comparison is the point: 180 shows 2020-2021 was heterogeneous
(100+ components) and 2022 onward is clonal. If ctx_jaccard is low in 2021
and near 1.0 in 2024, the conclusion is REGIME-SPECIFIC -- HGDHE's framing
is not refuted, it simply has nothing to grip on in a clonal population.

THIS IS AN INPUT-SIDE NECESSARY CONDITION, NOT A PROOF. Low context variety
means the term cannot help. High context variety does not guarantee it does.
The definitive test is to train 164 and ablate the dynamic branch by setting
dyn = stat; this run is the cheap precondition for that being worthwhile.

USAGE
    python scripts/182_interaction.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --out results/interaction.json

GIT
    git add scripts/182_interaction.py
    git commit -m "182: can HyperSAGNN's interaction term vary on this data"
    git push
"""

import argparse
import importlib.util
import json
import random
from collections import defaultdict

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--max-bg", type=int, default=200)
    ap.add_argument("--max-nodes", type=int, default=300,
                    help="nodes sampled per month")
    ap.add_argument("--max-ctx-pairs", type=int, default=200,
                    help="context pairs sampled per node")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    L = load_ladder(args.ladder)
    print("loading ...")
    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)
    print(f"  {len(months)} months")

    rows = []
    for m in months:
        sets = [S for S, _ in
                sorted(pops[m].items(), key=lambda kv: -kv[1])[:args.max_bg]]
        if len(sets) < 3:
            continue

        # sets containing each node
        by_node = defaultdict(list)
        for S in sets:
            for i in S:
                by_node[i].append(S)
        nodes = [i for i, ss in by_node.items() if len(ss) >= 2]
        if not nodes:
            continue
        if len(nodes) > args.max_nodes:
            nodes = rng.sample(nodes, args.max_nodes)

        jac, n_sets, n_distinct = [], [], []
        for i in nodes:
            ss = by_node[i]
            ctxs = [S - {i} for S in ss]
            n_sets.append(len(ctxs))
            n_distinct.append(len({frozenset(c) for c in ctxs}))
            k = len(ctxs)
            if k < 2:
                continue
            vals = []
            if k * (k - 1) // 2 <= args.max_ctx_pairs:
                for a in range(k):
                    for b in range(a + 1, k):
                        u = ctxs[a] | ctxs[b]
                        vals.append(len(ctxs[a] & ctxs[b]) / len(u)
                                    if u else 1.0)
            else:
                for _ in range(args.max_ctx_pairs):
                    a, b = rng.randrange(k), rng.randrange(k)
                    if a == b:
                        continue
                    u = ctxs[a] | ctxs[b]
                    vals.append(len(ctxs[a] & ctxs[b]) / len(u) if u else 1.0)
            if vals:
                jac.append(float(np.mean(vals)))
        if not jac:
            continue

        rows.append({
            "month": m,
            "n_sets": len(sets),
            "n_nodes": len(nodes),
            "ctx_jaccard": float(np.mean(jac)),
            "ctx_sets": float(np.mean(n_sets)),
            "ctx_distinct": float(np.mean(n_distinct)),
        })

    if not rows:
        print("  NO MONTHS")
        return

    print(f"\n  {'month':9s} {'n_sets':>7s} {'n_nodes':>8s} "
          f"{'ctx_jaccard':>12s} {'sets/node':>10s} {'distinct':>9s}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['month']:9s} {r['n_sets']:7d} {r['n_nodes']:8d} "
              f"{r['ctx_jaccard']:12.3f} {r['ctx_sets']:10.1f} "
              f"{r['ctx_distinct']:9.1f}")

    def span(lo, hi):
        v = [r["ctx_jaccard"] for r in rows if lo <= r["month"] <= hi]
        return float(np.mean(v)) if v else float("nan")

    het = span("2020-03", "2021-08")     # 180: many components
    clo = span("2022-01", "2025-09")     # 180: n_components = 1
    tst = span("2024-07", "2025-02")     # the evaluation window

    print(f"\n  mean ctx_jaccard")
    print(f"    heterogeneous regime 2020-03..2021-08   {het:.3f}")
    print(f"    clonal regime        2022-01..2025-09   {clo:.3f}")
    print(f"    evaluation window    2024-07..2025-02   {tst:.3f}")

    print("\n  ctx_jaccard near 1.0 in the evaluation window means each")
    print("  node sees essentially the same context in every set, so")
    print("  dyn is near-constant per node and (dyn - stat) cannot")
    print("  distinguish sets. The interaction term is inactive by")
    print("  construction, not by undertraining.")
    print("\n  Necessary condition only. Confirm by training 164 and")
    print("  ablating the dynamic branch (dyn = stat).")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"max_bg": args.max_bg,
                       "ctx_jaccard_heterogeneous": het,
                       "ctx_jaccard_clonal": clo,
                       "ctx_jaccard_eval_window": tst,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
