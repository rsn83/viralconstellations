#!/usr/bin/env python
"""
66_learned_transport.py

Why the previous transport failed
---------------------------------
The coupling between months is unidentifiable, so the COST FUNCTION decides
everything. Scripts 63-65 used edit distance, which by construction rewards not
moving: the result was near-identity pairings, and a model trained on them had
nothing to learn. Script 65 then lost to the marginal null on additions
(AP 0.090 vs 0.250), which is what an uninformative coupling looks like.

Three changes, and they compose:

1. LEARNED COST. The cost of moving mass from set c to set x is
   w . phi(c, x) with phi a handful of interpretable features. w is fitted so
   that pushing month t forward REPRODUCES month t+1 -- not so that movement is
   minimal. Nearest named literature is inverse optimal transport: recover the
   cost from observed behaviour rather than assuming it.

2. ACCUMULATION CONSTRAINT. min-edit-1 says the process accumulates, so
   transport is allowed only to c itself, to c + one circulating mutation, or to
   c - one of its own mutations. This kills the lateral swaps that produced most
   of the noise, and it makes the coupling interpretable: what was acquired.
   It also means candidate targets are GENERATED FROM c ALONE, so the coupling
   becomes a forecast rather than a pairing that needs month t+1 to exist.

3. GROWTH. Balanced OT forces every unit of mass somewhere, inventing
   descendants for lineages that died. Here each source is rescaled by a learned
   growth multiplier before transport, following the Waddington-OT device of
   scaling mass by an estimated growth rate. That multiplier IS the selection
   operator that has been missing: proposal in the cost, selection in the growth.

Objective
---------
No pairing is ever used as a target. The loss is the likelihood of the REAL next
month under the pushforward:

    NLL = - sum_x  observed_mass(x) * log predicted_mass(x)

plus a term on the label marginals. Both are computed against real data, so a
cost that lowers them has genuinely added information rather than manufactured a
target it then predicts. Mass on sets outside the candidate graph is reported
separately as the reachability ceiling.

Model size: ~13 parameters. Deliberately. Every neural attempt in this project
has lost to persistence; the constraint is 30-odd months of data, not capacity.

Evaluation (rolling origin, month t -> t+1)
-------------------------------------------
  NLL of the real next month
  vocabulary Jaccard and occupancy of the pushforward
  AP for APPEARANCE of new constellations, against the marginal null used in
  script 58 -- so the numbers are directly comparable
Baselines:
  persistence     predict month t unchanged
  edit_cost       the unlearned cost scripts 63-65 used, same machinery
  marginal        rank candidates by population frequency alone

Outputs
-------
outputs/66_forecast.csv     per test month, all models
outputs/66_weights.csv      fitted cost and growth weights
outputs/66_summary.csv      pooled

Usage
-----
python scripts/66_learned_transport.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    raise SystemExit("this script needs pytorch: pip install torch")

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

COST_FEATURES = [
    "is_stay", "n_add", "n_drop", "log_rho_added", "log_rho_dropped",
    "pmi_mean_added", "pmi_min_added", "log_size_c", "log_mass_c",
]
GROWTH_FEATURES = ["log_mass_c", "log_size_c", "growth_c", "mean_rho_c"]


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_months(data_dir, min_count, start_month=None, end_month=None):
    files = []
    for fn in os.listdir(data_dir):
        m = MONTH_RE.search(fn)
        if m:
            files.append((m.group(1), os.path.join(data_dir, fn)))
    files.sort()
    out = []
    for month, path in files:
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        with open(path, "rb") as f:
            occ = pickle.load(f)
        occ = {k: v for k, v in occ.items() if v >= min_count}
        if occ:
            out.append((month, occ))
    return out


def label_freq(occ):
    tot = float(sum(occ.values()))
    nc = defaultdict(float)
    for cs, w in occ.items():
        for l in cs:
            nc[l] += w
    return {l: v / tot for l, v in nc.items()}, tot


# ----------------------------------------------------------------------------
# candidate graph for one month, generated from month t alone
# ----------------------------------------------------------------------------

def build_graph(occ_t, prev_occ, rho_t, PMI, lab_index, max_sets,
                pool_size, max_drops):
    """
    Returns:
      sources   list of frozensets
      a         normalised mass per source
      gfeat     source-level growth features
      edges     dict with src, feat, and the target frozenset per edge
    Candidates per source: stay, +m for the top `pool_size` circulating labels,
    -l for its own `max_drops` rarest labels. All generated from month t only.
    """
    items = sorted(occ_t.items(), key=lambda kv: -kv[1])[:max_sets]
    sources = [c for c, _ in items]
    mass = np.array([v for _, v in items], dtype=float)
    a = mass / mass.sum()

    pool = [l for l, _ in sorted(rho_t.items(), key=lambda kv: -kv[1])[:pool_size]]
    pool_r = np.array([rho_t[l] for l in pool])

    src, feats, targets = [], [], []
    for i, c in enumerate(sources):
        cl = [l for l in c if l in lab_index]
        ci = np.array([lab_index[l] for l in cl], dtype=int)
        lsz = np.log(max(len(c), 1))
        lm = np.log(max(a[i], 1e-12))

        # stay
        src.append(i)
        feats.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lsz, lm])
        targets.append(c)

        # single addition
        for k, m in enumerate(pool):
            if m in c:
                continue
            if ci.size and m in lab_index:
                col = PMI[ci, lab_index[m]]
                pm, pmin = float(col.mean()), float(col.min())
            else:
                pm = pmin = 0.0
            src.append(i)
            feats.append([0.0, 1.0, 0.0, float(np.log(max(pool_r[k], 1e-12))),
                          0.0, pm, pmin, lsz, lm])
            targets.append(frozenset(c | {m}))

        # single deletion, of the rarest members
        if len(c) > 1:
            order = sorted(cl, key=lambda l: rho_t.get(l, 0.0))[:max_drops]
            for l in order:
                src.append(i)
                feats.append([0.0, 0.0, 1.0, 0.0,
                              float(np.log(max(rho_t.get(l, 1e-12), 1e-12))),
                              0.0, 0.0, lsz, lm])
                targets.append(frozenset(c - {l}))

    gr = np.array([np.log((occ_t[c] + 1.0) / (prev_occ.get(c, 0.0) + 1.0))
                   for c in sources])
    mean_rho = np.array([np.mean([rho_t.get(l, 0.0) for l in c]) if c else 0.0
                         for c in sources])
    gfeat = np.column_stack([
        np.log(np.clip(a, 1e-12, None)),
        np.log([max(len(c), 1) for c in sources]),
        gr, mean_rho,
    ])

    return sources, a, gfeat, (np.array(src), np.array(feats, dtype=np.float32),
                              targets)


def pushforward(a, gfeat, edges, w_cost, w_growth, target_ids, n_targets):
    """
    Predicted mass per target set. Softmax over each source's outgoing edges of
    -cost, weighted by the source's mass times its growth multiplier.
    """
    src, feat, _ = edges
    cost = feat @ w_cost
    src_t = torch.from_numpy(src).long()

    # per-source softmax over -cost, done with scatter reductions
    neg = -cost
    mx = torch.full((len(a),), -1e30, dtype=neg.dtype)
    mx = mx.scatter_reduce(0, src_t, neg, reduce="amax", include_self=True)
    ex = torch.exp(neg - mx[src_t])
    den = torch.zeros(len(a), dtype=ex.dtype).scatter_add(0, src_t, ex)
    p = ex / den[src_t].clamp(min=1e-30)

    g = torch.exp(torch.from_numpy(gfeat).float() @ w_growth)
    at = torch.from_numpy(a).float() * g
    at = at / at.sum().clamp(min=1e-30)

    contrib = at[src_t] * p
    out = torch.zeros(n_targets, dtype=contrib.dtype)
    out = out.scatter_add(0, torch.from_numpy(target_ids).long(), contrib)
    return out / out.sum().clamp(min=1e-30)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, y.size + 1)) * yy).sum() / y.sum())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else np.nan


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_sets", type=int, default=400)
    ap.add_argument("--pool_size", type=int, default=120)
    ap.add_argument("--max_drops", type=int, default=15)
    ap.add_argument("--train_months", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam_marg", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    rho, tot = {}, {}
    for m in names:
        rho[m], tot[m] = label_freq(occ_by[m])

    # label index and causal PMI, rebuilt per origin
    print("building per-origin candidate graphs ...")
    graphs = {}
    for t in range(T - 1):
        m_t = names[t]
        labs = sorted(rho[m_t].keys(), key=str)
        lab_index = {l: i for i, l in enumerate(labs)}
        n = len(labs)
        CO = np.zeros((n, n), dtype=np.float32)
        marg = np.zeros(n, dtype=np.float32)
        wtot = 0.0
        for j in range(t + 1):
            for cs, wv in occ_by[names[j]].items():
                ix = [lab_index[l] for l in cs if l in lab_index]
                if not ix:
                    continue
                wtot += wv
                arr = np.array(ix)
                marg[arr] += wv
                CO[np.ix_(arr, arr)] += wv
        if wtot <= 0:
            continue
        pm = np.clip(marg / wtot, 1e-9, None)
        PMI = np.log(np.clip(CO / wtot, 1e-12, None) / np.outer(pm, pm))
        np.fill_diagonal(PMI, 0.0)

        prev = occ_by[names[t - 1]] if t > 0 else {}
        sources, a, gfeat, edges = build_graph(
            occ_by[m_t], prev, rho[m_t], PMI, lab_index,
            args.max_sets, args.pool_size, args.max_drops)

        # target index: candidate sets, plus the real next month's sets
        nxt = occ_by[names[t + 1]]
        tgt_list = list(dict.fromkeys(edges[2]))
        tgt_id = {x: i for i, x in enumerate(tgt_list)}
        target_ids = np.array([tgt_id[x] for x in edges[2]])

        obs = np.zeros(len(tgt_list), dtype=np.float32)
        ntot = float(sum(nxt.values()))
        reach_mass = 0.0
        for x, wv in nxt.items():
            if x in tgt_id:
                obs[tgt_id[x]] += wv / ntot
                reach_mass += wv / ntot
        graphs[t] = dict(sources=sources, a=a, gfeat=gfeat, edges=edges,
                         target_ids=target_ids, tgt_list=tgt_list, obs=obs,
                         reach_mass=reach_mass, lab_index=lab_index,
                         n_targets=len(tgt_list))
        if t % 10 == 0:
            print(f"  {m_t}: {len(sources)} sources, {len(edges[0])} edges, "
                  f"{len(tgt_list)} candidate targets, "
                  f"reachable mass {reach_mass:.3f}")

    usable = sorted(graphs)
    print(f"origins with a graph: {len(usable)}")
    rm = np.mean([graphs[t]["reach_mass"] for t in usable])
    print(f"mean reachable mass of the real next month: {rm:.3f}")
    print("  this is the ceiling: mass on sets outside the candidate graph")
    print("  cannot be predicted by any cost on this graph.")

    # ---- fit cost and growth weights on the training origins ---------------
    w_cost = torch.zeros(len(COST_FEATURES), requires_grad=True)
    with torch.no_grad():
        w_cost[COST_FEATURES.index("n_add")] = 2.0
        w_cost[COST_FEATURES.index("n_drop")] = 2.0
    w_growth = torch.zeros(len(GROWTH_FEATURES), requires_grad=True)
    opt = torch.optim.Adam([w_cost, w_growth], lr=args.lr)

    train_t = [t for t in usable if t < args.train_months]
    test_t = [t for t in usable if t >= args.train_months]
    print(f"\ntraining origins: {len(train_t)}  test origins: {len(test_t)}")

    def loss_at(t):
        G = graphs[t]
        pred = pushforward(G["a"], G["gfeat"], G["edges"], w_cost, w_growth,
                           G["target_ids"], G["n_targets"])
        obs = torch.from_numpy(G["obs"])
        nll = -(obs * torch.log(pred.clamp(min=1e-12))).sum()
        # label marginals implied by the pushforward vs observed
        li = G["lab_index"]
        M = torch.zeros(len(G["tgt_list"]), len(li))
        for i, x in enumerate(G["tgt_list"]):
            for l in x:
                if l in li:
                    M[i, li[l]] = 1.0
        pm = pred @ M
        om = torch.zeros(len(li))
        nx = rho[names[t + 1]]
        for l, v in nx.items():
            if l in li:
                om[li[l]] = v
        marg = ((pm - om) ** 2).sum()
        return nll + args.lam_marg * marg

    for ep in range(args.epochs):
        opt.zero_grad()
        tot_loss = sum(loss_at(t) for t in train_t) / max(len(train_t), 1)
        tot_loss.backward()
        opt.step()
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep+1}/{args.epochs}  loss {float(tot_loss):.5f}")

    wc = w_cost.detach().numpy()
    wg = w_growth.detach().numpy()
    pd.DataFrame({"feature": COST_FEATURES + GROWTH_FEATURES,
                  "kind": ["cost"] * len(COST_FEATURES) +
                          ["growth"] * len(GROWTH_FEATURES),
                  "weight": list(wc) + list(wg)}
                 ).to_csv(f"{args.out_dir}/66_weights.csv", index=False)
    print("\nfitted weights:")
    for f, v in zip(COST_FEATURES, wc):
        print(f"  cost   {f:16s} {v:+.4f}")
    for f, v in zip(GROWTH_FEATURES, wg):
        print(f"  growth {f:16s} {v:+.4f}")
    print("  positive cost weight = discourages that move")

    # ---- evaluate on held-out origins --------------------------------------
    w_edit = torch.zeros(len(COST_FEATURES))
    w_edit[COST_FEATURES.index("n_add")] = 2.0
    w_edit[COST_FEATURES.index("n_drop")] = 2.0
    w_zero = torch.zeros(len(GROWTH_FEATURES))

    rows = []
    for t in test_t:
        G = graphs[t]
        H_t = set(occ_by[names[t]].keys())
        nxt = occ_by[names[t + 1]]
        true_new = {x for x in nxt if x not in H_t}
        true_vocab = set(rho[names[t + 1]].keys())
        cand = G["tgt_list"]
        is_new = np.array([1 if (x in nxt and x not in H_t) else 0
                           for x in cand])

        variants = {
            "learned": (w_cost.detach(), w_growth.detach()),
            "edit_cost": (w_edit, w_zero),
        }
        for nm, (wcv, wgv) in variants.items():
            with torch.no_grad():
                pred = pushforward(G["a"], G["gfeat"], G["edges"], wcv, wgv,
                                   G["target_ids"], G["n_targets"]).numpy()
            obs = G["obs"]
            nll = float(-(obs * np.log(np.clip(pred, 1e-12, None))).sum())
            keep = pred > (1.0 / 5000.0)
            pv = set()
            for i in np.flatnonzero(keep):
                pv |= set(cand[i])
            occ_pred = float((pred * np.array([len(x) for x in cand])).sum())
            rows.append({
                "origin": names[t], "target": names[t + 1], "model": nm,
                "nll": nll,
                "vocab_jaccard": jaccard(pv, true_vocab),
                "pred_vocab": len(pv), "true_vocab": len(true_vocab),
                "pred_occupancy": occ_pred,
                "true_occupancy": float(
                    sum(len(x) * w for x, w in nxt.items()) / sum(nxt.values())),
                "new_ap": average_precision(is_new, pred),
                "new_base": float(is_new.mean()),
                "n_candidates": len(cand),
                "reach_mass": G["reach_mass"],
            })

        # marginal null: score candidates by the frequency of the added label
        rt = rho[names[t]]
        s_marg = np.array([max([rt.get(l, 0.0) for l in x], default=0.0)
                           for x in cand])
        rows.append({
            "origin": names[t], "target": names[t + 1], "model": "marginal",
            "nll": np.nan, "vocab_jaccard": np.nan,
            "pred_vocab": np.nan, "true_vocab": len(true_vocab),
            "pred_occupancy": np.nan, "true_occupancy": np.nan,
            "new_ap": average_precision(is_new, s_marg),
            "new_base": float(is_new.mean()),
            "n_candidates": len(cand), "reach_mass": G["reach_mass"],
        })
        # persistence: month t population unchanged
        pv = set(rho[names[t]].keys())
        rows.append({
            "origin": names[t], "target": names[t + 1], "model": "persistence",
            "nll": np.nan,
            "vocab_jaccard": jaccard(pv, true_vocab),
            "pred_vocab": len(pv), "true_vocab": len(true_vocab),
            "pred_occupancy": float(sum(len(x) * w for x, w
                                        in occ_by[names[t]].items()) /
                                    sum(occ_by[names[t]].values())),
            "true_occupancy": float(
                sum(len(x) * w for x, w in nxt.items()) / sum(nxt.values())),
            "new_ap": np.nan, "new_base": float(is_new.mean()),
            "n_candidates": len(cand), "reach_mass": G["reach_mass"],
        })

    df = pd.DataFrame(rows)
    df["new_lift"] = df["new_ap"] / df["new_base"]
    df.to_csv(f"{args.out_dir}/66_forecast.csv", index=False)

    print("\n" + "=" * 74)
    print("HELD-OUT FORECAST, month t -> t+1")
    print("=" * 74)
    summ = df.groupby("model").agg(
        nll=("nll", "mean"),
        vocab_jaccard=("vocab_jaccard", "mean"),
        pred_vocab=("pred_vocab", "mean"), true_vocab=("true_vocab", "mean"),
        pred_occupancy=("pred_occupancy", "mean"),
        true_occupancy=("true_occupancy", "mean"),
        new_ap=("new_ap", "mean"), new_base=("new_base", "mean"),
        origins=("nll", "size"),
    ).reset_index()
    summ["new_lift"] = summ["new_ap"] / summ["new_base"]
    summ.to_csv(f"{args.out_dir}/66_summary.csv", index=False)
    print(summ.round(5).to_string(index=False))

    print("\nthe three comparisons that matter:")
    print("  learned vs edit_cost  -> did fitting the cost beat assuming it")
    print("  learned vs marginal   -> does set context beat frequency alone")
    print("                           (same null as script 58, comparable)")
    print("  learned vs persistence-> vocabulary and occupancy, the metric")
    print("                           every model in this project has lost")
    print(f"\nreachability ceiling: {rm:.3f} of next month's mass is on sets")
    print("in the candidate graph. The rest is unreachable by construction.")
    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
