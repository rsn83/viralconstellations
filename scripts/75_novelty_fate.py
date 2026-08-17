#!/usr/bin/env python
"""
75_novelty_fate.py

Three questions, in one pass over the data.

A. IS NOVELTY CONCENTRATED AT PARTICULAR POSITIONS?
   Script 74's month-by-month list showed the same positions producing novel
   residues repeatedly: 481 (T, G, R), 445 (S, Y, R, Q), 183 (D, G, V),
   264 (G, N, Y), 487 (T, D). Both 445 and 481 are RBD. Script 59 tested whether
   position history predicts novelty and got AP 0.121 against 0.100 for random,
   but it pooled every position rather than asking whether a small subset is
   hyper-variable. This measures concentration directly, against two nulls:
     uniform      novel events spread evenly over positions that ever mutate
     opportunity  spread in proportion to the residues a position has NOT yet
                  used, which is the fair null since a position that has already
                  produced 10 residues has fewer left to produce

B. DOES TAIL NOVELTY EVER MATTER?
   Every novel mutation appears in the rare tail -- median carrier rank 722 of
   ~1437 sets, carrier mass ~0.0000. The question is whether any of them go
   anywhere. For each one, the maximum population frequency it reaches within
   the following `--horizon` months, and then: is that outcome predictable from
   anything observable at first appearance?
   Features at appearance: carrier rank and mass, carrier size, distance to the
   dominant set, number of distinct backgrounds, and whether the position is one
   of the recurrent ones from part A.
   Scored by AP and by lift against a random ranking of the same events. If
   nothing predicts it, novel mutations are lottery tickets and the tail cannot
   be triaged.

C. WHAT HAPPENS TO DOMINANCE AROUND A TRANSITION?
   An event-aligned trajectory, months -6 to +6 around each known variant month:
   the modal set's share, the number of sets covering half the mass, and the
   mean pairwise distance. This settles the direction of the effect rather than
   arguing about it -- script 60's numbers suggest the dominant share DIPS AT the
   switch and recovers afterwards, not that it falls after.
   Known variant months are used ONLY to align this descriptive section. Nothing
   is fitted to them.

Outputs
-------
outputs/75_positions.csv     novel events per position, with the null comparison
outputs/75_fate.csv          one row per novel mutation, features and outcome
outputs/75_fate_scores.csv   AP and lift for each predictor of success
outputs/75_aligned.csv       event-aligned dominance trajectories

Usage
-----
python scripts/75_novelty_fate.py --min_count 3 --end_month 2024-12
python scripts/75_novelty_fate.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

KNOWN = {
    "2021-01": "Alpha", "2021-06": "Delta", "2022-01": "BA.1",
    "2022-03": "BA.2", "2022-06": "BA.5", "2023-02": "XBB", "2023-12": "JN.1",
}
# spike domains, approximate boundaries, for reading only
DOMAINS = [(13, 305, "NTD"), (306, 330, "linker"), (331, 527, "RBD"),
           (528, 685, "SD1/SD2"), (686, 815, "S2-FP"), (816, 1273, "S2")]


def domain_of(pos):
    for a, b, nm in DOMAINS:
        if a <= pos <= b:
            return nm
    return "other"


# ----------------------------------------------------------------------------
# data
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


def load_vocab(path):
    """node_idx -> (position, residue)."""
    if not os.path.exists(path):
        raise SystemExit(f"need the vocab file: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    idc = next((cols[c] for c in ("node_idx", "node", "id", "idx")
                if c in cols), None)
    pc = next((cols[c] for c in ("aa_pos", "pos", "position") if c in cols), None)
    rc = next((cols[c] for c in ("residue", "res", "aa") if c in cols), None)
    if pc is None or rc is None:
        raise SystemExit(f"no position/residue columns in {path}")
    out = {}
    for i, row in enumerate(df.itertuples(index=False)):
        d = dict(zip(df.columns, row))
        key = int(d[idc]) if idc else i
        out[key] = (int(str(d[pc]).strip()), str(d[rc]).strip())
    return out


def label_freq(occ):
    tot = float(sum(occ.values()))
    nc = defaultdict(float)
    for cs, w in occ.items():
        for l in cs:
            nc[l] += w
    return {l: v / tot for l, v in nc.items()}


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return np.nan
    order = np.lexsort((np.arange(y.size), -s))
    yy = y[order]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, y.size + 1)) * yy).sum() / y.sum())


def gini(counts):
    """Concentration of a count vector. 0 = perfectly even."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def self_test():
    print("self-test")

    assert domain_of(445) == "RBD" and domain_of(481) == "RBD"
    assert domain_of(19) == "NTD" and domain_of(950) == "S2"
    print("  domain assignment                                ok")

    # gini: even -> 0, all mass on one -> near 1
    assert abs(gini([5, 5, 5, 5])) < 1e-9
    assert gini([0, 0, 0, 20]) > 0.7
    print(f"  gini even 0.0, concentrated {gini([0,0,0,20]):.2f}          ok")

    # AP behaves
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    assert abs(average_precision(y, rng.random(5000)) - 0.1) < 0.03
    assert average_precision(y, y + rng.normal(0, 0.05, 5000)) > 0.9
    print("  AP unbiased for random, high for informative     ok")

    # concentration must be detectable: 30 events on 3 of 100 positions
    obs = np.zeros(100)
    obs[[10, 20, 30]] = 10
    null = np.full(100, 0.3)
    assert gini(obs) > gini(null) + 0.5
    print("  concentrated events score far above an even null ok")
    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--horizon", type=int, default=12,
                    help="months to follow a novel mutation forward")
    ap.add_argument("--success", type=float, default=0.01,
                    help="frequency a novel mutation must reach to count as a hit")
    ap.add_argument("--near", type=int, default=3)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    vocab = load_vocab(args.vocab or
                       os.path.join(args.data_dir, "posres_vocab.tsv"))

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    freq = {m: label_freq(occ_by[m]) for m in names}
    present = {m: set(freq[m].keys()) for m in names}

    # ---- find first-ever appearances, causally -----------------------------
    ever = set()
    events = []
    for t, m in enumerate(names):
        for lab in sorted(present[m] - ever, key=str):
            if lab not in vocab:
                continue
            pos, res = vocab[lab]
            occ = occ_by[m]
            tot = float(sum(occ.values()))
            items = sorted(occ.items(), key=lambda kv: -kv[1])
            ranks = {c: r + 1 for r, (c, _) in enumerate(items)}
            modal = items[0][0]
            carriers = sorted([(c, w) for c, w in occ.items() if lab in c],
                              key=lambda cw: -cw[1])
            if not carriers:
                continue
            host, hw = carriers[0]
            # distinct backgrounds, single linkage within `near` edits
            cs = [c for c, _ in carriers]
            nb = 1
            if len(cs) > 1:
                parent = list(range(len(cs)))

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        if len(cs[i] ^ cs[j]) <= args.near:
                            a, b = find(i), find(j)
                            if a != b:
                                parent[a] = b
                nb = len({find(i) for i in range(len(cs))})
            events.append({
                "month": m, "t": t, "label": lab, "pos": pos, "res": res,
                "name": f"{pos}{res}", "domain": domain_of(pos),
                "host_rank": ranks[host], "carrier_mass":
                    float(sum(w for _, w in carriers) / tot),
                "host_size": len(host),
                "dist_to_dominant": len(host ^ modal),
                "n_backgrounds": nb,
                "first_freq": freq[m].get(lab, 0.0),
            })
        ever |= present[m]

    ev = pd.DataFrame(events)
    print(f"first-ever appearances: {len(ev)}")

    # ========================================================================
    # A. positional concentration
    # ========================================================================
    print("\n" + "=" * 84)
    print("A. IS NOVELTY CONCENTRATED AT PARTICULAR POSITIONS?")
    print("=" * 84)

    all_pos = sorted({p for p, _ in vocab.values()})
    res_by_pos = defaultdict(set)
    for p, r in vocab.values():
        res_by_pos[p].add(r)

    cnt = ev.groupby("pos").size()
    obs = np.array([cnt.get(p, 0) for p in all_pos], dtype=float)
    n_ev = obs.sum()
    print(f"positions that ever mutate: {len(all_pos)}")
    print(f"positions producing at least one novel residue: "
          f"{(obs > 0).sum()}")
    print(f"observed Gini across positions: {gini(obs):.4f}")

    # null 1: uniform over positions
    g_uni = []
    for _ in range(args.n_perm):
        draw = rng.multinomial(int(n_ev), np.full(len(all_pos),
                                                  1.0 / len(all_pos)))
        g_uni.append(gini(draw.astype(float)))
    # null 2: proportional to residues not yet used at that position
    opp = np.array([max(20 - len(res_by_pos[p]), 1) for p in all_pos],
                   dtype=float)
    opp = opp / opp.sum()
    g_opp = [gini(rng.multinomial(int(n_ev), opp).astype(float))
             for _ in range(args.n_perm)]

    g_obs = gini(obs)
    print(f"\nnull 1, uniform over positions   : Gini "
          f"{np.mean(g_uni):.4f} +/- {np.std(g_uni):.4f}   "
          f"p = {(np.array(g_uni) >= g_obs).mean():.4f}")
    print(f"null 2, proportional to unused residues: Gini "
          f"{np.mean(g_opp):.4f} +/- {np.std(g_opp):.4f}   "
          f"p = {(np.array(g_opp) >= g_obs).mean():.4f}")
    print("  a Gini well above both nulls means a small set of positions")
    print("  produces most of the novelty, and the tail can be narrowed by")
    print("  position alone.")

    top = cnt.sort_values(ascending=False).head(20)
    pdf = pd.DataFrame({
        "pos": top.index, "n_novel": top.values,
        "domain": [domain_of(p) for p in top.index],
        "residues_ever_seen": [len(res_by_pos[p]) for p in top.index],
        "novel_residues": [", ".join(sorted(
            ev.loc[ev["pos"] == p, "res"].tolist())) for p in top.index],
    })
    pdf.to_csv(f"{args.out_dir}/75_positions.csv", index=False)
    print("\ntop 20 positions by novel residues produced:")
    print(pdf.to_string(index=False))
    print("\nby domain:")
    dd = ev.groupby("domain").size().sort_values(ascending=False)
    for d, n in dd.items():
        share_pos = sum(1 for p in all_pos if domain_of(p) == d) / len(all_pos)
        print(f"  {d:9s} {n:4d} novel ({n/len(ev):.3f} of events) vs "
              f"{share_pos:.3f} of positions")

    # ========================================================================
    # B. fate
    # ========================================================================
    print("\n" + "=" * 84)
    print("B. DOES TAIL NOVELTY EVER MATTER?")
    print("=" * 84)

    recurrent = set(cnt[cnt >= 3].index)
    fate = []
    for _, r in ev.iterrows():
        t = int(r["t"])
        hi = 0.0
        for h in range(1, args.horizon + 1):
            if t + h >= T:
                break
            hi = max(hi, freq[names[t + h]].get(r["label"], 0.0))
        fate.append({**r.to_dict(), "max_freq_after": hi,
                     "success": int(hi >= args.success),
                     "recurrent_position": int(r["pos"] in recurrent),
                     "months_followed": min(args.horizon, T - 1 - t)})
    fdf = pd.DataFrame(fate)
    fdf = fdf[fdf["months_followed"] >= 3]           # need room to observe
    fdf.to_csv(f"{args.out_dir}/75_fate.csv", index=False)

    print(f"novel mutations with at least 3 months of follow-up: {len(fdf)}")
    print(f"reach {args.success:.1%} population frequency within "
          f"{args.horizon} months: {fdf['success'].sum()} "
          f"({fdf['success'].mean():.3f})")
    print(f"median max frequency reached: {fdf['max_freq_after'].median():.6f}")
    print(f"90th percentile:              "
          f"{fdf['max_freq_after'].quantile(0.9):.6f}")

    y = fdf["success"].to_numpy()
    preds = {
        "carrier_mass": fdf["carrier_mass"].to_numpy(),
        "first_freq": fdf["first_freq"].to_numpy(),
        "neg_host_rank": -fdf["host_rank"].to_numpy(),
        "n_backgrounds": fdf["n_backgrounds"].to_numpy(),
        "neg_dist_to_dominant": -fdf["dist_to_dominant"].to_numpy(),
        "host_size": fdf["host_size"].to_numpy(),
        "recurrent_position": fdf["recurrent_position"].to_numpy(),
        "is_RBD": (fdf["domain"] == "RBD").astype(int).to_numpy(),
        "random": rng.random(len(fdf)),
    }
    base = float(y.mean()) if y.size else np.nan
    ap_rand = average_precision(y, preds["random"])
    rows = []
    for nm, s in preds.items():
        a = average_precision(y, s)
        rows.append({"predictor": nm, "ap": a, "base_rate": base,
                     "lift_vs_random": (a / ap_rand
                                        if ap_rand and ap_rand > 0 else np.nan),
                     "n": len(y), "n_success": int(y.sum())})
    sdf = pd.DataFrame(rows).sort_values("ap", ascending=False)
    sdf.to_csv(f"{args.out_dir}/75_fate_scores.csv", index=False)
    print("\nwhich features at first appearance predict later success:")
    print(sdf.round(4).to_string(index=False))
    print("  lift near 1 for every feature -> novel mutations are lottery")
    print("  tickets and the tail cannot be triaged at first sight.")

    print("\nsuccess rate by domain:")
    for d, g in fdf.groupby("domain"):
        print(f"  {d:9s} n={len(g):4d}  success {g['success'].mean():.3f}  "
              f"median max freq {g['max_freq_after'].median():.6f}")
    print("\nsuccess rate at recurrent vs one-off positions:")
    for k, g in fdf.groupby("recurrent_position"):
        lab = "recurrent (3+ novel)" if k else "one-off"
        print(f"  {lab:22s} n={len(g):4d}  success {g['success'].mean():.3f}")

    print("\nnovel mutations that reached the highest frequency:")
    tops = fdf.nlargest(20, "max_freq_after")[
        ["month", "name", "domain", "host_rank", "n_backgrounds",
         "dist_to_dominant", "max_freq_after"]]
    print(tops.round(5).to_string(index=False))

    # ========================================================================
    # C. event-aligned dominance
    # ========================================================================
    print("\n" + "=" * 84)
    print("C. DOMINANCE AROUND A TRANSITION  (descriptive; alignment only)")
    print("=" * 84)
    idx = {m: i for i, m in enumerate(names)}
    arows = []
    for vm, vn in KNOWN.items():
        if vm not in idx:
            continue
        c = idx[vm]
        for off in range(-6, 7):
            k = c + off
            if k < 0 or k >= T:
                continue
            occ = occ_by[names[k]]
            tot = float(sum(occ.values()))
            w = np.sort(np.array(list(occ.values()), dtype=float))[::-1] / tot
            arows.append({
                "variant": vn, "offset": off, "month": names[k],
                "modal_share": float(w[0]),
                "n_sets_half": int(np.searchsorted(np.cumsum(w), 0.5) + 1),
                "n_sets": len(occ),
            })
    adf = pd.DataFrame(arows)
    adf.to_csv(f"{args.out_dir}/75_aligned.csv", index=False)
    piv = adf.pivot_table(index="offset", values=["modal_share", "n_sets_half"],
                          aggfunc="mean")
    print(piv.round(4).to_string())
    print("\nper variant, modal share by offset:")
    print(adf.pivot_table(index="offset", columns="variant",
                          values="modal_share").round(3).to_string())
    print("\n  offset 0 is the variant month. If modal_share dips at 0 and")
    print("  recovers after, the dominant cluster loses share DURING the")
    print("  handover and the new one consolidates afterwards -- not that")
    print("  dominance falls after a takeover.")

    print(f"\nwrote 4 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
