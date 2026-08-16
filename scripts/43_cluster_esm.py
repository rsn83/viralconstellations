#!/usr/bin/env python
"""
43_cluster_esm.py

Does WHAT a cluster diversifies into matter, or only THAT it diversifies?
CPU, a few minutes.

WHERE THIS COMES FROM
---------------------
Script 41 established, out of sample over 21 walk-forward months:

  g_prev alone      +0.4333
  new_set_share     +0.5782
  both              +0.6054      gain +0.1721, better in 19/21, p=0.0001

against a permutation null centred at 0.067 (sd 0.039, p<0.005). A cluster's
share of sequences in newly-appeared constellations predicts its next-month
growth better than its recent growth rate does.

Script 42 then tested whether that advantage concentrates at variant
transitions and found it does not (p=0.128). So it is a general forecasting
signal, not an early-warning mechanism.

That leaves the mechanistic question open. new_set_share counts new
constellations without looking at their content. If a cluster acquiring
high-scoring mutations grows faster than one acquiring arbitrary ones, there
is a biological mechanism underneath. If not, the signal is purely about the
RATE of diversification.

THE FEATURES
------------
From script 21's reference-anchored per-mutation cache -- computed once against
the Wuhan spike, so they are properties of a substitution and carry no
population information:

  llr_ref   log p(mutant residue) - log p(wild-type) at that position, masked
            marginal. Hie et al.'s grammaticality. Validated by domain:
            HR1 -1.36, FP -0.91, RBD -0.18, NTD -0.09 -- conserved structural
            regions most negative, antigenic loops near zero.
  sem_ref   norm of the embedding change from the substitution, local window
            around the mutated position. Hie et al.'s semantic change.

Aggregated per cluster-month over the mutations carried by NEWLY-APPEARED
constellations only, weighted by sequence count. Four features:

  new_llr_mean, new_llr_max, new_sem_mean, new_sem_max

WHY EXPECT LITTLE
-----------------
These features were tested at the mutation level and came back near zero:
script 24 measured attachment background-dependence at +0.02 over marginal
frequency; script 33 found position choice conditional on background at
+0.000 against a marginal baseline already at AUC 0.97. Neither tested them at
CLUSTER level, which is why this is worth running -- but the prior is that
rate matters and content does not.

THE TEST
--------
Walk-forward, identical protocol to script 41. Four nested models:

  g_prev                          the momentum baseline
  g_prev + new_set_share          the established result, the bar to beat
  g_prev + new_set_share + esm    does content add to rate
  g_prev + esm                    does content work without rate

Usage
-----
  python scripts/43_cluster_esm.py
  python scripts/43_cluster_esm.py --esm outputs/esm_node_features_ref.pkl
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESM_FEATS = ["new_llr_mean", "new_llr_max", "new_sem_mean", "new_sem_max"]


def log(m):
    print(m, flush=True)


def constellations_of(occ):
    out = {}
    for c, v in occ.items():
        out[frozenset(c)] = v if isinstance(v, (int, float)) else 1
    return out


def cluster_sets(sets, thresh, metric, block=400):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    nodes = sorted({m for s in sets for m in s})
    idx = {m: i for i, m in enumerate(nodes)}
    M = np.zeros((len(sets), len(nodes)), dtype=np.float32)
    for i, s in enumerate(sets):
        for m in s:
            M[i, idx[m]] = 1
    size = M.sum(1)
    n = len(sets)
    D = np.empty((n, n), dtype=np.float32)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            inter = M[i0:i1] @ M[j0:j1].T
            si, sj = size[i0:i1][:, None], size[j0:j1][None, :]
            D[i0:i1, j0:j1] = (1.0 - inter / np.maximum(si + sj - inter, 1e-9)
                               if metric == "jaccard" else si + sj - 2.0 * inter)
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform((D + D.T) / 2.0, checks=False),
                            method="average"), t=thresh, criterion="distance") - 1


def spearman(a, b):
    from scipy.stats import rankdata
    ar, br = rankdata(a), rankdata(b)
    if ar.std() < 1e-12 or br.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--esm", default="outputs/esm_node_features_ref.pkl")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--min_seqs", type=int, default=5000)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--max_sets", type=int, default=6000)
    ap.add_argument("--metric", default="edit", choices=["edit", "jaccard"])
    ap.add_argument("--thresh", type=float, default=5.0)
    ap.add_argument("--min_cluster_seqs", type=int, default=50)
    ap.add_argument("--min_train_months", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "43_cluster_esm.csv"))
    args = ap.parse_args()

    # ---- ESM cache ----
    ep = ROOT / args.esm
    if not ep.exists():
        raise SystemExit(f"{ep} not found -- rebuild with script 21 and commit it")
    with open(ep, "rb") as fh:
        blob = pickle.load(fh)
    names = blob["names"]
    X = np.asarray(blob["features"], dtype=np.float32)
    j_llr, j_sem = names.index("llr_ref"), names.index("sem_ref")
    llr, sem = X[:, j_llr], X[:, j_sem]
    log(f"ESM features: {X.shape}, using llr_ref and sem_ref")

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]

    per_month, total = {}, Counter()
    for mo in months:
        with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
            raw = constellations_of(pickle.load(fh))
        f = {c: v for c, v in raw.items()
             if v >= args.min_count and 2 <= len(c) <= args.max_set_size}
        if sum(f.values()) < args.min_seqs:
            continue
        per_month[mo] = f
        for c, v in f.items():
            total[c] += v
    mos = sorted(per_month)
    sets = [c for c, _ in total.most_common(args.max_sets)]
    lab = cluster_sets(sets, args.thresh, args.metric)
    c2k = {c: int(lab[i]) for i, c in enumerate(sets)}
    log(f"{len(mos)} months, {int(lab.max())+1} clusters\n")

    N = X.shape[0]

    def set_scores(c):
        idx = [m for m in c if 0 <= m < N]
        if not idx:
            return 0.0, 0.0, 0.0, 0.0
        return (float(llr[idx].mean()), float(llr[idx].max()),
                float(sem[idx].mean()), float(sem[idx].max()))

    seen = set()
    rows, prev = [], {}
    for i, mo in enumerate(mos):
        f = per_month[mo]
        tot = sum(f.values())
        new_now = {c for c in f if c not in seen}
        by_k = defaultdict(list)
        for c, v in f.items():
            k = c2k.get(c)
            if k is not None:
                by_k[k].append((c, v))
        seen |= set(f)

        cur = {}
        for k, items in by_k.items():
            n_seq = sum(v for _, v in items)
            if n_seq < args.min_cluster_seqs:
                continue
            new_items = [(c, v) for c, v in items if c in new_now]
            new_seq = sum(v for _, v in new_items)
            # ESM aggregates over NEWLY-APPEARED constellations only, weighted
            # by sequence count. Zero when the cluster gained nothing new --
            # correct, and the has_new flag lets a model treat that separately.
            if new_items:
                w = np.array([v for _, v in new_items], float)
                w = w / w.sum()
                sc = np.array([set_scores(c) for c, _ in new_items], float)
                agg = (w[:, None] * sc).sum(0)
                mx = sc.max(0)
                esm = dict(new_llr_mean=agg[0], new_llr_max=mx[1],
                           new_sem_mean=agg[2], new_sem_max=mx[3])
            else:
                esm = dict.fromkeys(ESM_FEATS, 0.0)
            cur[k] = dict(freq=n_seq / tot, n_seq=n_seq,
                          n_sets=float(len(items)),
                          new_set_share=new_seq / n_seq,
                          has_new=float(bool(new_items)),
                          cluster=k, month=mo, month_i=i, depth=tot, **esm)

        for k, c in cur.items():
            p = prev.get(k)
            c["g_prev"] = float(np.log((c["freq"] + 1e-9) / (p["freq"] + 1e-9))) if p else 0.0
        prev = cur
        if i > 0:
            for r in rows:
                if r["month_i"] == i - 1 and r["cluster"] in cur:
                    r["g_next"] = float(np.log(
                        (cur[r["cluster"]]["freq"] + 1e-9) / (r["freq"] + 1e-9)))
        rows.extend(cur.values())

    df = pd.DataFrame([r for r in rows if "g_next" in r])
    if len(df) < 100:
        raise SystemExit(f"only {len(df)} cluster-months")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log(f"{len(df)} cluster-months, {df.cluster.nunique()} clusters, "
        f"{df.month.nunique()} months")
    log(f"cluster-months that gained a new constellation: {df.has_new.mean():.1%}\n")

    from scipy.stats import rankdata
    months_u = sorted(df.month.unique())
    MODELS = {
        "g_prev": ["g_prev"],
        "+new_set_share": ["g_prev", "new_set_share"],
        "+esm": ["g_prev"] + ESM_FEATS,
        "+both": ["g_prev", "new_set_share"] + ESM_FEATS,
    }
    res = []
    for pos, mo in enumerate(months_u):
        if pos < args.min_train_months:
            continue
        tr, te = df[df.month < mo], df[df.month == mo]
        if len(te) < 8 or len(tr) < 60:
            continue

        def fp(cols):
            A = np.column_stack([rankdata(tr[c]) for c in cols] + [np.ones(len(tr))])
            b, *_ = np.linalg.lstsq(A, rankdata(tr.g_next), rcond=None)
            B = np.column_stack([rankdata(te[c]) for c in cols] + [np.ones(len(te))])
            return B @ b

        yt = te.g_next.to_numpy(float)
        row = dict(month=mo, n=len(te))
        for nm, cols in MODELS.items():
            row[nm] = spearman(fp(cols), yt)
        res.append(row)
        log(f"  {mo}  n={len(te):3d} | " +
            " | ".join(f"{nm} {row[nm]:+.3f}" for nm in MODELS))

    if not res:
        raise SystemExit("no usable test months")
    r = pd.DataFrame(res)
    log("\n  " + "-" * 66)
    log(f"  over {len(r)} test months, {int(r.n.sum())} cluster-months")
    for nm in MODELS:
        log(f"    {nm:<18}{r[nm].mean():+.4f}")
    d_esm = r["+both"].mean() - r["+new_set_share"].mean()
    w_esm = int((r["+both"] > r["+new_set_share"]).sum())
    log(f"\n  ESM gain over new_set_share: {d_esm:+.4f}  "
        f"({w_esm}/{len(r)} months)")
    try:
        from scipy.stats import binomtest
        log(f"  sign test p = "
            f"{binomtest(w_esm, len(r), 0.5, alternative='greater').pvalue:.4f}")
    except Exception:
        pass

    log("\n" + "-" * 74)
    log("READ")
    log("-" * 74)
    if d_esm > 0.03 and w_esm > 0.65 * len(r):
        log("  What a cluster diversifies INTO matters, not only that it does.")
        log("  Reference-anchored plausibility of the newly-acquired mutations")
        log("  adds to the rate of diversification -- a mechanism, not just a")
        log("  statistical association.")
    elif d_esm < 0.01:
        log("  ESM content adds nothing over the rate of diversification. The")
        log("  signal is about HOW MUCH a cluster is throwing off new")
        log("  constellations, not which ones. Consistent with script 24 (+0.02)")
        log("  and script 33 (+0.000) at the mutation level, now confirmed at")
        log("  cluster level -- so drop the ESM dependency entirely and report")
        log("  new_set_share alone.")
    else:
        log("  Marginal. Weigh the per-month win count over the mean, and treat")
        log("  a small consistent gain as more credible than a large erratic one.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
