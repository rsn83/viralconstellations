#!/usr/bin/env python
"""
29_establishment_model.py

The actual target, at last.

TASK
----
At month t, generate candidate constellations (frontier: one mutation added to
a set circulating at t). Predict which will EXCEED a frequency threshold by
t+k. Walk-forward: each test month trains only on strictly earlier months.

WHY THIS AND NOT THE EARLIER FORMULATIONS
-----------------------------------------
Script 27: establishing sets have from_0 = 0.79-1.00. They are ABSENT at t,
not merely rare. So there is no trajectory to extrapolate -- copy_forward and
the set-history head are structurally useless here, and the sets must be
NAMED. That is the frontier's job.

Script 28: restricted to months with >= 1000 sequences, 60% of establishment
events at h=3 are one addition from a circulating set, median exactly one
source. In sparse months that figure collapses to 9-14% -- an artefact of not
observing the intermediates, not biology. Hence --min_seqs.

Script 28 also gives the sample size: 95 unique usable events at h=3. That is
enough to EVALUATE on and not enough to train a deep model, which is why this
uses logistic regression and a small GBM -- the model class that worked in
script 26.

MODEL CLASS AND FEATURES
------------------------
A candidate is a pair (source set s, added mutation m) giving c = s + {m}.
Features are deliberately split so the ablation is readable:

  source    log1p count of s, its growth over 1 and 3 months, months present
            in the window, size of s, abundance rank of s
  mutation  marginal frequency of m at t, its 1- and 3-month trend, how many
            circulating sets contain m, log count of m
  pair      how similar s is to the other sets that already contain m
            (leave-one-out cosine over set composition). This is the
            background-dependence term script 24 measured at +0.0195 AUC,
            37/38 months positive. Small but the most consistent signal found.
  esm       optional, from script 21: reference-anchored llr / semantic change
            for m, plus their means over s. Off unless --esm_features given.

BASELINES
---------
  freq_product   product of member marginal frequencies. THE bar -- this is
                 the independence-assumed baseline the project exists to beat.
                 Beating random is not a result.
  source_count   log1p count of the source alone. Tests whether establishment
                 is just "big sets spawn".

Usage
-----
  python scripts/29_establishment_model.py
  python scripts/29_establishment_model.py --esm_features outputs/esm_node_features_ref.pkl
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def constellations_of(occ):
    out = {}
    for c, v in occ.items():
        out[frozenset(c)] = v if isinstance(v, (int, float)) else 1
    return out


def auc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), dtype=float)
    r[o] = np.arange(1, len(s) + 1)
    ss, rs = s[o], r[o]
    k = 0
    while k < len(ss):
        j = k
        while j + 1 < len(ss) and ss[j + 1] == ss[k]:
            j += 1
        if j > k:
            rs[k:j + 1] = rs[k:j + 1].mean()
        k = j + 1
    r[o] = rs
    p, n = int(y.sum()), int((~y).sum())
    if p == 0 or n == 0:
        return float("nan")
    return (r[y].sum() - p * (p + 1) / 2) / (p * n)


def ap(y, s):
    o = np.argsort(-s, kind="stable")
    yy = y[o].astype(float)
    if yy.sum() == 0:
        return float("nan")
    cum = np.cumsum(yy)
    prec = cum / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def recall_at_k(y, s, k):
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="stable")[:k]
    return float(y[o].sum() / y.sum())


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--graphs_dir",
                     default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap_.add_argument("--min_count", type=int, default=3)
    ap_.add_argument("--end_month", type=str, default="2024-12")
    ap_.add_argument("--max_set_size", type=int, default=30)
    ap_.add_argument("--horizon", type=int, default=3)
    ap_.add_argument("--high", type=float, default=0.01,
                     help="candidate counts as established if it exceeds this "
                          "FREQUENCY at t+k (not raw count -- sequencing volume "
                          "swings ~20x across the study period)")
    ap_.add_argument("--min_seqs", type=int, default=1000,
                     help="skip months with fewer sequences. Script 28: dist-1 "
                          "reachability collapses from 60%% to 9-14%% in sparse "
                          "months, an artefact of unobserved intermediates.")
    ap_.add_argument("--window", type=int, default=6)
    ap_.add_argument("--top_sources", type=int, default=0,
                     help="expand from the N most abundant sets. 0 = ALL. "
                          "Capping this destroys coverage: script 28 measured a "
                          "median of exactly ONE source per establishing set, and "
                          "that source is usually rare, not abundant. top_sources=400 "
                          "cut 95 usable events down to 13.")
    ap_.add_argument("--top_muts", type=int, default=0,
                     help="only add the M most frequent mutations. 0 = ALL. "
                          "Safer cap than top_sources if the pool must be reduced.")
    ap_.add_argument("--min_train_months", type=int, default=4)
    ap_.add_argument("--low", type=float, default=0.001,
                     help="a candidate counts as a target only if its frequency at "
                          "t is AT OR BELOW this. Matches script 28. Sets already "
                          "circulating at low frequency are still establishment "
                          "targets -- excluding everything in H_t (the old behaviour) "
                          "dropped ~21%% of events, since script 27 found from_0=0.79 "
                          "at this threshold, not 1.00.")
    ap_.add_argument("--esm_features", type=str, default=None)
    ap_.add_argument("--out", default=str(ROOT / "outputs" / "29_establishment_model.csv"))
    args = ap_.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise SystemExit("needs scikit-learn:  pip install scikit-learn")

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]
    N = len(pd.read_csv(gd / "posres_vocab.tsv", sep="\t"))

    esm = None
    if args.esm_features:
        with open(ROOT / args.esm_features, "rb") as fh:
            blob = pickle.load(fh)
        names = blob["names"]
        keep = [names.index(n) for n in ("llr_ref", "sem_ref") if n in names]
        esm = np.asarray(blob["features"], dtype=np.float32)[:, keep]
        log(f"ESM features: {esm.shape}")

    cache = {}

    def H(mo):
        if mo not in cache:
            with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
                raw = constellations_of(pickle.load(fh))
            filt = {c: v for c, v in raw.items()
                    if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
            tot = sum(filt.values())
            fr = {c: v / max(tot, 1) for c, v in filt.items()}
            mm = Counter()
            for c, v in filt.items():
                for m in c:
                    mm[m] += v
            mf = {m: v / max(tot, 1) for m, v in mm.items()}
            cache[mo] = (filt, fr, tot, mf, mm)
        return cache[mo]

    FEATS = ["src_logc", "src_g1", "src_g3", "src_months", "src_size", "src_rank",
             "mut_freq", "mut_g1", "mut_g3", "mut_nsets", "mut_logc",
             "pair_affinity", "cand_size", "freq_product"]
    if esm is not None:
        FEATS += ["esm_llr_m", "esm_sem_m", "esm_llr_src_mean", "esm_sem_src_mean"]

    def build(i):
        """Feature rows for every frontier candidate at month months[i]."""
        k = args.horizon
        mt, mtk = months[i], months[i + k]
        Ht, ft, tot_t, mft, mmt = H(mt)
        Htk, ftk, tot_tk, _, _ = H(mtk)
        if tot_t < args.min_seqs or tot_tk < args.min_seqs:
            return None
        win = [H(months[j])[0] for j in range(max(0, i - args.window + 1), i + 1)]
        prev1 = H(months[i - 1]) if i >= 1 else H(mt)
        prev3 = H(months[i - 3]) if i >= 3 else H(mt)

        srcs = [c for c, _ in sorted(Ht.items(), key=lambda kv: -kv[1])
                if len(c) < args.max_set_size]
        if args.top_sources and args.top_sources > 0:
            srcs = srcs[:args.top_sources]
        muts = [m for m, _ in mmt.most_common()]
        if args.top_muts and args.top_muts > 0:
            muts = muts[:args.top_muts]
        if len(srcs) < 10 or len(muts) < 5:
            return None
        rank = {c: r / max(len(srcs) - 1, 1) for r, c in enumerate(srcs)}

        # sets containing each mutation, for the pair-affinity term
        contains = {m: [] for m in muts}
        for c in Ht:
            for m in c:
                if m in contains:
                    contains[m].append(c)

        def comp(c):
            v = np.zeros(N, dtype=np.float32)
            for m in c:
                if 0 <= m < N:
                    v[m] = 1.0
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        compc = {c: comp(c) for c in srcs}
        cent = {}
        for m in muts:
            cs = [compc[c] for c in contains[m] if c in compc]
            cent[m] = (np.sum(cs, axis=0), len(cs)) if cs else (np.zeros(N, np.float32), 0)

        rows, ys, keys = [], [], []
        seen_cand = set()
        for s in srcs:
            sc = Ht[s]
            g1 = np.log1p(sc) - np.log1p(prev1[0].get(s, 0.0))
            g3 = np.log1p(sc) - np.log1p(prev3[0].get(s, 0.0))
            nmon = sum(1 for w in win if s in w)
            if esm is not None:
                idx = [m for m in s if 0 <= m < N]
                e_src = esm[idx].mean(axis=0) if idx else np.zeros(esm.shape[1])
            for m in muts:
                if m in s:
                    continue
                c = frozenset(set(s) | {m})
                if len(c) > args.max_set_size:
                    continue
                # A candidate is a target if it is at or below `low` frequency
                # NOW. Excluding everything in H_t removes sets that are present
                # but rare, which script 27 showed are ~21% of establishment
                # events at this threshold.
                if ft.get(c, 0.0) > args.low:
                    continue
                if seen_cand is not None:
                    if c in seen_cand:
                        continue
                    seen_cand.add(c)
                tot_v, nadopt = cent[m]
                aff = float(compc[s] @ tot_v)
                if s in contains[m]:
                    aff -= float(compc[s] @ compc[s])
                    nadopt -= 1
                aff = aff / max(nadopt, 1)

                r = [np.log1p(sc), g1, g3, nmon, len(s), rank[s],
                     mft.get(m, 0.0),
                     mft.get(m, 0.0) - prev1[3].get(m, 0.0),
                     mft.get(m, 0.0) - prev3[3].get(m, 0.0),
                     len(contains[m]), np.log1p(mmt.get(m, 0)),
                     aff, len(c),
                     float(np.prod([mft.get(x, 1e-6) for x in c]) ** (1.0 / len(c)))]
                if esm is not None:
                    em = esm[m] if 0 <= m < N else np.zeros(esm.shape[1])
                    r += [float(em[0]), float(em[1]),
                          float(e_src[0]), float(e_src[1])]
                rows.append(r)
                ys.append(ftk.get(c, 0.0) > args.high)
                keys.append(c)
        if not rows:
            return None
        # COVERAGE: of all establishment events at this month pair, how many did
        # the candidate pool actually contain? This is a hard recall ceiling and
        # must be reported, never hidden. The earlier run silently had ~14%.
        all_ev = {c for c, f in ftk.items()
                  if f > args.high and ft.get(c, 0.0) <= args.low
                  and 2 <= len(c) <= args.max_set_size}
        in_pool = all_ev & set(keys)
        cov = len(in_pool) / len(all_ev) if all_ev else float("nan")
        return (np.asarray(rows, dtype=np.float32), np.asarray(ys, dtype=bool),
                mt, mtk, keys, cov, len(all_ev))

    data = {}
    for i in range(len(months) - args.horizon):
        d = build(i)
        if d is not None and d[1].sum() > 0:
            data[i] = d
            log(f"  {d[2]} -> {d[3]}  cands={len(d[1]):8d}  positives={int(d[1].sum()):4d}"
                f"  events={d[6]:4d}  coverage={d[5]:.1%}")
    log(f"\nusable months: {len(data)}  "
        f"total positives: {sum(int(d[1].sum()) for d in data.values())}\n")
    if len(data) <= args.min_train_months:
        raise SystemExit("not enough usable months; lower --min_seqs or --min_train_months")

    idxs = sorted(data)
    rows = []
    for pos, ti in enumerate(idxs):
        if pos < args.min_train_months:
            continue
        Xtr = np.vstack([data[j][0] for j in idxs[:pos]])
        ytr = np.concatenate([data[j][1] for j in idxs[:pos]])
        X, y, mt, mtk, _, cov, nev = data[ti]
        if ytr.sum() < 10 or y.sum() == 0:
            continue

        sc = StandardScaler().fit(Xtr)
        Xtr_s, X_s = sc.transform(Xtr), sc.transform(X)
        fp = FEATS.index("freq_product")
        srcc = FEATS.index("src_logc")

        scores = {
            "freq_product": X[:, fp],
            "source_count": X[:, srcc],
        }
        scores["logreg"] = LogisticRegression(
            max_iter=3000, class_weight="balanced").fit(Xtr_s, ytr).predict_proba(X_s)[:, 1]
        scores["gbm"] = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=0
        ).fit(Xtr_s, ytr).predict_proba(X_s)[:, 1]

        base = y.mean()
        out = dict(month_t=mt, month_tk=mtk, n_cand=len(y),
                   n_pos=int(y.sum()), base_rate=base,
                   coverage=cov, n_events_total=nev)
        for nm, s in scores.items():
            out[f"{nm}_auc"] = auc(y, s)
            a = ap(y, s)
            out[f"{nm}_ap"] = a
            out[f"{nm}_lift"] = a / base if base > 0 else np.nan
            out[f"{nm}_r50"] = recall_at_k(y, s, 50)
        rows.append(out)
        log(f"  {mt} n={len(y):6d} pos={int(y.sum()):3d} | " +
            " | ".join(f"{nm} AUC={out[f'{nm}_auc']:.3f} lift={out[f'{nm}_lift']:5.1f}"
                       for nm in scores))

    if not rows:
        raise SystemExit("no test months produced results")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("\n" + "=" * 78)
    log(f"SUMMARY over {len(df)} test months, {int(df.n_pos.sum())} positives")
    log("=" * 78)
    log(f"  {'model':<16}{'AUC':>8}{'AP':>9}{'lift':>8}{'recall@50':>11}")
    for nm in ["freq_product", "source_count", "logreg", "gbm"]:
        log(f"  {nm:<16}{df[f'{nm}_auc'].mean():8.3f}{df[f'{nm}_ap'].mean():9.4f}"
            f"{df[f'{nm}_lift'].mean():8.2f}{df[f'{nm}_r50'].mean():11.3f}")
    log(f"\n  mean frontier coverage {df.coverage.mean():.1%} "
        f"-- HARD CEILING on recall; events outside the pool cannot be found")
    log(f"  mean base rate {df.base_rate.mean():.5f}  "
        f"(lift is mechanically capped at {1/max(df.base_rate.mean(),1e-9):.0f}x)")

    log("\n" + "-" * 78)
    log("READ")
    log("-" * 78)
    fq = df["freq_product_auc"].mean()
    for nm in ["logreg", "gbm"]:
        g = df[f"{nm}_auc"].mean() - fq
        w = int((df[f"{nm}_auc"] > df["freq_product_auc"]).sum())
        log(f"  {nm:<8} AUC gain over freq_product: {g:+.4f}   beats it {w}/{len(df)} months")
    log("")
    best = max(df["logreg_auc"].mean(), df["gbm_auc"].mean()) - fq
    if best > 0.05:
        log("  Clear gain over the independence baseline on ESTABLISHMENT, which is")
        log("  the project's central claim. Report per-month wins and a sign test;")
        log("  with this few positives the mean alone is not enough.")
    elif best > 0.01:
        log("  Modest gain, consistent with script 24's +0.0195 on appearance.")
        log("  Check the per-month win count -- consistency matters more than size")
        log("  at this sample size.")
    else:
        log("  No gain over independence on establishment either. Combined with")
        log("  script 24, that is a clean negative on the central hypothesis and")
        log("  should be reported as such rather than engineered around.")
    log("")
    log("  recall@50 is the number a virologist cares about: of the sets that")
    log("  established, how many were in your top 50 candidates.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
