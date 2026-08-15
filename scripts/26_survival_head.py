#!/usr/bin/env python
"""
26_survival_head.py

Step 1 of the ground-up model: the KEEP/DROP decision.

WHY THIS FIRST
--------------
copy_forward keeps every set in H_t. About 52% survive to t+1, so it scores
recall 1.00 and precision ~0.52 on survival. Every set that dies is a free
win for any model that can tell which ones. No candidate generation, no
frontier, no recall ceiling -- the cheapest place a model can beat the
baseline, and it tests whether a set-level encoder works at all before any
effort goes into births, where script 24 measured the margin at ~0.02 AUC.

WHAT IT DOES
------------
One row per (set s in H_t, month t). Features are properties of the SET, not
of its member mutations pooled -- that was the gap in the hypergraph model,
where the score was a function of member node embeddings only and carried no
per-set history.

  own trajectory   log1p count over the W-month window, growth ratios,
                   months since first observed, months since last absent
  size             number of mutations
  member context   mean / min / max member marginal frequency at t, and the
                   mean member frequency trend

Label: does s appear in H_{t+1}.

Walk-forward evaluation: for each test month, train on all months strictly
before it. No leakage.

BASELINES
---------
  copy_forward   predict every set survives  (recall 1.0 by construction)
  count_only     logistic regression on log1p(count_t) alone -- this is the
                 real bar. If the full feature set does not beat it, the
                 trajectory and member context add nothing over abundance.

Usage
-----
  python scripts/26_survival_head.py
  python scripts/26_survival_head.py --window 6 --min_count 3 --end_month 2024-12
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


def prf(pred, true):
    tp = int((pred & true).sum())
    npred, ntrue = int(pred.sum()), int(true.sum())
    p = tp / npred if npred else float("nan")
    r = tp / ntrue if ntrue else float("nan")
    f = 2 * p * r / (p + r) if (npred and ntrue and (p + r) > 0) else float("nan")
    return p, r, f


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


FEATS = (["logc_w%d" % i for i in range(8)] +
         ["growth_1", "growth_3", "age", "size",
          "mem_freq_mean", "mem_freq_min", "mem_freq_max", "mem_trend_mean",
          "rank_frac"])


def build_rows(months, H, freq, W, t_idx):
    """Feature rows for every set in H_t, labelled by survival to t+1."""
    mt, mth = months[t_idx], months[t_idx + 1]
    Ht, Hth = H(mt), H(mth)
    if not Ht or not Hth:
        return None, None
    win = [H(months[j]) for j in range(max(0, t_idx - W + 1), t_idx + 1)]
    f_t = freq(mt)
    f_prev = freq(months[t_idx - 1]) if t_idx > 0 else f_t

    sets = list(Ht.keys())
    n = len(sets)
    X = np.zeros((n, len(FEATS)), dtype=np.float32)
    y = np.zeros(n, dtype=bool)

    order = {c: i for i, c in enumerate(
        sorted(Ht, key=lambda c: -Ht[c]))}

    for i, s in enumerate(sets):
        traj = [np.log1p(w.get(s, 0.0)) for w in win]
        traj = [0.0] * (8 - len(traj)) + traj[-8:]
        X[i, :8] = traj
        c_now = Ht[s]
        c_1 = win[-2].get(s, 0.0) if len(win) >= 2 else 0.0
        c_3 = win[-4].get(s, 0.0) if len(win) >= 4 else 0.0
        X[i, 8] = np.log1p(c_now) - np.log1p(c_1)
        X[i, 9] = np.log1p(c_now) - np.log1p(c_3)
        X[i, 10] = sum(1 for w in win if s in w)          # months present in window
        X[i, 11] = len(s)
        mf = [f_t.get(m, 0.0) for m in s] or [0.0]
        X[i, 12] = float(np.mean(mf))
        X[i, 13] = float(np.min(mf))
        X[i, 14] = float(np.max(mf))
        X[i, 15] = float(np.mean([f_t.get(m, 0.0) - f_prev.get(m, 0.0) for m in s]))
        X[i, 16] = order[s] / max(n - 1, 1)               # abundance rank, normalised
        y[i] = s in Hth
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", type=str, default=None)
    ap.add_argument("--end_month", type=str, default="2024-12")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--min_train_months", type=int, default=12)
    ap.add_argument("--max_set_size", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "26_survival.csv"))
    args = ap.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise SystemExit("needs scikit-learn:  pip install scikit-learn")

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    if args.start_month:
        months = [m for m in months if m >= args.start_month]
    if args.end_month:
        months = [m for m in months if m <= args.end_month]
    log(f"{len(months)} months: {months[0]} .. {months[-1]}  "
        f"min_count={args.min_count}  window={args.window}\n")

    hc, fc = {}, {}

    def H(mo):
        if mo not in hc:
            with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
                raw = constellations_of(pickle.load(fh))
            hc[mo] = {c: v for c, v in raw.items()
                      if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
        return hc[mo]

    def freq(mo):
        if mo not in fc:
            tot = Counter()
            n = 0
            for c, v in H(mo).items():
                n += v
                for m in c:
                    tot[m] += v
            fc[mo] = {m: v / max(n, 1) for m, v in tot.items()}
        return fc[mo]

    # build all rows once
    data = {}
    for i in range(len(months) - 1):
        X, y = build_rows(months, H, freq, args.window, i)
        if X is not None and len(X) >= 50 and 0 < y.sum() < len(y):
            data[i] = (X, y)
    log(f"usable months: {len(data)}\n")

    rows = []
    idxs = sorted(data)
    for pos, ti in enumerate(idxs):
        if pos < args.min_train_months:
            continue
        Xtr = np.vstack([data[j][0] for j in idxs[:pos]])
        ytr = np.concatenate([data[j][1] for j in idxs[:pos]])
        Xte, yte = data[ti]

        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

        # --- baselines ---
        base_rate = yte.mean()
        p_cf, r_cf, f_cf = prf(np.ones_like(yte, dtype=bool), yte)

        # count only: log1p(count_t) is feature index 7 (last window slot)
        lr_c = LogisticRegression(max_iter=2000).fit(Xtr_s[:, [7]], ytr)
        s_c = lr_c.predict_proba(Xte_s[:, [7]])[:, 1]

        # --- full feature models ---
        lr = LogisticRegression(max_iter=3000).fit(Xtr_s, ytr)
        s_lr = lr.predict_proba(Xte_s)[:, 1]

        gb = GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                        random_state=0).fit(Xtr_s, ytr)
        s_gb = gb.predict_proba(Xte_s)[:, 1]

        # threshold at 0.5 for the keep/drop decision
        out = dict(month_t=months[ti], month_th=months[ti + 1],
                   n_sets=len(yte), survive_rate=float(base_rate),
                   cf_prec=p_cf, cf_rec=r_cf, cf_f1=f_cf)
        for name, s in [("count", s_c), ("lr", s_lr), ("gb", s_gb)]:
            p, r, f = prf(s >= 0.5, yte)
            out[f"{name}_prec"], out[f"{name}_rec"], out[f"{name}_f1"] = p, r, f
            out[f"{name}_auc"] = auc(yte, s)
        rows.append(out)
        log(f"  {months[ti]} n={len(yte):5d} surv={base_rate:.3f} | "
            f"cf F1={f_cf:.3f} | count F1={out['count_f1']:.3f} "
            f"AUC={out['count_auc']:.3f} | lr F1={out['lr_f1']:.3f} "
            f"AUC={out['lr_auc']:.3f} | gb F1={out['gb_f1']:.3f} "
            f"AUC={out['gb_auc']:.3f}")

    if not rows:
        log("no test months (raise --end_month or lower --min_train_months)")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("\n" + "=" * 74)
    log(f"SUMMARY over {len(df)} test months")
    log("=" * 74)
    log(f"  {'model':<16}{'F1':>8}{'prec':>8}{'rec':>8}{'AUC':>8}   beats cf")
    log(f"  {'copy_forward':<16}{df.cf_f1.mean():8.3f}{df.cf_prec.mean():8.3f}"
        f"{df.cf_rec.mean():8.3f}{'--':>8}")
    for nm, lbl in [("count", "count_only"), ("lr", "logreg_full"), ("gb", "gbm_full")]:
        w = int((df[f"{nm}_f1"] > df.cf_f1).sum())
        log(f"  {lbl:<16}{df[f'{nm}_f1'].mean():8.3f}{df[f'{nm}_prec'].mean():8.3f}"
            f"{df[f'{nm}_rec'].mean():8.3f}{df[f'{nm}_auc'].mean():8.3f}"
            f"   {w}/{len(df)}")

    log("\n" + "-" * 74)
    log("READ")
    log("-" * 74)
    log("  copy_forward's recall is 1.000 BY CONSTRUCTION -- it keeps everything.")
    log("  Its precision equals the survival rate. Any model beating it on F1 is")
    log("  correctly dropping sets that die.")
    log("")
    log("  The bar that matters is count_only, not copy_forward. If logreg_full and")
    log("  gbm_full do not beat count_only, then survival is explained by abundance")
    log("  alone and the trajectory / member-context features add nothing -- which")
    log("  would mean a set-level encoder is not buying anything here either.")
    gap_lr = df["lr_auc"].mean() - df["count_auc"].mean()
    gap_gb = df["gb_auc"].mean() - df["count_auc"].mean()
    log(f"\n  AUC gain over count_only:  logreg {gap_lr:+.4f}   gbm {gap_gb:+.4f}")
    if max(gap_lr, gap_gb) > 0.03:
        log("  -> Set-level features carry real signal beyond abundance. The encoder")
        log("     is worth building. Proceed to the naming head.")
    elif max(gap_lr, gap_gb) < 0.01:
        log("  -> Survival is essentially abundance. A learned set encoder adds")
        log("     nothing on this half of the problem.")
    else:
        log("  -> Modest. Check whether the gain is consistent across months before")
        log("     building on it.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
