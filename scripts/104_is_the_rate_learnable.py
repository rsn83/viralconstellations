#!/usr/bin/env python3
"""
104_is_the_rate_learnable.py

Script 102 found drifting emissions recover the right mutations (486V, 452R,
the R493Q reversion, and the BA.2.12.1 pair 452Q/704L) but fit rates roughly
thirtyfold too shallow: 486V goes 0.003 -> 0.016 over the training window and
is forecast at 0.018, against 0.538 observed in the test month.

Before building any rate model, this asks whether the signal is there at all.

  If a mutation is flat for eleven months and then jumps after training ends,
  no reparameterisation of the rate recovers it -- the information was never in
  the window.

  If it is visibly climbing during training and the fit is flattening it, then
  the slope is under-estimated and a rate model can help.

WHAT IT PRINTS
  1  the raw monthly frequency of each target mutation, overall and within the
     dominant background, month by month
  2  a slope fitted on the last 3, 6 and all months, to see whether recent
     months carry a steeper signal that a whole-window fit dilutes
  3  where each slope extrapolates to at the test month, against what actually
     happened
  4  the same for every mutation that rose sharply in the test month, so the
     answer is not specific to the four we already know about

Usage:
  python 104_is_the_rate_learnable.py \
      --data-dir data/processed/full_data_graphs_posres \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --npz      results/91_K24.npz --K 24 \
      --train 2021-06:2022-05 --test 2022-06
"""
import argparse, pickle, csv, sys
from pathlib import Path
import numpy as np

EPS = 1e-12


def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def months_in_range(spec):
    if ":" not in spec: return [spec]
    a, b = spec.split(":"); out = [a]
    while out[-1] != b:
        out.append(ym_add(out[-1], 1))
        if len(out) > 300: sys.exit("bad range")
    return out


def load_month(data_dir, ym):
    p = Path(data_dir) / f"{ym}_occupied.pkl"
    if not p.exists(): return None
    obj = pickle.load(open(p, "rb"))
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        return [(frozenset(v), 1) for v in vals]
    items = list(obj)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(frozenset(s), int(c)) for s, c in items]
    return [(frozenset(s), 1) for s in items]


def load_names(path):
    names, V = {}, 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            names[i] = f"{row['aa_pos']}{row['residue'].strip()}"
    return names, V


def build(records, V):
    w = np.array([c for _, c in records], float)
    X = np.zeros((len(records), V), dtype=np.float32)
    for i, (s, _) in enumerate(records):
        X[i, [n for n in s if 0 <= n < V]] = 1.0
    return X, w


def logit(p, floor=1e-4):
    p = np.clip(p, floor, 1 - floor)
    return np.log(p / (1 - p))


def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_slope(y, tail=None):
    """OLS slope of logit(freq) on month index, optionally on the last `tail`."""
    t = np.arange(len(y), dtype=float)
    if tail is not None and tail < len(y):
        t, y = t[-tail:], y[-tail:]
    if len(y) < 2 or np.allclose(y, y[0]): return 0.0, logit(y[-1])
    z = logit(np.asarray(y, float))
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(coef[0]), float(coef[0] * t[-1] + coef[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--npz", default="", help="optional: restrict to the "
                    "dominant background's sequences using a fitted theta")
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--targets", default="486V,452R,493R,452Q,704L")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    names, V = load_names(args.vocab)
    idx = {v: k for k, v in names.items()}
    tr, te = months_in_range(args.train), months_in_range(args.test)

    # ---- monthly frequency, whole population ----
    F = []
    for ym in tr + te:
        r = load_month(args.data_dir, ym)
        X, w = build(r, V)
        F.append((w[:, None] * X).sum(0) / w.sum())
    F = np.array(F)
    T = len(tr)

    # ---- optionally restrict to the dominant background ----
    Fb = None
    if args.npz and Path(args.npz).exists():
        d = np.load(args.npz)
        theta = d["theta"] if "theta" in d else d[f"K{args.K}_theta"]
        Pi = d["Pi"] if "Pi" in d else d[f"K{args.K}_Pi"]
        kstar = int(np.argmax(Pi[-1]))
        lt, lc = np.log(theta + EPS), np.log(1 - theta + EPS)
        Fb = []
        for ym in tr + te:
            X, w = build(load_month(args.data_dir, ym), V)
            lp = X @ (lt - lc).T + lc.sum(1)[None, :]
            m = lp.argmax(1) == kstar
            Fb.append((w[m, None] * X[m]).sum(0) / max(w[m].sum(), 1))
        Fb = np.array(Fb)
        print(f"dominant background: blk{kstar}")

    targets = [t.strip() for t in args.targets.split(",") if t.strip() in idx]

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 88)
    print("1  MONTHLY FREQUENCY DURING TRAINING  (%)  -- is it climbing?")
    print("=" * 88)
    hdr = "".join(f"{m[2:]:>7}" for m in tr) + f"{'| ' + te[0][2:]:>10}"
    print(f"\n  whole population\n  {'mutation':<10}{hdr}")
    for nm in targets:
        n = idx[nm]
        row = "".join(f"{100*F[t, n]:>7.2f}" for t in range(T))
        print(f"  {nm:<10}{row}{'| ' + f'{100*F[T, n]:.2f}':>10}")
    if Fb is not None:
        print(f"\n  within the dominant background\n  {'mutation':<10}{hdr}")
        for nm in targets:
            n = idx[nm]
            row = "".join(f"{100*Fb[t, n]:>7.2f}" for t in range(T))
            print(f"  {nm:<10}{row}{'| ' + f'{100*Fb[T, n]:.2f}':>10}")

    # ---------------------------------------------------------------- 2 & 3
    src = Fb if Fb is not None else F
    print("\n" + "=" * 88)
    print("2  SLOPE FITTED ON DIFFERENT WINDOWS, AND WHERE IT EXTRAPOLATES")
    print("=" * 88)
    print(f"\n  {'mutation':<10}{'window':<10}{'slope/mo':>11}"
          f"{'forecast at ' + te[0]:>20}{'actual':>10}{'ratio':>9}")
    for nm in targets:
        n = idx[nm]
        y = src[:T, n]
        act = src[T, n]
        for lab, tail in (("all", None), ("last 6", 6), ("last 3", 3)):
            sl, last = fit_slope(y, tail)
            pred = sig(last + sl)
            ratio = act / pred if pred > 1e-9 else float("inf")
            print(f"  {nm if lab == 'all' else '':<10}{lab:<10}{sl:>+11.3f}"
                  f"{100*pred:>19.2f}%{100*act:>9.2f}%{ratio:>9.1f}x")
        print()

    # ---------------------------------------------------------------- 4
    print("=" * 88)
    print("4  EVERY MUTATION THAT ROSE SHARPLY IN THE TEST MONTH")
    print("=" * 88)
    rise = src[T] - src[T - 1]
    top = [int(i) for i in np.argsort(-rise)[:args.top] if rise[i] > .05]
    if not top:
        print("\n  none rose by more than 5 points")
    else:
        print(f"\n  {'mutation':<10}{'last train':>12}{'test':>9}{'rise':>9}"
              f"{'slope(all)':>12}{'slope(3)':>11}{'months to reach':>18}")
        for n in top:
            y = src[:T, n]; act = src[T, n]
            s_all, last_all = fit_slope(y)
            s_3, last_3 = fit_slope(y, 3)
            need = ((logit(act) - last_3) / s_3) if abs(s_3) > 1e-6 else np.inf
            print(f"  {names.get(n, n):<10}{100*y[-1]:>11.2f}%{100*act:>8.2f}%"
                  f"{100*(act-y[-1]):>+8.2f}{s_all:>12.3f}{s_3:>11.3f}"
                  f"{need:>18.0f}")

    print("""
HOW TO READ
  Column 'ratio' in section 2 is how far short the extrapolation falls. A ratio
  near 1 means the slope was right; 30 means the rise was thirtyfold faster
  than the fit expected.

  Compare the 'all' and 'last 3' slopes. If the recent slope is much steeper,
  the signal IS in the training window and a whole-window fit is diluting it --
  weighting recent months, or sharing a rate across mutations, should recover
  it. If the two slopes are similar and both far too shallow, the rise happened
  after training ended and no rate model can reach it from this window.

  'months to reach' in section 4 is how long the recent slope would need to
  arrive at the observed level. A number near 1 means the timing is roughly
  right; a large number means the model would be that many months late.
""")


if __name__ == "__main__":
    main()
