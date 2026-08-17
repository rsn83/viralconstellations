#!/usr/bin/env python
"""
57_vocab_simulator.py

A forecasting model, not a diagnostic.

The object
----------
f(t) in [0,1]^N, the within-month frequency of every mutation label. Both
quantities of interest are functionals of it, by identity:

    occupancy(t)  = sum_i f_i(t)            (mean labels per sequence)
    vocabulary(t) = { i : f_i(t) > eps }    (support at a fixed detection depth)

So they cannot be modelled separately without risking a predicted occupancy that
contradicts the predicted vocabulary. One simulation produces both.

The model
---------
Each label carries a state s_i(t) = (present/absent, current frequency, months
since last seen, months since first seen, run history). ONE transition kernel
P(s_i(t+1) | s_i(t)) is shared across all labels -- that is what makes it
estimable from ~400 labels x 60 months instead of needing a fit per label.

Two coupled heads on the same feature vector:

  presence head   P(present at t+1 | s_i(t))
                  L2 logistic regression. Script 56 measured this: entry AP
                  0.270 against a base rate of 0.052 (6.5x lift), winning 39/40
                  origins, with a strong negative weight on months_since_first_
                  seen -- old labels do not come back, recent ones do. That is
                  duration dependence, so the process is a renewal process and
                  not memoryless.

  frequency head  given present at t+1:
                    persisting label -> logit f(t+1) = logit f(t) + drift + noise
                                        drift regressed on features, noise sd
                                        estimated per frequency bin
                    entering  label -> logit f(t+1) drawn from the empirical
                                        distribution of entrant frequencies

Coupling: a label contributes to occupancy only if presence is sampled, and
enters the vocabulary only if its sampled frequency clears eps.

Forecasting to t+h
------------------
Sample presence, sample frequency, update state, repeat h times. Many
trajectories give a predictive DISTRIBUTION over vocabulary(t+h) and
occupancy(t+h), so intervals and calibration are available -- which a one-step
classifier cannot give.

Evaluation
----------
Rolling-origin (expanding window), h = 1, 3, 6. Fit on months <= t only.
Reported: MAE of the predictive median, and empirical coverage of the 80% and
95% intervals. Baselines: persistence (hold V(t), occupancy(t)) and linear
drift extrapolation of occupancy.

Honest limitations, stated up front
-----------------------------------
- Future sequencing effort is unknown at forecast time, so log n_seqs is held at
  its last observed value. Since support is judged at a fixed depth this mostly
  affects the frequency head, but it is an assumption, not a fact.
- Labels never yet observed cannot be simulated. Their count and frequency mass
  are reported separately as the cold-start ceiling.
- Labels are simulated independently given their own state. They are not
  independent -- linkage is real -- so interval widths are likely too narrow.
  The coverage numbers measure exactly this failure.

Outputs
-------
outputs/57_forecasts.csv     per origin, per horizon, predicted vs actual
outputs/57_summary.csv       MAE and coverage by horizon and model
outputs/57_presence_skill.csv per-label presence AP at each horizon

Usage
-----
python scripts/57_vocab_simulator.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

LOGF_FLOOR = -9.0
FMIN, FMAX = 1e-6, 0.999999
BUF = 12  # months of history kept for windowed features

FEATURES = [
    "present_now", "log_freq_now", "months_since_last_seen",
    "months_since_first_seen", "n_months_present", "n_runs",
    "current_run_length", "longest_gap", "log_freq_last_seen",
    "slope3", "slope6", "occ3", "occ6", "occ12",
    "log_n_seqs", "d_log_n_seqs",
]


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


def node_freqs(occ):
    total = float(sum(occ.values()))
    nc = defaultdict(float)
    for cs, c in occ.items():
        for lab in cs:
            nc[lab] += c
    return {lab: v / total for lab, v in nc.items()}, total


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    draws = rng.multinomial(depth, counts / total)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def _logit(x):
    x = np.clip(x, FMIN, FMAX)
    return np.log(x / (1 - x))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ----------------------------------------------------------------------------
# state tracker: identical code path for real history and simulated futures,
# which is what keeps training features and simulation features consistent
# ----------------------------------------------------------------------------

class StateTracker:
    def __init__(self, n):
        self.n = n
        self.t = -1
        self.Pbuf = np.zeros((n, BUF), dtype=bool)
        self.Fbuf = np.zeros((n, BUF))
        self.since_last = np.zeros(n, dtype=float)
        self.first_t = np.full(n, -1, dtype=float)
        self.ever = np.zeros(n, dtype=bool)
        self.n_present = np.zeros(n)
        self.n_runs = np.zeros(n)
        self.run_len = np.zeros(n)
        self.longest_gap = np.zeros(n)
        self.logf_last = np.full(n, LOGF_FLOOR)

    def copy(self):
        c = StateTracker(self.n)
        c.t = self.t
        for k in ["Pbuf", "Fbuf", "since_last", "first_t", "ever", "n_present",
                  "n_runs", "run_len", "longest_gap", "logf_last"]:
            setattr(c, k, getattr(self, k).copy())
        return c

    def update(self, present, freq):
        """Advance one month with the given presence/frequency vectors."""
        self.t += 1
        prev_since = self.since_last.copy()
        prev_present = self.Pbuf[:, -1].copy() if self.t > 0 else np.zeros(self.n, bool)

        self.Pbuf = np.roll(self.Pbuf, -1, axis=1)
        self.Fbuf = np.roll(self.Fbuf, -1, axis=1)
        self.Pbuf[:, -1] = present
        self.Fbuf[:, -1] = np.where(present, freq, 0.0)

        # a run starts wherever a label is present now but was not last month
        starts = present & ~prev_present
        self.n_runs += starts
        # gaps only count for labels seen before
        gap_now = np.where(starts & self.ever, prev_since, 0.0)
        self.longest_gap = np.maximum(self.longest_gap, gap_now)

        self.since_last = np.where(present, 0.0, prev_since + 1.0)
        newly = present & ~self.ever
        self.first_t = np.where(newly, float(self.t), self.first_t)
        self.ever |= present
        self.n_present += present
        self.run_len = np.where(present, self.run_len + 1.0, 0.0)
        self.logf_last = np.where(present,
                                  np.log(np.clip(freq, 1e-9, None)),
                                  self.logf_last)

    @staticmethod
    def _slope(Pw, Fw):
        """OLS slope of log f over a window, using only months where present."""
        w = Pw.shape[1]
        x = np.arange(w, dtype=float)[None, :]
        m = Pw.astype(float)
        y = np.log(np.clip(Fw, 1e-9, None))
        n = m.sum(axis=1)
        sx = (x * m).sum(axis=1)
        sy = (y * m).sum(axis=1)
        sxy = (x * y * m).sum(axis=1)
        sxx = (x ** 2 * m).sum(axis=1)
        den = n * sxx - sx ** 2
        out = np.zeros(Pw.shape[0])
        ok = (n >= 2) & (np.abs(den) > 1e-12)
        out[ok] = (n[ok] * sxy[ok] - sx[ok] * sy[ok]) / den[ok]
        return out

    def features(self, log_nseqs, d_log_nseqs):
        present = self.Pbuf[:, -1]
        fnow = self.Fbuf[:, -1]
        since_last = np.where(self.ever, self.since_last, float(self.t + 1))
        since_first = np.where(self.ever, self.t - self.first_t, 0.0)
        X = np.column_stack([
            present.astype(float),
            np.where(present, np.log(np.clip(fnow, 1e-9, None)), LOGF_FLOOR),
            since_last, since_first, self.n_present, self.n_runs,
            self.run_len, self.longest_gap, self.logf_last,
            self._slope(self.Pbuf[:, -3:], self.Fbuf[:, -3:]),
            self._slope(self.Pbuf[:, -6:], self.Fbuf[:, -6:]),
            self.Pbuf[:, -3:].mean(axis=1),
            self.Pbuf[:, -6:].mean(axis=1),
            self.Pbuf.mean(axis=1),
            np.full(self.n, log_nseqs),
            np.full(self.n, d_log_nseqs),
        ])
        return X


# ----------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, n_iter=100, tol=1e-7):
    n, p = X.shape
    w = np.zeros(p)
    R = l2 * np.eye(p)
    R[0, 0] = 0.0
    for _ in range(n_iter):
        mu = _sigmoid(X @ w)
        g = X.T @ (mu - y) + R @ w
        s = np.clip(mu * (1 - mu), 1e-6, None)
        H = X.T @ (X * s[:, None]) + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


def fit_ridge(X, y, l2=1.0):
    p = X.shape[1]
    R = l2 * np.eye(p)
    R[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + R, X.T @ y)


def standardiser(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def add_const(X):
    return np.column_stack([np.ones(len(X)), X])


def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, y.size + 1)
    return float((prec * yy).sum() / y.sum())


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
    ap.add_argument("--depth", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--horizons", default="1,3,6")
    ap.add_argument("--n_sim", type=int, default=300)
    ap.add_argument("--min_train", type=int, default=18)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    horizons = [int(h) for h in args.horizons.split(",")]

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    print(f"loaded {len(months)} months: {names[0]} .. {names[-1]}")

    # ---- support at fixed depth, and raw frequencies ------------------------
    support, freq, nseq = {}, {}, {}
    for month, occ in months:
        f, tot = node_freqs(occ)
        freq[month], nseq[month] = f, tot
        seen, nrep = defaultdict(int), 0
        for _ in range(args.reps):
            sub = rarefy(occ, args.depth, args.min_count, rng)
            if sub is None:
                continue
            nrep += 1
            sf, _ = node_freqs(sub)
            for lab in sf:
                seen[lab] += 1
        support[month] = ({l for l, c in seen.items() if c >= nrep / 2}
                          if nrep else set(f.keys()))

    names = [m for m in names if support[m]]
    T = len(names)
    labels = sorted({l for m in names for l in support[m]}, key=str)
    L = {l: i for i, l in enumerate(labels)}
    N = len(labels)
    eps = args.min_count / args.depth
    print(f"labels ever in support: {N};  detection floor f > {eps:.5f}")

    Pmat = np.zeros((N, T), dtype=bool)
    Fmat = np.zeros((N, T))
    for j, m in enumerate(names):
        for lab in support[m]:
            Pmat[L[lab], j] = True
        for lab, v in freq[m].items():
            if lab in L:
                Fmat[L[lab], j] = v
    log_nseqs = np.log(np.array([nseq[m] for m in names]))
    d_log = np.diff(log_nseqs, prepend=log_nseqs[0])

    true_vocab = Pmat.sum(axis=0).astype(float)
    true_occ = np.array([sum(freq[m].values()) for m in names])

    # ---- roll the tracker through real history, caching features per month --
    print("building features ...")
    tracker_at = {}
    Xat = {}
    tr = StateTracker(N)
    for j in range(T):
        tr.update(Pmat[:, j], Fmat[:, j])
        tracker_at[j] = tr.copy()
        Xat[j] = tr.features(log_nseqs[j], d_log[j])

    # ---- rolling-origin evaluation ------------------------------------------
    print("rolling-origin simulation ...\n")
    rows, prows = [], []

    for t in range(args.min_train, T - 1):
        # ---------------- fit on months <= t only ----------------
        Xtr = np.vstack([Xat[j] for j in range(t)])
        ytr = np.concatenate([Pmat[:, j + 1].astype(float) for j in range(t)])
        mu, sd = standardiser(Xtr)
        Xtr_s = add_const((Xtr - mu) / sd)
        w_pres = fit_logistic(Xtr_s, ytr, l2=args.l2)

        # frequency head: drift for labels present at both j and j+1
        Xd, yd = [], []
        entrants = []
        for j in range(t):
            p0, p1 = Pmat[:, j], Pmat[:, j + 1]
            both = p0 & p1
            if both.any():
                Xd.append(Xat[j][both])
                yd.append(_logit(Fmat[both, j + 1]) - _logit(Fmat[both, j]))
            ent = (~p0) & p1
            if ent.any():
                entrants.append(_logit(np.clip(Fmat[ent, j + 1], FMIN, FMAX)))
        if not Xd:
            continue
        Xd = np.vstack(Xd)
        yd = np.concatenate(yd)
        Xd_s = add_const((Xd - mu) / sd)
        w_drift = fit_ridge(Xd_s, yd, l2=args.l2)
        resid = yd - Xd_s @ w_drift
        # heteroscedastic noise: sd by tercile of current log frequency
        lf = Xd[:, FEATURES.index("log_freq_now")]
        cuts = np.quantile(lf, [1 / 3, 2 / 3])
        bins = np.digitize(lf, cuts)
        sd_bin = np.array([resid[bins == b].std() if (bins == b).sum() > 5
                           else resid.std() for b in range(3)])
        ent_pool = (np.concatenate(entrants) if entrants
                    else np.array([_logit(eps)]))

        # ---------------- simulate forward ----------------
        maxh = max(horizons)
        if t + maxh >= T:
            continue
        universe = tracker_at[t].ever.copy()   # causal: seen at or before t

        vocab_sim = {h: np.zeros(args.n_sim) for h in horizons}
        occ_sim = {h: np.zeros(args.n_sim) for h in horizons}
        pres_prob = {h: np.zeros(N) for h in horizons}

        for s in range(args.n_sim):
            st = tracker_at[t].copy()
            for step in range(1, maxh + 1):
                X = st.features(log_nseqs[t], d_log[t])  # effort held constant
                Xs = add_const((X - mu) / sd)
                p = _sigmoid(Xs @ w_pres) * universe
                present = rng.random(N) < p

                prev_p = st.Pbuf[:, -1]
                lf_prev = _logit(np.clip(st.Fbuf[:, -1], FMIN, FMAX))
                drift = Xs @ w_drift
                b = np.digitize(X[:, FEATURES.index("log_freq_now")], cuts)
                noise = rng.normal(0.0, sd_bin[np.clip(b, 0, 2)])
                lf_new = lf_prev + drift + noise
                # entrants get a frequency drawn from the entrant distribution
                is_entry = present & ~prev_p
                if is_entry.any():
                    lf_new[is_entry] = rng.choice(ent_pool, size=is_entry.sum())
                fnew = np.where(present, _sigmoid(lf_new), 0.0)
                fnew = np.clip(fnew, 0.0, FMAX)

                st.update(present, fnew)
                if step in horizons:
                    vocab_sim[step][s] = float((fnew > eps).sum())
                    occ_sim[step][s] = float(fnew.sum())
                    pres_prob[step] += present

        for h in horizons:
            pres_prob[h] /= args.n_sim
            j = t + h
            actual_v, actual_o = true_vocab[j], true_occ[j]
            cold = int((Pmat[:, j] & ~universe).sum())
            cold_mass = float(Fmat[Pmat[:, j] & ~universe, j].sum())

            for mname, med, lo80, hi80, lo95, hi95 in [
                ("simulator",
                 np.median(vocab_sim[h]),
                 *np.quantile(vocab_sim[h], [0.10, 0.90]),
                 *np.quantile(vocab_sim[h], [0.025, 0.975])),
            ]:
                rows.append({
                    "origin": names[t], "target": names[j], "h": h,
                    "model": mname, "quantity": "vocabulary",
                    "actual": actual_v, "pred": med,
                    "lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95,
                    "cold_start_labels": cold, "cold_start_mass": cold_mass,
                })
            rows.append({
                "origin": names[t], "target": names[j], "h": h,
                "model": "persistence", "quantity": "vocabulary",
                "actual": actual_v, "pred": true_vocab[t],
                "lo80": np.nan, "hi80": np.nan, "lo95": np.nan, "hi95": np.nan,
                "cold_start_labels": cold, "cold_start_mass": cold_mass,
            })

            rows.append({
                "origin": names[t], "target": names[j], "h": h,
                "model": "simulator", "quantity": "occupancy",
                "actual": actual_o, "pred": float(np.median(occ_sim[h])),
                "lo80": float(np.quantile(occ_sim[h], 0.10)),
                "hi80": float(np.quantile(occ_sim[h], 0.90)),
                "lo95": float(np.quantile(occ_sim[h], 0.025)),
                "hi95": float(np.quantile(occ_sim[h], 0.975)),
                "cold_start_labels": cold, "cold_start_mass": cold_mass,
            })
            rows.append({
                "origin": names[t], "target": names[j], "h": h,
                "model": "persistence", "quantity": "occupancy",
                "actual": actual_o, "pred": true_occ[t],
                "lo80": np.nan, "hi80": np.nan, "lo95": np.nan, "hi95": np.nan,
                "cold_start_labels": cold, "cold_start_mass": cold_mass,
            })
            # linear drift baseline on occupancy, last 6 months
            wlen = min(6, t + 1)
            xs = np.arange(wlen, dtype=float)
            ys = true_occ[t - wlen + 1:t + 1]
            slope = np.polyfit(xs, ys, 1)[0] if wlen >= 2 else 0.0
            rows.append({
                "origin": names[t], "target": names[j], "h": h,
                "model": "drift", "quantity": "occupancy",
                "actual": actual_o, "pred": float(true_occ[t] + slope * h),
                "lo80": np.nan, "hi80": np.nan, "lo95": np.nan, "hi95": np.nan,
                "cold_start_labels": cold, "cold_start_mass": cold_mass,
            })

            msk = universe
            prows.append({
                "origin": names[t], "h": h,
                "ap": average_precision(Pmat[msk, j].astype(int),
                                        pres_prob[h][msk]),
                "base_rate": float(Pmat[msk, j].mean()),
                "n_universe": int(msk.sum()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/57_forecasts.csv", index=False)
    pdf = pd.DataFrame(prows)
    pdf["lift"] = pdf["ap"] / pdf["base_rate"]
    pdf.to_csv(f"{args.out_dir}/57_presence_skill.csv", index=False)

    # ---- summary ------------------------------------------------------------
    df["abs_err"] = (df["pred"] - df["actual"]).abs()
    df["cov80"] = ((df["actual"] >= df["lo80"]) & (df["actual"] <= df["hi80"]))
    df["cov95"] = ((df["actual"] >= df["lo95"]) & (df["actual"] <= df["hi95"]))

    summ = df.groupby(["quantity", "h", "model"]).agg(
        mae=("abs_err", "mean"),
        mean_actual=("actual", "mean"),
        cov80=("cov80", "mean"),
        cov95=("cov95", "mean"),
        n_origins=("abs_err", "count"),
    ).reset_index()
    summ.loc[summ["model"] != "simulator", ["cov80", "cov95"]] = np.nan
    summ.to_csv(f"{args.out_dir}/57_summary.csv", index=False)

    print("=" * 74)
    print("MULTI-STEP FORECASTS (rolling-origin, expanding window)")
    print("=" * 74)
    for q in ["vocabulary", "occupancy"]:
        print(f"\n--- {q} ---")
        print(summ[summ["quantity"] == q].round(4).to_string(index=False))

    print("\n--- per-label presence skill at horizon h ---")
    print(pdf.groupby("h").agg(ap=("ap", "mean"), base=("base_rate", "mean"),
                               lift=("lift", "mean")).round(4).to_string())

    cs = df.groupby("h").agg(cold_labels=("cold_start_labels", "mean"),
                             cold_mass=("cold_start_mass", "mean")).reset_index()
    print("\n--- cold start (labels never seen by the origin; unreachable) ---")
    print(cs.round(5).to_string(index=False))

    print("\nread:")
    print("  MAE below persistence at h=3 and h=6 means the simulator is a")
    print("  forecaster and not just a restatement of the present.")
    print("  cov80 near 0.80 and cov95 near 0.95 means the predictive intervals")
    print("  are honest. Coverage well below nominal is the expected failure:")
    print("  labels are simulated independently but are linked in real genomes,")
    print("  so the simulator understates joint variance. That gap is the")
    print("  quantitative case for modelling co-occurrence rather than marginals.")
    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
