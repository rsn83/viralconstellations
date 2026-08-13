"""
src/viralconstellations/checks.py

Pure diagnostic functions for two empirical checks:

  1. extinction_vs_frequency: is constellation extinction from O_t
     memoryless (i.i.d. thinning) or frequency/fitness-driven?
  2. cooccurrence_artifact: is the near-zero co-occurrence coefficient in
     LogisticFrontierScorer a real null result or a collinearity artifact?

No dependency on your repo's model/config code -- these take plain
dicts/DataFrames so they can be unit tested and reused from either a
notebook or the two adapter scripts (09_check_extinction_vs_frequency.py,
10_check_cooccurrence_artifact.py).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import average_precision_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ===========================================================================
# Check 1: extinction vs. frequency
# ===========================================================================

def occupied_set(freqs: dict, min_count: int) -> set:
    """Threshold a {constellation: count} dict down to the 'occupied' set."""
    return {c for c, n in freqs.items() if n >= min_count}


def extinction_rate_series(monthly_freqs: list[dict], min_count: int = 5) -> pd.DataFrame:
    """
    For each consecutive window (t, t+1):
        extinct_t = occupied(t) \\ occupied(t+1)
        rate_t    = |extinct_t| / |occupied(t)|
    """
    rows = []
    for t in range(len(monthly_freqs) - 1):
        occ_t = occupied_set(monthly_freqs[t], min_count)
        occ_t1 = occupied_set(monthly_freqs[t + 1], min_count)
        if len(occ_t) == 0:
            continue
        extinct = occ_t - occ_t1
        rows.append({"t": t, "n_occupied": len(occ_t),
                      "n_extinct": len(extinct), "rate": len(extinct) / len(occ_t)})
    return pd.DataFrame(rows).set_index("t")


def extinction_by_frequency_bin(monthly_freqs: list[dict], min_count: int = 5,
                                 n_bins: int = 10) -> pd.DataFrame:
    """
    Long-form DataFrame, one row per (constellation, window) pair that was
    occupied at t: columns [t, freq_t, extinct]. Also prints the pooled
    per-frequency-bin summary.
    """
    records = []
    for t in range(len(monthly_freqs) - 1):
        occ_t = occupied_set(monthly_freqs[t], min_count)
        occ_t1 = occupied_set(monthly_freqs[t + 1], min_count)
        for c in occ_t:
            records.append({"t": t, "freq_t": monthly_freqs[t][c],
                             "extinct": int(c not in occ_t1)})
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No occupied constellations found -- check min_count / input format.")

    df["freq_bin"] = pd.qcut(df["freq_t"], q=n_bins, duplicates="drop")
    summary = (df.groupby("freq_bin", observed=True)
               .agg(n=("extinct", "size"), extinction_rate=("extinct", "mean"),
                    mean_freq=("freq_t", "mean")).reset_index())
    print("\nExtinction rate by parent-frequency bin (pooled across windows):")
    print(summary.to_string(index=False))
    return df


def clustered_logistic_freq_effect(df_long: pd.DataFrame) -> None:
    """P(extinct at t+1) ~ log(freq_t), SEs clustered by window t (GEE)."""
    d = df_long.copy()
    d["log_freq_t"] = np.log(d["freq_t"].astype(float))
    result = smf.gee("extinct ~ log_freq_t", groups="t", data=d,
                      family=sm.families.Binomial()).fit()
    print("\nClustered logistic regression: P(extinct at t+1) ~ log(freq_t)")
    print(result.summary())
    coef, pval = result.params["log_freq_t"], result.pvalues["log_freq_t"]
    print(f"\nlog_freq_t coefficient = {coef:.4f}  (p = {pval:.4g})")
    if pval < 0.05 and coef < 0:
        print("-> Significant negative effect: higher parent frequency predicts "
              "LOWER extinction probability. Extinction looks frequency-driven, "
              "NOT memoryless. The i.i.d.-thinning assumption is likely violated.")
    elif pval < 0.05 and coef > 0:
        print("-> Significant POSITIVE effect (unexpected direction) -- inspect data.")
    else:
        print("-> No significant frequency effect detected. Weakly consistent with "
              "memoryless thinning (does not prove it).")


def plot_extinction_series(rate_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rate_df.index, rate_df["rate"], marker="o", ms=3)
    ax.set_xlabel("window index t"); ax.set_ylabel("extinction rate (t -> t+1)")
    ax.set_title("Extinction rate over time")
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_extinction_by_bin(df_long: pd.DataFrame, n_bins: int, out_path: str):
    d = df_long.copy()
    d["freq_bin"] = pd.qcut(d["freq_t"], q=n_bins, duplicates="drop")
    summary = d.groupby("freq_bin", observed=True).agg(
        extinction_rate=("extinct", "mean"), mean_freq=("freq_t", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(summary["mean_freq"], summary["extinction_rate"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("mean parent frequency in bin (log scale)")
    ax.set_ylabel("extinction rate")
    ax.set_title("Extinction rate vs. parent frequency (pooled across windows)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


# ===========================================================================
# Check 2: co-occurrence artifact
# ===========================================================================

def correlation_and_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].astype(float)
    print("Feature correlation matrix:")
    print(X.corr().round(3).to_string())
    Xc = X.assign(const=1.0)
    vif_df = pd.DataFrame([
        {"feature": col, "VIF": variance_inflation_factor(Xc.values, i)}
        for i, col in enumerate(feature_cols)
    ])
    print("\nVariance Inflation Factors (>~5 meaningful collinearity, >10 severe):")
    print(vif_df.to_string(index=False))
    return vif_df


def fit_and_predict_walk_forward(df: pd.DataFrame, feature_cols: list[str],
                                  window_col: str = "window", label_col: str = "label"
                                  ) -> pd.DataFrame:
    """Train on all windows < w, predict on window w, for each w in order."""
    out = df.copy()
    out["pred"] = np.nan
    windows = sorted(out[window_col].unique())
    for w in windows[1:]:
        train = out[out[window_col] < w]
        test_idx = out[out[window_col] == w].index
        if train[label_col].nunique() < 2:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(train[feature_cols], train[label_col])
        out.loc[test_idx, "pred"] = clf.predict_proba(out.loc[test_idx, feature_cols])[:, 1]
    return out


def held_out_ap_comparison(df: pd.DataFrame, freq_cols: list[str], cooc_col: str,
                            window_col: str = "window", label_col: str = "label",
                            n_bootstrap: int = 2000, seed: int = 0) -> None:
    pred_a = fit_and_predict_walk_forward(df, freq_cols, window_col, label_col)
    pred_b = fit_and_predict_walk_forward(df, freq_cols + [cooc_col], window_col, label_col)

    valid = pred_a["pred"].notna() & pred_b["pred"].notna()
    y = df.loc[valid, label_col].values
    windows = df.loc[valid, window_col].values
    pa, pb = pred_a.loc[valid, "pred"].values, pred_b.loc[valid, "pred"].values

    ap_a, ap_b = average_precision_score(y, pa), average_precision_score(y, pb)
    print(f"\nHeld-out AP, frequency-only:    {ap_a:.4f}")
    print(f"Held-out AP, frequency + cooc:  {ap_b:.4f}")
    print(f"Delta AP:                       {ap_b - ap_a:+.4f}")

    unique_windows = np.unique(windows)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_windows, size=len(unique_windows), replace=True)
        mask = np.isin(windows, sampled)
        if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
            continue
        deltas.append(average_precision_score(y[mask], pb[mask]) -
                       average_precision_score(y[mask], pa[mask]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"Bootstrap 95% CI on Delta AP: [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print("-> CI excludes zero: co-occurrence adds real predictive value.")
    elif hi < 0:
        print("-> CI excludes zero (negative): co-occurrence hurts held-out performance.")
    else:
        print("-> CI includes zero: no evidence co-occurrence adds value beyond frequency.")


def residualize_cooccurrence(df: pd.DataFrame, freq_cols: list[str], cooc_col: str) -> pd.Series:
    X, y = df[freq_cols].astype(float), df[cooc_col].astype(float)
    reg = LinearRegression().fit(X, y)
    print(f"\nR^2 of {cooc_col} ~ frequency features: {reg.score(X, y):.4f} "
          "(fraction of co-occurrence 'explained away' by frequency alone)")
    return y - reg.predict(X)


def held_out_ap_with_residual(df: pd.DataFrame, freq_cols: list[str], cooc_col: str,
                               window_col: str = "window", label_col: str = "label") -> None:
    df = df.copy()
    df["cooc_residual"] = residualize_cooccurrence(df, freq_cols, cooc_col)
    print("\nRe-running held-out AP comparison with RESIDUALIZED co-occurrence:")
    held_out_ap_comparison(df, freq_cols, "cooc_residual", window_col, label_col)


def frequency_matched_stratification(df: pd.DataFrame, freq_col: str, cooc_col: str,
                                      label_col: str = "label", n_freq_bins: int = 5
                                      ) -> pd.DataFrame:
    d = df.copy()
    d["freq_bin"] = pd.qcut(d[freq_col], q=n_freq_bins, duplicates="drop")
    rows = []
    for fbin, sub in d.groupby("freq_bin", observed=True):
        med = sub[cooc_col].median()
        low, high = sub[sub[cooc_col] <= med], sub[sub[cooc_col] > med]
        rows.append({"freq_bin": str(fbin), "n_low_cooc": len(low),
                      "rate_low_cooc": low[label_col].mean() if len(low) else np.nan,
                      "n_high_cooc": len(high),
                      "rate_high_cooc": high[label_col].mean() if len(high) else np.nan})
    result = pd.DataFrame(rows)
    result["rate_diff"] = result["rate_high_cooc"] - result["rate_low_cooc"]
    print("\nAppearance rate by frequency bin, split by co-occurrence (median split):")
    print(result.to_string(index=False))
    print(f"\nMean rate_diff across bins: {result['rate_diff'].mean():+.4f}")
    return result


def plot_stratification(result: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(result)); width = 0.35
    ax.bar(x - width / 2, result["rate_low_cooc"], width, label="low cooc")
    ax.bar(x + width / 2, result["rate_high_cooc"], width, label="high cooc")
    ax.set_xticks(x)
    ax.set_xticklabels(result["freq_bin"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("appearance rate")
    ax.set_title("Appearance rate: high vs low co-occurrence, within frequency bins")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")