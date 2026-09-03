"""
Continuous state-space model for spike forecasting.

Architecture:
  - State: s_t ∈ ℝ^d latent factors
  - Dynamics: s_{t+1} = A s_t + drift + noise
  - Emission: per-position marginals θ_i(s_t) = sigmoid(w_i · s_t + b_i)

Evaluation:
  - Train/test split at 70%
  - Forecast h=1,2,3,4,5,6 months ahead
  - Compare Brier score: model vs persistence baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

DATA_FILE = Path("data/processed/spike_haplotypes_monthly.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("="*70, flush=True)
print("CONTINUOUS STATE-SPACE MODEL FOR SPIKE FORECASTING", flush=True)
print("="*70, flush=True)

print("\n[STEP 1] Loading preprocessed haplotypes...", flush=True)
if not DATA_FILE.exists():
    print(f"  ERROR: {DATA_FILE} not found", flush=True)
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
months = sorted(df['month'].unique())
print(f"  ✓ Loaded {len(df):,} rows", flush=True)
print(f"  Months: {len(months)} ({months[0]} to {months[-1]})", flush=True)

# Extract unique positions
positions = set()
for hap in df['haplotype']:
    for part in hap.split(';'):
        pos_str, _ = part.split(':')
        positions.add(int(pos_str))
positions = sorted(positions)
print(f"  Positions: {len(positions)}", flush=True)

def parse_haplotype(hap_str):
    """Parse 'pos:res;pos:res;...' to dict"""
    result = {}
    for part in hap_str.split(';'):
        pos, res = part.split(':')
        result[int(pos)] = res
    return result

# Compute monthly per-position derived allele frequencies
print("\n[STEP 2] Computing monthly per-position frequencies...", flush=True)
monthly_freqs = {}

for month in months:
    month_data = df[df['month'] == month]
    total_seqs = month_data['count'].sum()
    
    freqs = {}
    for pos in positions:
        counts = {}
        for _, row in month_data.iterrows():
            hap_dict = parse_haplotype(row['haplotype'])
            res = hap_dict.get(pos, 'wt')
            counts[res] = counts.get(res, 0) + row['count']
        
        # Frequency of derived allele (anything non-wt)
        derived_count = sum(c for r, c in counts.items() if r != 'wt')
        freqs[pos] = derived_count / total_seqs
    
    monthly_freqs[month] = freqs

print(f"  ✓ Computed frequencies for {len(monthly_freqs)} months", flush=True)

# Convert to matrix: months x positions
freq_matrix = np.array([[monthly_freqs[m][p] for p in positions] for m in months])
print(f"  Frequency matrix shape: {freq_matrix.shape} (months x positions)", flush=True)

# Fit latent state via SVD
print("\n[STEP 3] Learning latent state representation (SVD)...", flush=True)
d = 3  # State dimension
U, S, Vt = np.linalg.svd(freq_matrix.T, full_matrices=False)
state_matrix = freq_matrix @ U[:, :d]  # months x d

print(f"  State dimension: {d}", flush=True)
print(f"  State matrix shape: {state_matrix.shape}", flush=True)
print(f"  Explained variance (top-{d}): {(S[:d]**2).sum() / (S**2).sum():.1%}", flush=True)

# Fit state dynamics: s_{t+1} = A s_t + drift
print("\n[STEP 4] Fitting state dynamics...", flush=True)
s_t = state_matrix[:-1]
s_t_plus_1 = state_matrix[1:]
s_t_aug = np.hstack([s_t, np.ones((len(s_t), 1))])

A_aug = np.linalg.lstsq(s_t_aug, s_t_plus_1, rcond=None)[0]
A = A_aug[:-1, :]
drift = A_aug[-1, :]

residuals = s_t_plus_1 - s_t @ A.T - drift
noise_cov = np.cov(residuals.T) if residuals.shape[1] > 1 else np.cov(residuals.T).reshape(1, 1)

print(f"  A matrix shape: {A.shape}", flush=True)
print(f"  Drift vector: {drift}", flush=True)
print(f"  Residual std: {np.sqrt(np.diag(noise_cov))}", flush=True)

# Fit emission: state → per-position frequencies
print("\n[STEP 5] Fitting emission (state → frequencies)...", flush=True)
w_matrix = []
b_vector = []

for i, pos in enumerate(positions):
    freqs = freq_matrix[:, i]
    freqs_clipped = np.clip(freqs, 0.01, 0.99)
    logit_freqs = np.log(freqs_clipped / (1 - freqs_clipped))
    
    s_aug = np.hstack([state_matrix, np.ones((len(state_matrix), 1))])
    coef = np.linalg.lstsq(s_aug, logit_freqs, rcond=None)[0]
    
    w_matrix.append(coef[:-1])
    b_vector.append(coef[-1])

w_matrix = np.array(w_matrix)
b_vector = np.array(b_vector)
print(f"  Emission weights: {w_matrix.shape}", flush=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forecast_state(s_t, h, A, drift):
    """Forecast state h steps ahead"""
    s = s_t.copy()
    for _ in range(h):
        s = s @ A.T + drift
    return s

def forecast_frequencies(s, w_matrix, b_vector):
    """Get position frequencies for a state"""
    logits = s @ w_matrix.T + b_vector
    return sigmoid(logits)

# Evaluation
print("\n[STEP 6] Train/test split and evaluation...", flush=True)
train_split = int(0.7 * len(months))
train_months = months[:train_split]
test_months = months[train_split:]

print(f"  Train: {len(train_months)} months ({train_months[0]} to {train_months[-1]})", flush=True)
print(f"  Test: {len(test_months)} months ({test_months[0]} to {test_months[-1]})", flush=True)

results = []

for h in [1, 2, 3, 4, 5, 6]:
    print(f"\n  Horizon h={h}:", flush=True)
    
    brier_model = []
    brier_persist = []
    mae_model = []
    mae_persist = []
    n_forecasts = 0
    
    for i in range(len(test_months) - h):
        origin_idx = train_split + i
        forecast_idx = origin_idx + h
        
        origin_month = months[origin_idx]
        forecast_month = months[forecast_idx]
        
        # Model forecast
        s_origin = state_matrix[origin_idx]
        s_forecast = forecast_state(s_origin, h, A, drift)
        freqs_model = forecast_frequencies(s_forecast, w_matrix, b_vector)
        
        # Persistence baseline
        freqs_persist = freq_matrix[origin_idx]
        
        # Ground truth
        freqs_true = freq_matrix[forecast_idx]
        
        # Metrics
        brier_model.append(np.mean((freqs_model - freqs_true)**2))
        brier_persist.append(np.mean((freqs_persist - freqs_true)**2))
        mae_model.append(np.mean(np.abs(freqs_model - freqs_true)))
        mae_persist.append(np.mean(np.abs(freqs_persist - freqs_true)))
        n_forecasts += 1
    
    if brier_model:
        mean_brier_model = np.mean(brier_model)
        mean_brier_persist = np.mean(brier_persist)
        mean_mae_model = np.mean(mae_model)
        mean_mae_persist = np.mean(mae_persist)
        ratio_brier = mean_brier_model / mean_brier_persist if mean_brier_persist > 0 else 1.0
        
        print(f"    Forecasts: {n_forecasts}", flush=True)
        print(f"    Brier - Model: {mean_brier_model:.6f}, Persist: {mean_brier_persist:.6f}, Ratio: {ratio_brier:.3f}", flush=True)
        print(f"    MAE - Model: {mean_mae_model:.6f}, Persist: {mean_mae_persist:.6f}", flush=True)
        
        better = "MODEL" if ratio_brier < 1.0 else "PERSIST"
        print(f"    Winner: {better}", flush=True)
        
        results.append({
            'h': h,
            'n_forecasts': n_forecasts,
            'brier_model': mean_brier_model,
            'brier_persist': mean_brier_persist,
            'ratio_brier': ratio_brier,
            'mae_model': mean_mae_model,
            'mae_persist': mean_mae_persist,
            'winner': better
        })

# Save results
print("\n[STEP 7] Saving results...", flush=True)
results_df = pd.DataFrame(results)
results_csv = RESULTS_DIR / "continuous_state_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"  ✓ Saved to {results_csv}", flush=True)

# Summary
print("\n" + "="*70, flush=True)
print("SUMMARY", flush=True)
print("="*70, flush=True)

model_wins = (results_df['ratio_brier'] < 1.0).sum()
total = len(results_df)

print(f"\nModel beats persistence on {model_wins}/{total} horizons", flush=True)
print("\nDetailed results:", flush=True)
print(results_df.to_string(index=False), flush=True)

if model_wins > 0:
    print(f"\n✓ Model shows promise at some horizons", flush=True)
else:
    print(f"\n✗ Persistence is competitive across all horizons", flush=True)

print("\n" + "="*70, flush=True)
