"""
Evaluation metrics for categorical mutation constellations.
All metrics operate at the population (distribution) level.
NaN-safe: returns nan rather than crashing on degenerate inputs.
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict


def _safe_pearsonr(x, y):
    """pearsonr with NaN fallback for zero-variance inputs."""
    try:
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return float('nan')
        r, _ = pearsonr(x, y)
        return float(r)
    except Exception:
        return float('nan')


def _to_binary(mat: np.ndarray) -> np.ndarray:
    return (mat > 0).astype(np.float32)


def pos_frequency_correlation(generated: np.ndarray, real: np.ndarray) -> Dict:
    gen_freq  = _to_binary(generated).mean(axis=0)
    real_freq = _to_binary(real).mean(axis=0)
    return {"pos_freq_r": _safe_pearsonr(gen_freq, real_freq)}


def pairwise_cooccurrence(generated: np.ndarray, real: np.ndarray,
                          top_k: int = 100) -> Dict:
    real_freq = _to_binary(real).mean(axis=0)
    top_sites = np.argsort(real_freq)[::-1][:top_k]
    gen_bin   = _to_binary(generated)[:, top_sites]
    real_bin  = _to_binary(real)[:, top_sites]
    gen_coo   = (gen_bin.T  @ gen_bin)  / max(len(gen_bin), 1)
    real_coo  = (real_bin.T @ real_bin) / max(len(real_bin), 1)
    iu        = np.triu_indices(min(top_k, len(top_sites)), k=1)
    return {"pairwise_coo_r": _safe_pearsonr(gen_coo[iu], real_coo[iu])}


def independence_baseline_coo(real: np.ndarray, top_k: int = 100,
                               seed: int = 42) -> Dict:
    rng       = np.random.default_rng(seed)
    real_freq = _to_binary(real).mean(axis=0)
    baseline  = rng.binomial(1, real_freq, size=(len(real), len(real_freq)))
    top_sites = np.argsort(real_freq)[::-1][:top_k]
    bl_bin    = baseline[:, top_sites].astype(np.float32)
    real_bin  = _to_binary(real)[:, top_sites]
    bl_coo    = (bl_bin.T  @ bl_bin)   / max(len(bl_bin), 1)
    real_coo  = (real_bin.T @ real_bin) / max(len(real_bin), 1)
    iu        = np.triu_indices(min(top_k, len(top_sites)), k=1)
    return {"baseline_pairwise_coo_r": _safe_pearsonr(bl_coo[iu], real_coo[iu])}


def mmd_hamming(generated: np.ndarray, real: np.ndarray,
                sigma: float = 5.0, n_sub: int = 500, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    G   = _to_binary(generated[rng.choice(len(generated), min(n_sub,len(generated)), replace=False)])
    R   = _to_binary(real[rng.choice(len(real),           min(n_sub,len(real)),       replace=False)])
    def rbf(X, Y):
        diff = (X[:,None,:] - Y[None,:,:]) ** 2
        return np.exp(-diff.sum(-1) / (2*sigma**2))
    mmd = float(rbf(G,G).mean() + rbf(R,R).mean() - 2*rbf(G,R).mean())
    return {"mmd": mmd}


def frontier_coverage(generated: np.ndarray, real: np.ndarray,
                      hamming_r: int = 1, n_check: int = 300,
                      seed: int = 42) -> Dict:
    rng      = np.random.default_rng(seed)
    n_check  = min(n_check, len(real))
    real_sub = _to_binary(real[rng.choice(len(real), n_check, replace=False)])
    gen_bin  = _to_binary(generated)
    covered  = sum(
        1 for rv in real_sub
        if (rv[None,:] != gen_bin).sum(axis=1).min() <= hamming_r
    )
    return {f"frontier_coverage_H{hamming_r}": covered / max(n_check, 1)}


def mean_mut_count(generated: np.ndarray, real: np.ndarray) -> Dict:
    return {
        "mean_mut_count":      float((generated > 0).sum(axis=1).mean()),
        "mean_mut_count_real": float((real      > 0).sum(axis=1).mean()),
    }


def all_metrics_categorical(
    generated: np.ndarray,
    real:      np.ndarray,
    top_k:     int = 100,
    mmd_n_sub: int = 500,
    hamming_r: int = 1,
) -> Dict:
    # Guard against degenerate generated sequences (all zeros)
    if (generated > 0).sum() == 0:
        return {
            "pos_freq_r": float('nan'),
            "pairwise_coo_r": float('nan'),
            "baseline_pairwise_coo_r": float('nan'),
            "mmd": float('nan'),
            f"frontier_coverage_H{hamming_r}": float('nan'),
            "mean_mut_count": 0.0,
            "mean_mut_count_real": float((real > 0).sum(axis=1).mean()),
            "warning": "generated sequences are all reference — length head may be untrained",
        }
    out = {}
    out.update(pos_frequency_correlation(generated, real))
    out.update(pairwise_cooccurrence(generated, real, top_k))
    out.update(independence_baseline_coo(real, top_k))
    out.update(mmd_hamming(generated, real, n_sub=mmd_n_sub))
    out.update(frontier_coverage(generated, real, hamming_r))
    out.update(mean_mut_count(generated, real))
    return out
