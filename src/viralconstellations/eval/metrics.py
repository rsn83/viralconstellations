"""
Evaluation metrics for categorical mutation constellations.

All metrics operate at the population (distribution) level.
Input matrices are (n_seq, P) int8: 0=reference, 1-20=amino acid.
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict


def _to_binary(mat: np.ndarray) -> np.ndarray:
    """Convert categorical (P,) to binary presence/absence. 1 if any non-reference."""
    return (mat > 0).astype(np.float32)


def pos_frequency_correlation(generated: np.ndarray, real: np.ndarray) -> Dict:
    """
    Pearson r between per-position mutation frequencies.
    Did the model reproduce which positions are commonly mutated?
    (Binary presence/absence, ignores which residue.)
    """
    gen_freq  = _to_binary(generated).mean(axis=0)
    real_freq = _to_binary(real).mean(axis=0)
    r, p = pearsonr(gen_freq, real_freq)
    return {"pos_freq_r": float(r), "pos_freq_p": float(p)}


def pairwise_cooccurrence(generated: np.ndarray, real: np.ndarray,
                          top_k: int = 100) -> Dict:
    """
    Pearson r between pairwise joint mutation frequencies (binary presence).
    Restricted to top-k most frequently mutated positions.
    """
    real_freq  = _to_binary(real).mean(axis=0)
    top_sites  = np.argsort(real_freq)[::-1][:top_k]
    gen_bin    = _to_binary(generated)[:, top_sites]
    real_bin   = _to_binary(real)[:, top_sites]
    gen_coo    = (gen_bin.T  @ gen_bin)  / len(gen_bin)
    real_coo   = (real_bin.T @ real_bin) / len(real_bin)
    iu = np.triu_indices(top_k, k=1)
    r, _ = pearsonr(gen_coo[iu], real_coo[iu])
    return {"pairwise_coo_r": float(r)}


def independence_baseline_coo(real: np.ndarray, top_k: int = 100,
                               seed: int = 42) -> Dict:
    """
    Pairwise co-occurrence for the independence baseline.
    Baseline samples each position from its marginal frequency (binary).
    """
    rng = np.random.default_rng(seed)
    real_freq  = _to_binary(real).mean(axis=0)
    baseline   = rng.binomial(1, real_freq, size=(len(real), len(real_freq)))
    top_sites  = np.argsort(real_freq)[::-1][:top_k]
    bl_bin     = baseline[:, top_sites].astype(np.float32)
    real_bin   = _to_binary(real)[:, top_sites]
    bl_coo     = (bl_bin.T  @ bl_bin)   / len(bl_bin)
    real_coo   = (real_bin.T @ real_bin) / len(real_bin)
    iu = np.triu_indices(top_k, k=1)
    r, _ = pearsonr(bl_coo[iu], real_coo[iu])
    return {"baseline_pairwise_coo_r": float(r)}


def mmd_hamming(generated: np.ndarray, real: np.ndarray,
                sigma: float = 5.0, n_sub: int = 500, seed: int = 42) -> Dict:
    """MMD with RBF kernel over Hamming distances on binary presence vectors."""
    rng = np.random.default_rng(seed)
    G   = _to_binary(generated[rng.choice(len(generated), min(n_sub, len(generated)), replace=False)])
    R   = _to_binary(real[rng.choice(len(real),          min(n_sub, len(real)),       replace=False)])

    def rbf(X, Y):
        diff = (X[:, None, :] - Y[None, :, :]) ** 2
        return np.exp(-diff.sum(-1) / (2 * sigma ** 2))

    mmd = rbf(G, G).mean() + rbf(R, R).mean() - 2 * rbf(G, R).mean()
    return {"mmd": float(mmd)}


def frontier_coverage(generated: np.ndarray, real: np.ndarray,
                      hamming_r: int = 1, n_check: int = 300,
                      seed: int = 42) -> Dict:
    """
    Fraction of real constellations within Hamming distance hamming_r
    of at least one generated constellation (binary presence comparison).
    """
    rng      = np.random.default_rng(seed)
    n_check  = min(n_check, len(real))
    real_sub = _to_binary(real[rng.choice(len(real), n_check, replace=False)])
    gen_bin  = _to_binary(generated)
    covered  = sum(
        1 for rv in real_sub
        if (rv[None, :] != gen_bin).sum(axis=1).min() <= hamming_r
    )
    return {f"frontier_coverage_H{hamming_r}": covered / n_check}


def mean_mut_count(generated: np.ndarray, real: np.ndarray) -> Dict:
    """Mean number of non-reference positions per sequence."""
    return {
        "mean_mut_count":       float((generated > 0).sum(axis=1).mean()),
        "mean_mut_count_real":  float((real      > 0).sum(axis=1).mean()),
    }


def all_metrics_categorical(
    generated: np.ndarray,
    real:      np.ndarray,
    top_k:     int = 100,
    mmd_n_sub: int = 500,
    hamming_r: int = 1,
) -> Dict:
    out = {}
    out.update(pos_frequency_correlation(generated, real))
    out.update(pairwise_cooccurrence(generated, real, top_k))
    out.update(independence_baseline_coo(real, top_k))
    out.update(mmd_hamming(generated, real, n_sub=mmd_n_sub))
    out.update(frontier_coverage(generated, real, hamming_r))
    out.update(mean_mut_count(generated, real))
    return out
