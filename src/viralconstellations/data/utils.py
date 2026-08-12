"""Shared data utilities used across scripts."""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def load_vocab(vocab_dir: Path) -> List[str]:
    """Load the global mutation vocabulary."""
    return (vocab_dir / "mutation_vocab.txt").read_text().splitlines()


def load_month_matrix(matrix_dir: Path, month: str) -> np.ndarray:
    """Load the binary mutation matrix for one month. Shape: (n_seq, V) uint8."""
    return np.load(matrix_dir / f"{month}.npy")


def load_month_freq(matrix_dir: Path, month: str) -> np.ndarray:
    """
    Load the per-site frequency vector for one month. Shape: (V,) float32.
    This is the conditioning signal: freq[j] = fraction of sequences
    in this month that carry mutation j.
    """
    return np.load(matrix_dir / f"{month}_freq.npy")


def load_all_months(matrix_dir: Path) -> Dict[str, np.ndarray]:
    """Load binary matrices for all available months."""
    index = pd.read_csv(matrix_dir / "index.tsv", sep="\t")
    return {
        row["month"]: np.load(matrix_dir / f"{row['month']}.npy")
        for _, row in index.iterrows()
    }


def train_test_split_months(
    available_months: List[str],
    train_months: List[str] = None,
    test_month: str = None,
) -> Tuple[List[str], str]:
    """
    Determine train/test split from config or defaults.

    Default: use all months except the last as training,
    last month as test (temporal holdout — no leakage).
    """
    months = sorted(available_months)
    if not months:
        raise ValueError("No months available.")
    if test_month and test_month in months:
        t = test_month
        tr = train_months if train_months else [m for m in months if m < t]
    else:
        t  = months[-1]
        tr = train_months if train_months else months[:-1]
    return tr, t
