"""
src/viralconstellations/model/esm_embeddings.py

Loads the cache built by scripts/17_extract_esm_embeddings.py (ESM run
ONCE, offline, on GPU) and, for a given month's occupied dict, produces
a (N, esm_dim) tensor: for each node, the weighted mean of the cached
embeddings of that node across THIS MONTH's real carrier constellations
(weighted by how many sequences carried each one).

This is a pure lookup + weighted average -- no model inference here, so
it's fast enough to run per month, every month, on CPU, and can be
cached in your existing month_cache dict without the memory blowup the
earlier from-scratch conv approach had (this is small: N x esm_dim
floats, e.g. 1180 x 640 x 4 bytes ~= 3MB per month).
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import torch


class ESMEmbeddingCache:
    def __init__(self, cache_path: Path):
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
        self.embeddings: dict[frozenset, dict[int, np.ndarray]] = data["embeddings"]
        self.esm_dim: int = data["esm_dim"]
        self._miss_count = 0
        self._hit_count = 0

    def build_month_node_embeddings(self, occ: dict, N: int) -> torch.Tensor:
        """
        occ: {constellation: count} for one month (same dict already
             loaded elsewhere in your pipeline via load_month/get_month).
        Returns (N, esm_dim) float32 tensor.
        """
        sums = np.zeros((N, self.esm_dim), dtype=np.float64)
        weight_totals = np.zeros(N, dtype=np.float64)

        for constellation, count in occ.items():
            key = frozenset(constellation)
            node_embs = self.embeddings.get(key)
            if node_embs is None:
                self._miss_count += 1
                continue  # constellation wasn't in the cache (e.g. below
                          # --min_count when the cache was built) -- skip,
                          # its nodes fall back to whatever other carriers they have
            self._hit_count += 1
            w = float(count) if isinstance(count, (int, float)) else 1.0
            for node_idx, emb in node_embs.items():
                if node_idx < N:
                    sums[node_idx] += w * emb
                    weight_totals[node_idx] += w

        out = np.zeros((N, self.esm_dim), dtype=np.float32)
        nonzero = weight_totals > 0
        out[nonzero] = (sums[nonzero] / weight_totals[nonzero, None]).astype(np.float32)
        return torch.tensor(out, dtype=torch.float32)

    def build_month_node_raw_embeddings(self, occ: dict, N: int, K: int = 8):
        """
        Alternative to build_month_node_embeddings: instead of collapsing
        each node's per-constellation ESM embeddings into one fixed
        weighted-mean vector, returns the raw (up to K) individual
        embeddings themselves, so a LEARNABLE attention pool (see
        AttentionPoolESMAdapter in graph_temporal_scorer_v2.py) can
        decide which contexts matter, instead of a fixed count-weighted
        average baked in before the model ever sees the data.

        Selection: top-K carrier constellations by real sequence count
        (deterministic, not random -- reproducible across runs). If a
        node has fewer than K real carriers, remaining slots are masked
        out (zero-padded, not counted).

        Returns:
          raw:  (N, K, esm_dim) float32 tensor
          mask: (N, K) bool tensor, True = real (non-padding) slot
        """
        per_node: list[list[tuple[float, np.ndarray]]] = [[] for _ in range(N)]

        for constellation, count in occ.items():
            key = frozenset(constellation)
            node_embs = self.embeddings.get(key)
            if node_embs is None:
                self._miss_count += 1
                continue
            self._hit_count += 1
            w = float(count) if isinstance(count, (int, float)) else 1.0
            for node_idx, emb in node_embs.items():
                if node_idx < N:
                    per_node[node_idx].append((w, emb))

        raw = np.zeros((N, K, self.esm_dim), dtype=np.float32)
        mask = np.zeros((N, K), dtype=bool)
        for i in range(N):
            items = per_node[i]
            if not items:
                continue
            items.sort(key=lambda item: -item[0])  # top-K by count, deterministic
            for k, (w, emb) in enumerate(items[:K]):
                raw[i, k] = emb
                mask[i, k] = True

        return torch.tensor(raw, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

    def report_coverage(self):
        total = self._hit_count + self._miss_count
        pct = 100.0 * self._hit_count / total if total else float("nan")
        print(f"[ESMEmbeddingCache] constellation cache hit rate: "
              f"{self._hit_count}/{total} ({pct:.1f}%)")
