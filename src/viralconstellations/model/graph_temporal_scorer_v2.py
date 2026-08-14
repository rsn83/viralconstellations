"""
src/viralconstellations/model/graph_temporal_scorer.py  (v2)

Complete graph-temporal architecture over (position,residue) nodes,
built against the real N=1180 vocabulary from
data/processed/full_data_graphs_posres/.

Components (each independently ablatable -- see ablation flags in
GraphTemporalScorer.__init__):
  - Relational graph conv (cooc, profile_sim, background_overlap, struct)
  - GRU over the input window (per-node temporal state)
  - Edge-history encoder (a specific pair's own g_t trajectory)
  - Decoder: node_i + node_j + edge_history -> score

Ablation flags let you turn off the GNN, the RNN, or the edge-history
pathway independently, so run_ablation.py can produce a clean
leave-one-component-out comparison against the same full model.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_context_profile(g_t: torch.Tensor, f_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return g_t / (f_t.unsqueeze(1) + eps)


def profile_similarity_matrix(profiles: torch.Tensor) -> torch.Tensor:
    normed = F.normalize(profiles, dim=-1, eps=1e-8)
    return normed @ normed.T


def compute_distinct_constellation_stats(occupied: dict, N: int):
    import numpy as np
    G = np.zeros((N, N), dtype=np.float64)
    F_ = np.zeros(N, dtype=np.float64)
    for constellation in occupied.keys():
        nodes = [n for n in constellation if n < N]
        for n in nodes:
            F_[n] += 1
        for a in range(len(nodes)):
            for b in range(a, len(nodes)):
                i, j = nodes[a], nodes[b]
                G[i, j] += 1
                if i != j:
                    G[j, i] += 1
    return G, F_


def background_overlap_matrix(G_distinct: torch.Tensor, F_distinct: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    profiles = G_distinct / (F_distinct.unsqueeze(1) + eps)
    return profile_similarity_matrix(profiles)


class RelationalGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, relation_names: list[str]):
        super().__init__()
        self.relation_names = relation_names
        self.weights = nn.ModuleDict({
            rel: nn.Linear(in_dim, out_dim, bias=False) for rel in relation_names
        })
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, node_feats: torch.Tensor, adj_by_relation: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.self_loop(node_feats)
        for rel in self.relation_names:
            if rel not in adj_by_relation:
                continue
            A = adj_by_relation[rel]
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
            A_norm = A / deg
            message = A_norm @ node_feats
            out = out + self.weights[rel](message)
        return F.relu(out)


class NodeTemporalEncoder(nn.Module):
    """
    use_gnn=False: skip graph conv entirely, feed raw node features
                   straight into the GRU (ablation: does graph structure matter).
    use_rnn=False: skip the GRU, use only the LAST month's conv output
                   (ablation: does temporal memory matter).
    """
    def __init__(self, node_feat_dim: int, hidden_dim: int, relation_names: list[str],
                 n_conv_layers: int = 2, use_gnn: bool = True, use_rnn: bool = True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim

        if use_gnn:
            self.convs = nn.ModuleList([
                RelationalGraphConv(
                    node_feat_dim if l == 0 else hidden_dim, hidden_dim, relation_names
                ) for l in range(n_conv_layers)
            ])
        else:
            self.proj = nn.Linear(node_feat_dim, hidden_dim)

        if use_rnn:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_feats_seq: list[torch.Tensor],
                adj_seq: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        P = node_feats_seq[0].shape[0]
        h = torch.zeros(P, self.hidden_dim, device=node_feats_seq[0].device)

        months = zip(node_feats_seq, adj_seq) if self.use_rnn else [(node_feats_seq[-1], adj_seq[-1])]
        for x_t, adj_t in months:
            if self.use_gnn:
                z = x_t
                for conv in self.convs:
                    z = conv(z, adj_t)
            else:
                z = F.relu(self.proj(x_t))

            h = self.gru(z, h) if self.use_rnn else z
        return h


class EdgeHistoryEncoder(nn.Module):
    def __init__(self, window: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_history: torch.Tensor) -> torch.Tensor:
        return self.net(edge_history)


class GraphTemporalScorer(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, relation_names: list[str],
                 edge_history_window: int, n_conv_layers: int = 2,
                 use_gnn: bool = True, use_rnn: bool = True, use_edge_history: bool = True):
        super().__init__()
        self.use_edge_history = use_edge_history
        self.node_encoder = NodeTemporalEncoder(
            node_feat_dim, hidden_dim, relation_names, n_conv_layers, use_gnn, use_rnn
        )
        if use_edge_history:
            self.edge_history_encoder = EdgeHistoryEncoder(edge_history_window, hidden_dim)
            decoder_in = hidden_dim * 3
        else:
            decoder_in = hidden_dim * 2

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats_seq, adj_seq, pair_i, pair_j, edge_history=None) -> torch.Tensor:
        node_h = self.node_encoder(node_feats_seq, adj_seq)
        h_i, h_j = node_h[pair_i], node_h[pair_j]

        if self.use_edge_history:
            edge_h = self.edge_history_encoder(edge_history)
            combined = torch.cat([h_i, h_j, edge_h], dim=-1)
        else:
            combined = torch.cat([h_i, h_j], dim=-1)

        return self.decoder(combined).squeeze(-1)


class NaivePersistenceBaseline:
    """Predicts g_{t+h}[i,j] = g_t[i,j] (unchanged). No learning -- this is
    the bar the real model must beat, given Check C found raw persistence
    r=0.85-0.98."""
    def score(self, g_t: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> torch.Tensor:
        return g_t[pair_i, pair_j]
