"""
Graph-temporal scorer: predicts P(edge (i,j) present at t+h) from a
sequence of monthly co-occurrence graphs.

Architecture, in three pieces, late-fused at the end:

1. NODE PATHWAY: multi-relational graph convolution (one weight matrix
   per edge type, e.g. 'cooc', 'struct') at each month, producing per-node
   features, then a GRU over months producing a temporal node embedding.

2. EDGE-HISTORY PATHWAY: for a specific pair (i,j), its own recent
   trajectory of g_t[i,j] values (independent of node embeddings),
   encoded by a small linear/GRU layer.

3. DECODER: combine node embeddings for i and j with the edge-history
   embedding for (i,j) -> MLP -> P(edge present at t+h).

This is a skeleton, verified to run end-to-end on synthetic data (forward
pass, shapes, gradients). Wiring in real g_t/f_t sequences from
data/processed/full_data_graphs/ is the next step once you're ready to
train it -- see the __main__ block for exactly what real inputs to build.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Multi-relational graph convolution (one node-feature update per month)
# ---------------------------------------------------------------------------
class RelationalGraphConv(nn.Module):
    """
    One graph-conv layer supporting multiple edge types. Each relation
    type gets its own weight matrix; messages are summed across relations
    before the nonlinearity (standard R-GCN-style combination).
    """
    def __init__(self, in_dim: int, out_dim: int, relation_names: list[str]):
        super().__init__()
        self.relation_names = relation_names
        self.weights = nn.ModuleDict({
            rel: nn.Linear(in_dim, out_dim, bias=False) for rel in relation_names
        })
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, node_feats: torch.Tensor, adj_by_relation: dict[str, torch.Tensor]
                ) -> torch.Tensor:
        """
        node_feats: (P, in_dim)
        adj_by_relation: {relation_name: (P, P) adjacency/weight matrix}
                          Missing relations (e.g. 'struct' not yet built)
                          can be omitted or passed as an all-zero matrix.
        Returns: (P, out_dim)
        """
        out = self.self_loop(node_feats)
        for rel in self.relation_names:
            if rel not in adj_by_relation:
                continue
            A = adj_by_relation[rel]
            # normalize by degree to keep magnitudes stable across months
            # with very different total edge weight (e.g. 2022-01 vs 2025-11)
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
            A_norm = A / deg
            message = A_norm @ node_feats            # (P, in_dim)
            out = out + self.weights[rel](message)    # (P, out_dim)
        return F.relu(out)


# ---------------------------------------------------------------------------
# 2. Node temporal pathway: graph conv per month, then GRU over months
# ---------------------------------------------------------------------------
class NodeTemporalEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, relation_names: list[str],
                 n_conv_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList([
            RelationalGraphConv(
                node_feat_dim if l == 0 else hidden_dim, hidden_dim, relation_names
            ) for l in range(n_conv_layers)
        ])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, node_feats_seq: list[torch.Tensor],
                adj_seq: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        node_feats_seq: list of (P, node_feat_dim), one per month
        adj_seq: list of {relation: (P,P)}, one dict per month, same length
        Returns: (P, hidden_dim) -- final node embedding after the whole window
        """
        P = node_feats_seq[0].shape[0]
        h = torch.zeros(P, self.hidden_dim, device=node_feats_seq[0].device)

        for x_t, adj_t in zip(node_feats_seq, adj_seq):
            z = x_t
            for conv in self.convs:
                z = conv(z, adj_t)               # (P, hidden_dim)
            h = self.gru(z, h)                    # (P, hidden_dim), per-node GRU cell
        return h


# ---------------------------------------------------------------------------
# 3. Edge-history pathway: a specific pair's own recent trajectory
# ---------------------------------------------------------------------------
class EdgeHistoryEncoder(nn.Module):
    """
    Encodes a window of past g_t[i,j] scalar values for a SPECIFIC pair
    into a small embedding -- independent of the node pathway. This is
    what lets the model use "this edge itself has been rising/falling"
    on top of "these two nodes' broader neighborhoods have been evolving".
    """
    def __init__(self, window: int, hidden_dim: int):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Linear(window, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_history: torch.Tensor) -> torch.Tensor:
        """
        edge_history: (n_pairs, window) -- past g_t[i,j] values (already
        log1p-transformed upstream is recommended, given Check A's finding
        that raw magnitude underfits).
        Returns: (n_pairs, hidden_dim)
        """
        return self.net(edge_history)


# ---------------------------------------------------------------------------
# Full model: late fusion of node pathway + edge-history pathway
# ---------------------------------------------------------------------------
class GraphTemporalScorer(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, relation_names: list[str],
                 edge_history_window: int, n_conv_layers: int = 2):
        super().__init__()
        self.node_encoder = NodeTemporalEncoder(
            node_feat_dim, hidden_dim, relation_names, n_conv_layers
        )
        self.edge_history_encoder = EdgeHistoryEncoder(edge_history_window, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),   # node_i, node_j, edge_history
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats_seq: list[torch.Tensor],
                adj_seq: list[dict[str, torch.Tensor]],
                pair_i: torch.Tensor, pair_j: torch.Tensor,
                edge_history: torch.Tensor) -> torch.Tensor:
        """
        pair_i, pair_j: (n_pairs,) long tensors -- indices of candidate pairs to score
        edge_history: (n_pairs, window) -- as in EdgeHistoryEncoder
        Returns: (n_pairs,) logits for P(edge present at t+h)
        """
        node_h = self.node_encoder(node_feats_seq, adj_seq)      # (P, hidden_dim)
        edge_h = self.edge_history_encoder(edge_history)          # (n_pairs, hidden_dim)

        h_i = node_h[pair_i]                                       # (n_pairs, hidden_dim)
        h_j = node_h[pair_j]                                       # (n_pairs, hidden_dim)

        combined = torch.cat([h_i, h_j, edge_h], dim=-1)           # (n_pairs, hidden_dim*3)
        logits = self.decoder(combined).squeeze(-1)                # (n_pairs,)
        return logits


# ---------------------------------------------------------------------------
# Synthetic end-to-end test: verifies shapes, forward pass, and that
# gradients flow through every component (node conv, GRU, edge history,
# decoder) before you wire in real data.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    P = 20                  # number of positions (nodes)
    T = 6                   # months in the input window
    node_feat_dim = 4       # e.g. [frequency, freq_trend, eve_score, degree]
    hidden_dim = 16
    edge_window = 6
    n_pairs = 50

    relation_names = ["cooc", "struct"]  # struct can be all-zero until Check B is done

    model = GraphTemporalScorer(node_feat_dim, hidden_dim, relation_names, edge_window)

    # synthetic monthly node features and adjacency matrices
    node_feats_seq = [torch.randn(P, node_feat_dim) for _ in range(T)]
    adj_seq = []
    for _ in range(T):
        cooc = torch.rand(P, P)
        cooc = (cooc + cooc.T) / 2  # symmetric, like a real co-occurrence graph
        struct = torch.zeros(P, P)  # placeholder until real PDB distances are wired in
        adj_seq.append({"cooc": cooc, "struct": struct})

    pair_i = torch.randint(0, P, (n_pairs,))
    pair_j = torch.randint(0, P, (n_pairs,))
    edge_history = torch.randn(n_pairs, edge_window)

    logits = model(node_feats_seq, adj_seq, pair_i, pair_j, edge_history)
    print(f"Output logits shape: {logits.shape}  (expected: ({n_pairs},))")
    assert logits.shape == (n_pairs,)

    # verify gradients flow through every component
    labels = torch.randint(0, 2, (n_pairs,)).float()
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    grad_report = {
        "node conv (cooc relation, layer 0)":
            model.node_encoder.convs[0].weights["cooc"].weight.grad is not None,
        "node conv (struct relation, layer 0)":
            model.node_encoder.convs[0].weights["struct"].weight.grad is not None,
        "GRU cell": model.node_encoder.gru.weight_hh.grad is not None,
        "edge history encoder": model.edge_history_encoder.net[0].weight.grad is not None,
        "decoder": model.decoder[0].weight.grad is not None,
    }
    print("\nGradient flow check (all should be True):")
    for name, ok in grad_report.items():
        print(f"  {name}: {ok}")
    assert all(grad_report.values()), "Gradient did not reach every component!"

    print("\nForward pass, backward pass, and gradient flow all verified OK.")
    print(f"Loss: {loss.item():.4f}")
