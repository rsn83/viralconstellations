"""
src/viralconstellations/model/hypergraph_scorer.py

HyperSAGNN hyperedge scoring (Zhang, Zou, Ma, "Hyper-SAGNN: a
self-attention based graph neural network for hypergraphs",
arXiv:1911.02613 / ICLR 2020), replacing the pairwise decoder in
graph_temporal_scorer_v2.py.

FIDELITY NOTE: the scorer below (HyperSAGNNScorer) was checked directly
against the paper's equations 3-7 -- static embedding, self-excluded
attention for the dynamic embedding, and squared-difference
compatibility all match the paper's formulation (see that class's own
docstring for the one deliberate, flagged deviation: sigmoid dropped
since this project regresses a count, not a binary link probability).
An earlier version of this file used a plain dot-product decoder
inspired by the general idea but not checked against the source --
that version has been replaced.

WHAT WAS A SIMPLIFICATION, NOW FIXED: the node encoder previously
reused NodeTemporalEncoder from graph_temporal_scorer_v2.py, which
builds node embeddings from PAIRWISE relations (cooc, profile_sim,
background_overlap, struct) as N x N adjacency matrices. That's
replaced below by HypergraphConv -- true incidence-matrix-based
hypergraph convolution (Feng et al., "Hypergraph Neural Networks",
AAAI 2019), which aggregates each node's embedding through the ACTUAL
hyperedges (constellations) it participates in, not a pairwise
projection of them. This matters concretely: pairwise cooc collapses
"A, B, and C all co-occurred together" into three separate pairwise
counts, discarding the ternary correlation; incidence-based
convolution preserves it, since propagation happens through the real
hyperedge structure directly.

Propagation rule (HGNN, Feng et al. 2019):
    X' = relu( D_v^-1/2 H W D_e^-1 H^T D_v^-1/2 X Theta )
  H: incidence matrix (N nodes x M hyperedges), built fresh each month
     directly from occ.keys() -- no decomposition.
  W: diagonal hyperedge weight = real sequence count for that
     constellation that month (uses information already available,
     not just unweighted structure).
  D_v, D_e: vertex and hyperedge degree matrices.
  Theta: learnable weight matrix.

Why the scoring change is still a genuine improvement even with that
caveat: a hyperedge (one full mutation constellation) was previously
decomposed into C(k,2) pairwise edges before scoring -- a lossy,
many-to-one mapping (different constellations can produce identical
pairwise projections). Scoring the full candidate set directly, via
this scorer, is lossless at that step: each candidate is scored as
itself, not as a bag of decomposed pairs.

Continuous-time point process machinery from HyperNodeTPP (Gracious et
al., AAAI 2025) is deliberately NOT used here, consistent with your own
project notes: monthly data isn't precisely timestamped, so the
discrete monthly window (GRU) already in place is the right level of
temporal resolution -- no need for a continuous-time component.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from viralconstellations.model.graph_temporal_scorer_v2 import ESMNodeAdapter


class HypergraphConv(nn.Module):
    """
    One layer of HGNN propagation (Feng et al., AAAI 2019), operating
    on a sparse incidence matrix built fresh each month directly from
    real constellations -- see build_incidence() below.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, X: torch.Tensor, H_sparse: torch.Tensor, W_diag: torch.Tensor,
                D_e_inv: torch.Tensor, D_v_inv_sqrt: torch.Tensor) -> torch.Tensor:
        """
        X: (N, in_dim) node features
        H_sparse: (N, M) sparse incidence matrix
        W_diag, D_e_inv: (M,) hyperedge weight and inverse degree
        D_v_inv_sqrt: (N,) inverse sqrt vertex degree
        """
        Xv = D_v_inv_sqrt.unsqueeze(-1) * X                          # (N, in_dim)
        Xe = torch.sparse.mm(H_sparse.t(), Xv)                        # (M, in_dim) node -> hyperedge
        Xe = Xe * (W_diag * D_e_inv).unsqueeze(-1)                     # weight + normalize by hyperedge size
        Xv2 = torch.sparse.mm(H_sparse, Xe)                            # (N, in_dim) hyperedge -> node
        Xv2 = D_v_inv_sqrt.unsqueeze(-1) * Xv2
        return F.relu(self.theta(Xv2))


def build_incidence(occ: dict, N: int, device):
    """
    Builds incidence data for one month, directly from occ
    (constellation -> count), no decomposition. Returns None if the
    month has zero constellations (degenerate, guarded against below).

    Returns TWO vertex-degree normalizations sharing the same H_sparse
    and D_e_inv, supporting two distinct hyperedge relations:
      - weighted (D_v_inv_sqrt_w):   uses real sequence counts as W --
        "how much this combination actually happened"
      - unweighted (D_v_inv_sqrt_u): uses W=1 for every real
        constellation regardless of popularity -- "pure topological
        co-membership", independent of how common each one was. This
        is the hyperedge analog of what background_overlap/profile_sim
        captured in the old pairwise model (structure from DISTINCT
        constellations, not weighted by frequency).
    """
    edges = []
    for c, v in occ.items():
        nodes = [n for n in c if n < N]
        if len(nodes) < 1:
            continue
        count = v if isinstance(v, (int, float)) else 1
        edges.append((nodes, max(float(count), 1.0)))
    if not edges:
        return None

    rows, cols = [], []
    for e_idx, (nodes, _) in enumerate(edges):
        for v in nodes:
            rows.append(v)
            cols.append(e_idx)
    M = len(edges)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    H_sparse = torch.sparse_coo_tensor(indices, values, size=(N, M)).coalesce().to(device)

    W_diag = torch.tensor([c for _, c in edges], dtype=torch.float32, device=device)
    ones_diag = torch.ones(M, dtype=torch.float32, device=device)
    D_e = torch.tensor([len(nodes) for nodes, _ in edges], dtype=torch.float32, device=device).clamp(min=1.0)
    D_e_inv = 1.0 / D_e

    D_v_w = torch.sparse.mm(H_sparse, W_diag.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
    D_v_inv_sqrt_w = D_v_w.pow(-0.5)
    D_v_u = torch.sparse.mm(H_sparse, ones_diag.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
    D_v_inv_sqrt_u = D_v_u.pow(-0.5)

    return H_sparse, W_diag, ones_diag, D_e_inv, D_v_inv_sqrt_w, D_v_inv_sqrt_u


class MultiRelationalHypergraphConv(nn.Module):
    """
    Two hyperedge relations, each with its own learnable Theta, summed
    -- the hyperedge analog of how RelationalGraphConv summed multiple
    pairwise relations in the old model. See build_incidence() for what
    each relation means.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_weighted = HypergraphConv(in_dim, out_dim)
        self.conv_unweighted = HypergraphConv(in_dim, out_dim)

    def forward(self, X: torch.Tensor, H_sparse, W_diag, ones_diag, D_e_inv,
                D_v_inv_sqrt_w, D_v_inv_sqrt_u) -> torch.Tensor:
        out_w = self.conv_weighted(X, H_sparse, W_diag, D_e_inv, D_v_inv_sqrt_w)
        out_u = self.conv_unweighted(X, H_sparse, ones_diag, D_e_inv, D_v_inv_sqrt_u)
        return out_w + out_u


class PairwiseStructConv(nn.Module):
    """
    The ONE genuinely pairwise relation: real structural (3D) proximity
    between residues, from an actual PDB structure (see
    scripts/20_build_structural_prior.py) -- NOT the all-zero
    placeholder that existed in the original pairwise model
    (graph_temporal_scorer_v2.py's `struct` was literally
    torch.zeros(N, N) the entire time, never populated).

    Kept as a separate, dedicated pairwise pathway rather than folded
    into the hyperedge convolution, because CA-CA distance is a
    property of a PAIR of residues -- it has no natural "hyperedge
    weight" interpretation the way real co-occurrence does. Combined
    additively with the hyperedge conv output before the GRU, same
    principle as summing multiple relations elsewhere in this project.

    struct_adj is STATIC: identical every month (it's the folded
    protein's real geometry, not data-dependent), computed once by
    script 20 and passed in unchanged for every window.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
        self.weight = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, struct_adj: torch.Tensor) -> torch.Tensor:
        deg = struct_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        message = (struct_adj / deg) @ X
        return F.relu(self.self_loop(X) + self.weight(message))


class HypergraphNodeTemporalEncoder(nn.Module):
    """
    True-hypergraph counterpart to NodeTemporalEncoder in
    graph_temporal_scorer_v2.py. Three relations now, matching what
    was ACTUALLY real in the old pairwise model (it claimed 4, but
    `struct` was always an unused all-zero placeholder -- see
    hypergraph_scorer.py module docstring):
      1. weighted hyperedge incidence  (real, was `cooc`'s honest job)
      2. unweighted hyperedge incidence (real, was `background_overlap`/
         `profile_sim`'s honest job)
      3. real structural proximity (NOW real, from an actual PDB --
         `struct` was never populated before)
    """
    def __init__(self, node_feat_dim: int, hidden_dim: int, n_conv_layers: int = 2,
                 use_gnn: bool = True, use_rnn: bool = True, use_esm_context: bool = True,
                 use_struct: bool = True, esm_dim: int = 640, esm_adapter_dim: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_rnn = use_rnn
        self.use_esm_context = use_esm_context
        self.use_struct = use_struct
        self.hidden_dim = hidden_dim

        if use_esm_context:
            self.esm_adapter = ESMNodeAdapter(esm_dim, esm_adapter_dim, dropout=dropout)
            total_in_dim = node_feat_dim + esm_adapter_dim
        else:
            total_in_dim = node_feat_dim

        # Always available: primary path when use_gnn=False, and a
        # fallback for the rare degenerate month with zero constellations
        # (build_incidence returns None) even when use_gnn=True.
        self.proj = nn.Linear(total_in_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)

        if use_gnn:
            self.convs = nn.ModuleList([
                MultiRelationalHypergraphConv(total_in_dim if l == 0 else hidden_dim, hidden_dim)
                for l in range(n_conv_layers)
            ])
            self.conv_dropout = nn.Dropout(dropout)

        if use_struct:
            self.struct_conv = PairwiseStructConv(total_in_dim, hidden_dim)
            self.struct_dropout = nn.Dropout(dropout)

        if use_rnn:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
            self.gru_dropout = nn.Dropout(dropout)

    def forward(self, node_feats_seq: list[torch.Tensor], incidence_seq: list,
                struct_adj: torch.Tensor | None = None,
                esm_seq: list[torch.Tensor] | None = None) -> torch.Tensor:
        """
        incidence_seq: list of (H_sparse, W_diag, ones_diag, D_e_inv,
                       D_v_inv_sqrt_w, D_v_inv_sqrt_u) tuples, or None
                       for a degenerate month -- one entry per month.
        struct_adj: (N, N) static structural prior, SAME every month
                    (only needed if use_struct=True).
        """
        P = node_feats_seq[0].shape[0]
        h = torch.zeros(P, self.hidden_dim, device=node_feats_seq[0].device)

        if self.use_esm_context:
            assert esm_seq is not None
            months_data = list(zip(node_feats_seq, incidence_seq, esm_seq))
        else:
            months_data = list(zip(node_feats_seq, incidence_seq, [None] * len(node_feats_seq)))

        months = months_data if self.use_rnn else [months_data[-1]]
        for x_t, inc_t, esm_t in months:
            if self.use_esm_context:
                esm_emb = self.esm_adapter(esm_t)
                x_t = torch.cat([x_t, esm_emb], dim=-1)

            if self.use_gnn and inc_t is not None:
                z = x_t
                for conv in self.convs:
                    z = conv(z, *inc_t)
                z = self.conv_dropout(z)
            else:
                z = self.proj_dropout(F.relu(self.proj(x_t)))  # no-gnn path, or degenerate-month fallback

            if self.use_struct and struct_adj is not None:
                z = z + self.struct_dropout(self.struct_conv(x_t, struct_adj))

            h = self.gru_dropout(self.gru(z, h)) if self.use_rnn else z
        return h


class HyperSAGNNScorer(nn.Module):
    """
    Faithful implementation of Hyper-SAGNN (Zhang, Zou, Ma, "Hyper-SAGNN:
    a self-attention based graph neural network for hypergraphs",
    arXiv:1911.02613 / ICLR 2020), verified directly against the paper's
    equations 3-7, not reconstructed from memory.

    Per the paper:
      - static embedding:  s_i = tanh(W_s^T x_i)                    (eq, Sec 3.2)
      - dynamic embedding: d_i = tanh(sum_{j != i} alpha_ij W_V^T x_j)
        where attention weights alpha_ij are computed via standard
        scaled dot-product attention, but the j=i term is EXPLICITLY
        EXCLUDED (eq 3-5). The paper tested including it ("Variant
        Type I", Appendix A.1) and found excluding it performs as
        well or better -- this matters, so it's implemented exactly
        as specified, not approximated with a library that attends to
        self by default.
      - compatibility:     o_i = W_o^T((d_i - s_i)^2) + b            (eq 6)
        a "pseudo-euclidean distance" between static and dynamic
        embeddings -- NOT a dot product (that was an earlier,
        unverified simplification, corrected here).
      - hyperedge score:   p = mean_i(sigmoid(o_i))                  (eq 7)

    ONE DELIBERATE DEVIATION, clearly flagged: the paper's task is
    binary hyperedge existence (hence sigmoid -> probability). This
    project predicts log1p(count) via regression, so the final sigmoid
    is dropped -- o_i is averaged directly as a raw regression score.
    Everything upstream of that (static/dynamic embeddings, the
    self-excluded attention, the squared-difference compatibility) is
    unchanged from the paper.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.static_proj = nn.Linear(hidden_dim, hidden_dim)   # W_s
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)         # W_Q
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)         # W_K
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)         # W_V
        self.compat = nn.Linear(hidden_dim, 1)                  # W_o, scalar o_i per member
        self.dropout = nn.Dropout(dropout)

    def forward(self, member_embeds: torch.Tensor, member_mask: torch.Tensor) -> torch.Tensor:
        """
        member_embeds: (batch, max_set_size, hidden_dim)
        member_mask:   (batch, max_set_size) bool, True = real member
        returns:       (batch,) raw regression score per candidate hyperedge
        """
        B, L, H = member_embeds.shape

        static = torch.tanh(self.static_proj(member_embeds))  # s_i

        Q = self.q_proj(member_embeds).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(member_embeds).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(member_embeds).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, L, L)

        # Mask out: (a) padding keys, (b) the diagonal -- self-exclusion
        # is required by the paper (eq 3-5 exclude j=i explicitly).
        pad_mask = (~member_mask).unsqueeze(1).unsqueeze(2).expand(B, self.n_heads, L, L)
        self_mask = torch.eye(L, dtype=torch.bool, device=member_embeds.device).view(1, 1, L, L)
        full_mask = pad_mask | self_mask
        scores = scores.masked_fill(full_mask, float("-inf"))

        # Defensive guard: a real member with zero valid (non-self,
        # non-padding) neighbors would produce an all -inf row ->
        # softmax NaN. Shouldn't occur for set size >= 2, but guarded.
        all_masked = full_mask.all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, H)
        dynamic = torch.tanh(out)  # d_i

        diff_sq = (dynamic - static) ** 2       # (d_i - s_i)^2, "pseudo-euclidean distance"
        o = self.compat(diff_sq).squeeze(-1)    # o_i, (B, L)

        o = o.masked_fill(~member_mask, 0.0)
        denom = member_mask.sum(dim=-1).clamp(min=1).float()
        score = o.sum(dim=-1) / denom  # mean over real members, matching eq 7's pooling
        # NOTE: paper applies sigmoid(o_i) before averaging (probability
        # output for binary link existence); dropped here since this
        # project's target is a raw regression value (log1p count), not
        # a probability. See class docstring.
        return score


class HypergraphTemporalScorer(nn.Module):
    """
    Drop-in hyperedge counterpart to GraphTemporalScorer. Node encoder
    is now true hypergraph convolution (HypergraphNodeTemporalEncoder,
    above) plus a real structural relation, not pairwise-only. Final
    scoring step is HyperSAGNN.
    """
    def __init__(self, node_feat_dim: int, hidden_dim: int,
                 n_conv_layers: int = 2, use_gnn: bool = True, use_rnn: bool = True,
                 use_esm_context: bool = True, use_struct: bool = True,
                 esm_dim: int = 640, esm_adapter_dim: int = 32,
                 dropout: float = 0.0, use_horizon_embed: bool = True, max_horizon: int = 12,
                 n_attn_heads: int = 4):
        super().__init__()
        self.use_horizon_embed = use_horizon_embed
        self.node_encoder = HypergraphNodeTemporalEncoder(
            node_feat_dim, hidden_dim, n_conv_layers, use_gnn, use_rnn,
            use_esm_context=use_esm_context, use_struct=use_struct,
            esm_dim=esm_dim, esm_adapter_dim=esm_adapter_dim, dropout=dropout,
        )
        if use_horizon_embed:
            self.horizon_embed = nn.Embedding(max_horizon + 1, hidden_dim)
        self.hyperedge_scorer = HyperSAGNNScorer(hidden_dim, n_heads=n_attn_heads, dropout=dropout)

    def forward(self, node_feats_seq, incidence_seq, member_indices: torch.Tensor,
                member_mask: torch.Tensor, struct_adj: torch.Tensor | None = None,
                esm_seq=None, horizon_ids=None) -> torch.Tensor:
        """
        member_indices: (batch, max_set_size) long, node index per slot
                         (padding value doesn't matter, masked out anyway)
        member_mask:    (batch, max_set_size) bool
        struct_adj:     (N, N) static structural prior, same every call
        horizon_ids:    (batch,) long, broadcast onto every member if
                         use_horizon_embed
        """
        node_h = self.node_encoder(node_feats_seq, incidence_seq, struct_adj, esm_seq)  # (N, hidden_dim)
        member_embeds = node_h[member_indices]  # (batch, max_set_size, hidden_dim)

        if self.use_horizon_embed:
            assert horizon_ids is not None
            hz = self.horizon_embed(horizon_ids)          # (batch, hidden_dim)
            member_embeds = member_embeds + hz.unsqueeze(1)  # broadcast onto every member

        return self.hyperedge_scorer(member_embeds, member_mask)
