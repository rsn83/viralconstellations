"""
Discriminative frontier model — three components:

  1. Trajectory encoder (frozen from existing checkpoint):
     GRU([posfreq_{t-W},...,posfreq_t]) → h_t → TransitionModel × h → h_{t+h}
     FrequencyRegressionHead → predicted posfreq_{t+h}
     Weights frozen — already proven to work (Pearson r=0.98).

  2. Candidate encoder (new, DeepSets on the RIGHT object):
     Input: candidate c = frozenset of (position, residue) pairs
     φ(pos_j, res_j, h_{t+h}) → mean pool over mutations → ρ → emb(c)
     Key: h_{t+h} is fed INTO φ so the embedding of each mutation is
     conditioned on where the population is going (Option C).
     This is DeepSets applied to the candidate constellation, not the population.

  3. Scorer (discriminative MLP):
     concat(emb(c), hand_features) → MLP → sigmoid → P(c appears)
     hand_features: 7 features from frontier.py (parent freq, trend, etc.)

Training objective: weighted BCE on (frontier candidate, window) pairs.
  Positive = candidate appeared in O_{t+h} but not O_t
  Negative = frontier candidate that didn't appear

Walk-forward evaluation: train on first N windows, evaluate on last M.
Comparison: random baseline, logistic regression (hand features only),
            neural discriminative (learned embedding + hand features).
"""

import numpy as np
import torch
import torch.nn as nn


N_RESIDUES  = 21   # 0=reference, 1-20=amino acids
N_HAND_FEAT = 7    # features from frontier.py extract_features


class CandidateEncoder(nn.Module):
    """
    DeepSets encoder for a candidate mutation constellation.

    Encodes c = {(pos_1, res_1), ..., (pos_k, res_k)} as a fixed-size vector.
    Represented as a (P,) categorical sequence: 0=reference, 1-20=residue.

    φ per mutation:
        concat(pos_emb(j), res_emb(r), h_{t+h}) → MLP → mutation embedding
        h_{t+h} is included so each mutation's embedding is conditioned on
        where the population trajectory is heading (Option C).

    Mean pool over mutated positions only → ρ → emb(c).

    Why this is DeepSets on the RIGHT object:
        The existing PopulationEncoder applies DeepSets to SEQUENCES (mean
        over sequences), discarding within-sequence co-occurrence.
        This encoder applies DeepSets to MUTATIONS within one constellation,
        preserving the set structure of the candidate.
    """
    def __init__(self, n_positions: int, d_model: int, phi_hidden: int = 128):
        super().__init__()
        d_half = d_model // 2
        self.pos_emb = nn.Embedding(n_positions, d_half)
        self.res_emb = nn.Embedding(N_RESIDUES, d_half)

        # φ: (pos_emb + res_emb + h_context) → mutation embedding
        self.phi = nn.Sequential(
            nn.Linear(d_model + d_model, phi_hidden), nn.SiLU(),
            nn.Linear(phi_hidden, d_model),
        )
        # ρ: (pooled mutations + h_context) → constellation embedding
        self.rho = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, P) int — candidate sequences (0=ref, 1-20=residue)
        h_ctx: (B, d_model) — population hidden state h_{t+h}
        Returns: (B, d_model) — candidate constellation embeddings
        """
        B, P = x.shape
        pos_idx = torch.arange(P, device=x.device).unsqueeze(0).expand(B, -1)

        # Per-position embeddings: (B, P, d_model)
        emb = torch.cat([self.pos_emb(pos_idx), self.res_emb(x)], dim=-1)

        # Condition φ on population context at each position
        ctx_exp = h_ctx.unsqueeze(1).expand(B, P, -1)   # (B, P, d_model)
        phi_in  = torch.cat([emb, ctx_exp], dim=-1)      # (B, P, 2*d_model)
        phi_out = self.phi(phi_in)                        # (B, P, d_model)

        # Mean pool over MUTATED positions only (ignore reference positions)
        is_mut  = (x > 0).float().unsqueeze(-1)          # (B, P, 1)
        n_mut   = is_mut.sum(dim=1).clamp(min=1.0)       # (B, 1)
        pooled  = (phi_out * is_mut).sum(dim=1) / n_mut  # (B, d_model)

        # ρ: combine pooled mutations with population context
        return self.rho(torch.cat([pooled, h_ctx], dim=-1))  # (B, d_model)


class FrontierDiscriminator(nn.Module):
    """
    Scores P(candidate c appears in O_{t+h}) given:
      - emb(c): learned candidate embedding (already conditioned on h_{t+h})
      - hand_features: 7 interpretable features from frontier.py

    Note: h_{t+h} is already encoded in emb(c) via the CandidateEncoder's
    Option C conditioning. We don't add it again here to avoid redundancy.

    The hand features provide an inductive bias from domain knowledge:
    parent frequency, trend direction, lattice depth, etc.
    The learned embedding adds nonlinear interaction between mutations
    in the candidate and the population trajectory context.

    Comparison:
      LogisticFrontierScorer: hand_features only → linear boundary
      FrontierDiscriminator:  emb(c) + hand_features → nonlinear boundary
    If FrontierDiscriminator AP > LogisticFrontierScorer AP:
      learned embeddings add value beyond hand-crafted features.
    """
    def __init__(self, d_model: int, n_hand_features: int = N_HAND_FEAT):
        super().__init__()
        in_dim = d_model + n_hand_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb_c: torch.Tensor,
                hand_features: torch.Tensor) -> torch.Tensor:
        """
        emb_c:         (B, d_model)
        hand_features: (B, n_hand_features)
        Returns:       (B,) predicted appearance probabilities
        """
        return self.net(torch.cat([emb_c, hand_features], dim=-1)).squeeze(-1)


class DiscriminativeModel(nn.Module):
    """Wraps CandidateEncoder + FrontierDiscriminator for convenience."""
    def __init__(self, n_positions: int, d_model: int,
                 phi_hidden: int = 128, n_hand: int = N_HAND_FEAT):
        super().__init__()
        self.encoder     = CandidateEncoder(n_positions, d_model, phi_hidden)
        self.discriminator = FrontierDiscriminator(d_model, n_hand)

    def forward(self, x: torch.Tensor, h_ctx: torch.Tensor,
                hand: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x, h_ctx)
        return self.discriminator(emb, hand)

    def score_candidates(self, candidate_seqs: np.ndarray,
                         hand_features: np.ndarray,
                         h_state: torch.Tensor,
                         device: torch.device) -> np.ndarray:
        """
        Score a batch of frontier candidates.
        candidate_seqs: (N, P) int8
        hand_features:  (N, 7) float32
        Returns: (N,) float scores in [0,1]
        """
        self.eval()
        with torch.no_grad():
            x    = torch.tensor(candidate_seqs.astype(np.int64), device=device)
            hand = torch.tensor(hand_features, device=device)
            ctx  = h_state.unsqueeze(0).expand(len(x), -1)
            return self(x, ctx, hand).cpu().numpy()
