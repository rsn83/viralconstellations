"""
Trajectory Encoder and Transition Model.

Implements the state space model (SSM) structure:

  Filter (backward):    GRU over observed monthly embeddings → h_t
  Transition (forward): GRUCell × k steps → h_{t+k}

Together these are the neural analog of a Kalman filter:
  - Filter step:      observe new month → update hidden state h_t
  - Prediction step:  propagate h_t forward k months without new data

The filter is the GRUTrajectoryEncoder (reads past observations).
The prediction is the TransitionModel (extrapolates into the future).

Alternative: VelocityEncoder — encodes one-step change only, no transition.
"""

import torch
import torch.nn as nn


class GRUTrajectoryEncoder(nn.Module):
    """
    Filter GRU: [c_{t-W}, ..., c_t] → h_t.

    Processes the observed window of monthly population embeddings
    chronologically. Final hidden state h_t is the filtered state.
    Temporal position embeddings tell the GRU which month is current.
    """
    def __init__(self, d_model: int, gru_hidden: int, window_size: int):
        super().__init__()
        self.d_model = d_model
        W = window_size + 1
        self.temporal_pos = nn.Embedding(W, d_model)
        self.gru  = nn.GRU(d_model, gru_hidden, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden, d_model), nn.LayerNorm(d_model)
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: (W+1, d) or (B, W+1, d) → (d,) or (B, d)."""
        sq = window.dim() == 2
        if sq: window = window.unsqueeze(0)
        B, T, _ = window.shape
        pos = torch.arange(T, device=window.device).unsqueeze(0).expand(B, -1)
        inp = window + self.temporal_pos(pos)
        _, h_n = self.gru(inp)
        out = self.proj(h_n.squeeze(0))   # (B, d)
        return out.squeeze(0) if sq else out


class TransitionModel(nn.Module):
    """
    Prediction step: h_t → h_{t+1} → ... → h_{t+k}.

    A GRUCell with a learned step_token as input — "advance one month"
    without any observation. Residual connection keeps changes small
    per step (realistic for monthly viral evolution).

    step(h): one step forward.
    forward(h, k): roll k steps, return final state + all intermediate states.
    Intermediate states are used for multi-step frequency supervision.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.step_token = nn.Parameter(torch.zeros(d_model))
        self.gru_cell   = nn.GRUCell(d_model, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def step(self, h: torch.Tensor) -> torch.Tensor:
        sq = h.dim() == 1
        if sq: h = h.unsqueeze(0)
        token = self.step_token.unsqueeze(0).expand(h.shape[0], -1)
        out   = self.norm(self.gru_cell(token, h) + h)
        return out.squeeze(0) if sq else out

    def forward(self, h_t: torch.Tensor, k: int):
        """
        Returns: (h_{t+k}, [h_t, h_{t+1}, ..., h_{t+k}])
        states[i] has same shape as h_t.
        """
        sq = h_t.dim() == 1
        if sq: h_t = h_t.unsqueeze(0)
        h = h_t; states = [h]
        for _ in range(k):
            h = self.step(h); states.append(h)
        if sq:
            return states[-1].squeeze(0), [s.squeeze(0) for s in states]
        return states[-1], states


class VelocityEncoder(nn.Module):
    """
    Alternative to GRU+Transition: encodes [posfreq_t, Δposfreq] → (d_model,).
    One-step velocity only. No forward propagation capability.
    Useful as ablation: compare velocity vs full GRU+Transition.
    """
    def __init__(self, n_positions: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_positions * 21 * 2, d_model * 2),
            nn.LayerNorm(d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )

    def forward(self, pf_t, pf_prev):
        sq = pf_t.dim() == 2
        if sq: pf_t = pf_t.unsqueeze(0); pf_prev = pf_prev.unsqueeze(0)
        B = pf_t.shape[0]
        x = torch.cat([pf_t.reshape(B,-1), (pf_t-pf_prev).reshape(B,-1)], -1)
        out = self.net(x)
        return out.squeeze(0) if sq else out


def build_trajectory_encoder(mode, d_model, gru_hidden, window_size, n_positions):
    if mode == "gru":      return GRUTrajectoryEncoder(d_model, gru_hidden, window_size)
    if mode == "velocity": return VelocityEncoder(n_positions, d_model)
    raise ValueError(f"Unknown trajectory mode: {mode!r}")
