"""
Viral Mutation Constellation Model.

Honest architecture description:
  A state space model whose latent state conditions a DILM-style
  discrete diffusion process, with a DeepSets population encoder
  and a deterministic frequency regression head.

Components and their origins:

  1. MutationSetEncoder / PopulationEncoder  [DeepSets, NeurIPS 2017]
     Two-level permutation-invariant aggregation.
     Inner:  φ(position, residue) → mean over one sequence's mutation set
     Outer:  mean over all sequences in one month → population embedding c_t

  2. GRUTrajectoryEncoder + TransitionModel  [Kalman filter / SSM structure]
     Filter:     GRU([c_{t-W},...,c_t]) → h_t  (backward, over observations)
     Transition: GRUCell × k steps → h_{t+k}    (forward, no observations)
     Together: explicit temporal state that propagates between months.

  3. ConstellationTransformer               [DILM / CTMC, AISTATS 2026]
     DILM-M generalization to unordered mutation sets.
     Forward process:  revert mutated positions to reference (deletion = reversion)
     Reverse process:  predict residue at each position from noisy state
     Conditioned on:   h_{t+h} (latent state) + horizon h + diffusion step t
     Output:           (P, 21) logits per position per denoising step

  4. LengthToGoHead                         [DILM-S length prediction]
     Predicts expected mutation count at horizon h.
     Encodes molecular clock: h=1 → small change, h=6 → larger change.

  5. FrequencyRegressionHead                [deterministic, not diffusion]
     Maps h_{t+k} → (P, 21) predicted posfreq at month t+k.
     Trained with KL divergence against empirical residue frequencies.
     This is the direct answer to "what frequency will each mutation be at?"
     No sampling required — evaluated with Pearson r against real posfreq.

One diffusion process (not two):
  Only the ConstellationTransformer uses diffusion.
  The FrequencyRegressionHead is deterministic.
  This is the correct description: CDiff's interacting diffusion mechanism
  is not implemented here.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

N_RESIDUES = 21   # 0=reference, 1-20=amino acids A..Y


# ── Dataset ──────────────────────────────────────────────────────────────────

class MultiHorizonDataset(Dataset):
    """
    Training examples for multi-horizon conditional generation.

    Each item:
      context_emb  (d_model,)  — cached population embedding from month t
                                  This is mean(φ(x_i)) before ρ is applied,
                                  so gradients flow through ρ at training time.
      target       (P,) int16  — categorical sequence from month t+h
      horizon      int         — h months ahead
      mut_count    float       — non-reference positions in target (for LengthHead)
    """
    def __init__(self, context_embs, targets, horizons, mut_counts):
        self.contexts   = context_embs   # (N, d_model)
        self.targets    = targets        # (N, P) int16
        self.horizons   = horizons       # (N,)
        self.mut_counts = mut_counts     # (N,)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx):
        return (self.contexts[idx], self.targets[idx],
                self.horizons[idx], self.mut_counts[idx])


# ── Noising (DILM forward process) ───────────────────────────────────────────

def survival_prob(t: int, T: int) -> float:
    """Cosine schedule: rho(0)=1 (intact), rho(T)=0 (all reverted)."""
    return 0.5 * (1.0 + np.cos(np.pi * t / T))


def reversion_noising(x: torch.Tensor, t: torch.Tensor, T: int) -> torch.Tensor:
    """
    DILM forward process adapted to categorical mutation sets.

    DILM deletes tokens → here we revert mutated positions to reference (0).
    Each non-reference position independently reverts with prob (1 - rho(t)).

    x : (B, P) int tensor, values 0-20
    t : (B,) int in [1, T]
    """
    rho  = torch.tensor(
        [survival_prob(ti.item(), T) for ti in t],
        dtype=torch.float32, device=x.device
    ).unsqueeze(-1)                                   # (B, 1)
    keep = torch.bernoulli(rho.expand(x.shape))       # (B, P)
    return x * keep.long()                            # revert where keep=0


# ── DeepSets: inner (per-sequence) ───────────────────────────────────────────

class MutationSetEncoder(nn.Module):
    """
    Inner DeepSets φ: encode ONE sequence as a set of (position, residue) pairs.

    φ(position j, residue k) = concat(pos_emb(j), res_emb(k)) → MLP
    Pool: mean over non-reference positions only.

    Key property: (position, residue) embeddings are factored, so unseen
    combinations at test time get embeddings from their components separately.
    This is what allows extrapolation to residues never seen during training
    at a given position.
    """
    def __init__(self, n_positions: int, d_model: int, phi_hidden: int = 256):
        super().__init__()
        d_half = d_model // 2
        self.pos_emb = nn.Embedding(n_positions, d_half)
        self.res_emb = nn.Embedding(N_RESIDUES, d_half)
        self.phi = nn.Sequential(
            nn.Linear(d_model, phi_hidden), nn.SiLU(),
            nn.Linear(phi_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P) int → (B, d_model) sequence embeddings."""
        B, P = x.shape
        pos_idx = torch.arange(P, device=x.device).unsqueeze(0).expand(B, -1)
        emb = torch.cat([self.pos_emb(pos_idx), self.res_emb(x)], dim=-1)
        emb = self.phi(emb)                                    # (B, P, d_model)
        is_mut = (x > 0).float().unsqueeze(-1)                 # (B, P, 1)
        n_mut  = is_mut.sum(dim=1).clamp(min=1.0)             # (B, 1)
        return (emb * is_mut).sum(dim=1) / n_mut               # (B, d_model)


# ── DeepSets: outer (population) ─────────────────────────────────────────────

class PopulationEncoder(nn.Module):
    """
    Outer DeepSets ρ: encode the POPULATION (set of sequences in one month).

    encode_population(): applies MutationSetEncoder to all sequences,
                         mean-pools, applies ρ → (d_model,).
    forward(): applies ρ to pre-pooled mean(φ(x_i)) — used at training
               time so gradients flow through ρ each batch step.
    """
    def __init__(self, n_positions: int, d_model: int, phi_hidden: int = 256):
        super().__init__()
        self.seq_encoder = MutationSetEncoder(n_positions, d_model, phi_hidden)
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            nn.SiLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

    def forward(self, pooled_emb: torch.Tensor) -> torch.Tensor:
        """pooled_emb: (B, d_model) → (B, d_model). Applies ρ only."""
        return self.rho(pooled_emb)

    @torch.no_grad()
    def encode_population(self, mat: np.ndarray, device: torch.device,
                          batch_size: int = 512) -> torch.Tensor:
        """
        Full pipeline: φ over all sequences → mean pool → ρ → (d_model,).
        Called once per month per epoch (cached). Not called per batch.
        """
        self.eval()
        phi_sum, n_total = None, 0
        for i in range(0, len(mat), batch_size):
            chunk   = torch.tensor(mat[i:i+batch_size].astype(np.int64), device=device)
            phi_out = self.seq_encoder(chunk)
            phi_sum = phi_out.sum(0) if phi_sum is None else phi_sum + phi_out.sum(0)
            n_total += len(chunk)
        pop_mean = phi_sum / n_total
        result   = self.rho(pop_mean.unsqueeze(0)).squeeze(0)
        self.train()
        return result


# ── Trajectory embedding cache ────────────────────────────────────────────────

class TrajectoryEmbeddingCache:
    """
    Stores per-month population embeddings and posfreq matrices.
    Refreshed once per epoch as encoder weights update.

    get_window(month_t, W): returns (W+1, d_model) stacked embeddings,
                            oldest first, most recent last.
                            Pads with earliest embedding if history is short.
    """
    def __init__(self, encoder, month_matrices, month_posfreqs,
                 device, batch_size=512):
        self.encoder        = encoder
        self.month_matrices = month_matrices
        self.month_posfreqs = month_posfreqs
        self.device         = device
        self.batch_size     = batch_size
        self._emb_cache: dict  = {}
        self._freq_cache: dict = {}
        self._sorted_months = sorted(month_matrices.keys())

    def refresh(self):
        self._emb_cache = {}; self._freq_cache = {}
        for m, mat in self.month_matrices.items():
            self._emb_cache[m] = self.encoder.encode_population(
                mat, self.device, self.batch_size
            ).detach()
            self._freq_cache[m] = torch.tensor(
                self.month_posfreqs[m], dtype=torch.float32, device=self.device
            )

    def get_embedding(self, month: str) -> torch.Tensor:
        return self._emb_cache[month]   # (d_model,)

    def get_window(self, month_t: str, W: int) -> torch.Tensor:
        all_m = self._sorted_months
        t_idx = all_m.index(month_t)
        idx   = [max(0, t_idx - W + i) for i in range(W + 1)]
        return torch.stack([self._emb_cache[all_m[i]] for i in idx])  # (W+1, d)

    def get_posfreq(self, month: str) -> torch.Tensor:
        return self._freq_cache[month]   # (P, 21)

    def get_posfreq_prev(self, month: str) -> torch.Tensor:
        all_m = self._sorted_months
        prev  = all_m[max(0, all_m.index(month) - 1)]
        return self._freq_cache[prev]


# ── Length-to-go head (DILM-S molecular clock) ───────────────────────────────

class LengthToGoHead(nn.Module):
    """
    Predicts expected mutation count at horizon h given hidden state h_{t+k}.

    Encodes the molecular clock: larger horizon → more mutations predicted.
    Loss: MSE against observed mutation count in real sequences.
    Output is non-negative (Softplus activation).
    """
    def __init__(self, d_model: int, max_h: int):
        super().__init__()
        self.d_model   = d_model
        self.max_h     = max_h
        self.h_proj    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(),
            nn.Linear(d_model, 1), nn.Softplus(),
        )

    def _h_emb(self, h: torch.Tensor) -> torch.Tensor:
        half  = self.d_model // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=h.device).float() / half
        )
        args  = h.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.h_proj(emb)

    def forward(self, hidden: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """hidden: (B, d_model), h: (B,) → (B,) predicted mut count."""
        return self.net(torch.cat([hidden, self._h_emb(h)], dim=-1)).squeeze(-1)


# ── Frequency regression head (deterministic) ─────────────────────────────────

class FrequencyRegressionHead(nn.Module):
    """
    Maps hidden state h_{t+k} → predicted posfreq at month t+k.

    Output: (B, P, 21) — per-position residue probability distributions.
    Each row sums to 1 (softmax over residues).

    This is the direct, deterministic answer to:
    "What frequency will each mutation be at h months from now?"

    Loss: KL divergence against empirical posfreq (cross-entropy on soft labels).
    Evaluation: Pearson r between predicted and real per-position mutation rates.

    NOT a diffusion process — purely deterministic regression.
    """
    def __init__(self, d_model: int, n_positions: int):
        super().__init__()
        self.P   = n_positions
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, n_positions * 21),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → (B, P, 21) probability distributions."""
        return F.softmax(self.net(h).reshape(-1, self.P, 21), dim=-1)

    def loss(self, h: torch.Tensor, target_posfreq: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss.
        target_posfreq: (B, P, 21) or (P, 21) empirical frequency matrix.
        """
        if target_posfreq.dim() == 2:
            target_posfreq = target_posfreq.unsqueeze(0).expand(h.shape[0], -1, -1)
        logits   = self.net(h).reshape(-1, self.P, 21)
        log_prob = F.log_softmax(logits, dim=-1)
        return -(target_posfreq * log_prob).sum(dim=-1).mean()


# ── Co-occurrence regression head (explicit joint prediction) ─────────────────

class CooccurrenceRegressionHead(nn.Module):
    """
    Maps h_{t+h} → predicted pairwise co-occurrence matrix.

    Output: upper triangle of (P, P) matrix, ~P*(P-1)/2 values.
    Entry [i,j] = P(position i AND position j are both mutated in month t+h).

    This is the EXPLICIT answer to "does background structure predict
    future co-occurrence beyond the independence assumption?"

    Independence baseline predicts: P(i AND j) = freq_i × freq_j
    This head predicts: P(i AND j) directly from the population trajectory.

    If this head's predicted matrix correlates better with the real
    co-occurrence matrix than the independence baseline does, then
    the model has learned joint structure — the central scientific claim.

    Training loss: binary cross-entropy against empirical co-occurrence.
    (BCE is correct here: each pair is a probability in [0,1].)
    """
    def __init__(self, d_model: int, n_positions: int):
        super().__init__()
        self.P       = n_positions
        n_pairs      = n_positions * (n_positions - 1) // 2
        self.n_pairs = n_pairs
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, n_pairs),
            nn.Sigmoid(),   # output is probability in [0,1]
        )
        # Upper triangle indices — computed once, registered as buffer
        iu = torch.triu_indices(n_positions, n_positions, offset=1)
        self.register_buffer("iu_row", iu[0])
        self.register_buffer("iu_col", iu[1])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) → (B, n_pairs) predicted co-occurrence probs."""
        return self.net(h)

    def loss(self, h: torch.Tensor, mat_th: np.ndarray) -> torch.Tensor:
        """
        Compute BCE loss against empirical pairwise co-occurrence.

        h:      (B, d_model) — hidden state, same for all seqs in this month
        mat_th: (n_seq, P) int8 — real sequences from target month

        Empirical co-occurrence[i,j] = fraction of sequences where
        both position i AND position j are mutated (non-zero).
        """
        # Empirical co-occurrence from real sequences — all on h.device
        bin_mat  = torch.tensor(
            (mat_th > 0).astype(np.float32), device=h.device
        )                                                    # (n_seq, P)
        n        = len(bin_mat)
        coo_full = (bin_mat.T @ bin_mat) / n                # (P, P) on h.device
        # iu_row/iu_col are registered buffers — already on h.device
        target   = coo_full[self.iu_row, self.iu_col]       # (n_pairs,)

        # Predicted
        pred = self.forward(h)                               # (B, n_pairs)
        # Broadcast target to match batch
        target = target.unsqueeze(0).expand(pred.shape[0], -1)

        return F.binary_cross_entropy(pred, target)

    def predict_matrix(self, h: torch.Tensor) -> np.ndarray:
        """Return (P, P) predicted co-occurrence matrix."""
        with torch.no_grad():
            pairs = self.forward(h.unsqueeze(0)).squeeze(0).cpu().numpy()
        mat = np.zeros((self.P, self.P), dtype=np.float32)
        iu_r = self.iu_row.cpu().numpy()
        iu_c = self.iu_col.cpu().numpy()
        mat[iu_r, iu_c] = pairs
        mat[iu_c, iu_r] = pairs   # symmetric
        return mat


def independence_cooccurrence(posfreq: np.ndarray) -> np.ndarray:
    """
    Independence baseline: P(i AND j) = freq_i × freq_j.
    posfreq: (P, 21) — per-position residue frequencies.
    Returns (P, P) predicted co-occurrence under independence assumption.
    """
    mut_rates = 1.0 - posfreq[:, 0]    # P(any mutation at position j)
    return np.outer(mut_rates, mut_rates).astype(np.float32)


def evaluate_cooccurrence(
    pred_coo:  np.ndarray,   # (P, P) model prediction
    real_coo:  np.ndarray,   # (P, P) empirical from real sequences
    indep_coo: np.ndarray,   # (P, P) independence baseline
    P:         int,
) -> dict:
    """
    Compare model co-occurrence prediction against real and independence baseline.
    Key metric: does model Pearson r > independence Pearson r?
    """
    iu         = np.triu_indices(P, 1)
    pred_vals  = pred_coo[iu]
    real_vals  = real_coo[iu]
    indep_vals = indep_coo[iu]

    r_model, _ = pearsonr(pred_vals,  real_vals)
    r_indep, _ = pearsonr(indep_vals, real_vals)

    return {
        "coo_pearson_r_model":   float(r_model),
        "coo_pearson_r_indep":   float(r_indep),
        "coo_mse_model":         float(np.mean((pred_vals  - real_vals)**2)),
        "coo_mse_indep":         float(np.mean((indep_vals - real_vals)**2)),
        "model_beats_indep":     bool(r_model > r_indep),
        "delta_coo_pearson_r":   float(r_model - r_indep),
    }


# ── Constellation transformer (DILM-M reverse process) ───────────────────────

class ConstellationTransformer(nn.Module):
    """
    DILM-M generalized to unordered mutation sets.

    At each denoising step, given the current noisy categorical sequence x_t,
    predict P(true residue k at position j) for all j simultaneously.

    Conditioning:
      - Diffusion step t  (sinusoidal embedding)
      - Horizon h         (sinusoidal embedding)
      - Hidden state ctx  (from GRU filter + TransitionModel)

    All three combined and broadcast over all P positions.
    Transformer self-attention is permutation-equivariant — correct for
    unordered mutation sets (no positional ordering assumption).

    Per-position embedding: separate learned embeddings for:
      - The RESIDUE at each position (state_emb, 21 states)
      - The POSITION itself (position_emb, P positions)
    Both contribute to what the transformer sees at each site.
    """
    def __init__(self, n_positions, d_model=128, n_heads=4, n_layers=3,
                 dropout=0.1, max_T=100, max_h=6):
        super().__init__()
        self.P     = n_positions
        self.max_T = max_T
        d = d_model

        self.state_emb    = nn.Embedding(N_RESIDUES, d)
        self.position_emb = nn.Embedding(n_positions, d)

        for name in ("time_proj", "horizon_proj"):
            setattr(self, name, nn.Sequential(
                nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
            ))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4*d,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out         = nn.Linear(d, N_RESIDUES)

        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def _sin_emb(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        d    = self.state_emb.embedding_dim
        half = d // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=x.device).float() / half
        )
        args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return proj(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))

    def forward(self, x_t, t, ctx, h):
        """
        x_t : (B, P) int — noisy categorical sequence
        t   : (B,) int   — diffusion step
        ctx : (B, d_model) — hidden state h_{t+horizon} from SSM
        h   : (B,) int   — horizon
        Returns: (B, P, N_RESIDUES) logits
        """
        B, P = x_t.shape
        pos  = torch.arange(P, device=x_t.device).unsqueeze(0).expand(B, -1)
        emb  = self.state_emb(x_t) + self.position_emb(pos)           # (B, P, d)
        cond = self._sin_emb(t, self.time_proj) + \
               self._sin_emb(h, self.horizon_proj) + ctx               # (B, d)
        emb  = emb + cond.unsqueeze(1)                                 # (B, P, d)
        return self.out(self.transformer(emb))                         # (B, P, 21)


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_from_hidden(model, length_head, h_state, horizon,
                         n_samples, P, T, device, temperature=1.0):
    """
    Generate n_samples mutation constellations using a pre-computed hidden state.

    DILM-S length-controlled generation — three steps:

    Step 1 — Molecular clock (LengthToGoHead):
      Predict how many mutations each generated sequence should have.
      target_counts[i] = predicted mutation count for sequence i at this horizon.
      This encodes the molecular clock: h=1 → small increase from current mean,
      h=6 → larger increase. The rate is learned from training data, not hardcoded.

    Step 2 — Residue selection (DILM-M denoising):
      Run the reverse diffusion process T→0 to determine WHICH RESIDUE belongs
      at each variable position. This encodes co-occurrence structure — positions
      that tend to mutate together are jointly predicted by the transformer.

    Step 3 — Count enforcement (DILM-S stopping criterion):
      Do one final forward pass to get P(mutated) at each position.
      For each sequence, keep only the top-k positions by P(mutated),
      where k = target_count from Step 1.
      This is the DILM-S stopping criterion adapted to sets:
      instead of stopping when length-to-go = 0, we hard-enforce the count.

    Why the three-step decomposition?
      Steps 1 and 2 are independent: the LengthToGoHead controls HOW MANY
      mutations; the transformer controls WHICH RESIDUE at each position.
      Step 3 combines them: it selects which positions to use (by P(mutated))
      and what residue each selected position gets (from the denoised x).
      This mirrors DILM-S exactly: length head says how long the sequence is,
      the insertion process fills it in.
    """
    model.eval(); length_head.eval()
    ctx = h_state.unsqueeze(0).expand(n_samples, -1)       # (N, d_model)
    h_t = torch.full((n_samples,), horizon, dtype=torch.long, device=device)

    # ── Step 1: Molecular clock — how many mutations? ─────────────────────
    target_counts = (
        length_head(ctx, h_t)
        .round()
        .clamp(min=1, max=P)
        .long()
    )                                                       # (N,)
    print(f"    Predicted mean mutations at h={horizon}: "
          f"{target_counts.float().mean().item():.1f}")

    # ── Step 2: Denoising — which residue at each position? ───────────────
    # Start from all-reference (all zeros = no mutations)
    x = torch.zeros(n_samples, P, dtype=torch.long, device=device)
    for t_val in range(T, 0, -1):
        t_tens = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
        logits = model(x, t_tens, ctx, h_t) / temperature  # (N, P, 21)
        probs  = F.softmax(logits, dim=-1).reshape(-1, N_RESIDUES)
        x      = torch.multinomial(probs, 1).squeeze(-1).reshape(n_samples, P)
        rho    = survival_prob(t_val - 1, T)
        keep   = torch.bernoulli(
            torch.full((n_samples, P), rho, device=device)
        ).long()
        x = x * keep                                        # (N, P) sampled residues

    # ── Step 3: Count enforcement — keep top-k by P(mutated) ─────────────
    # Final forward pass at t=1 to get clean per-position probabilities
    t_final      = torch.full((n_samples,), 1, dtype=torch.long, device=device)
    final_logits = model(x, t_final, ctx, h_t)             # (N, P, 21)
    final_probs  = F.softmax(final_logits, dim=-1)          # (N, P, 21)

    # P(mutated at position j) = 1 - P(reference residue 0)
    p_mutated = 1.0 - final_probs[:, :, 0]                 # (N, P)

    # For each sequence i: keep only the top-k positions by p_mutated,
    # using the sampled residue from the denoising process at those positions.
    # Positions not in the top-k revert to reference (0).
    final = torch.zeros(n_samples, P, dtype=torch.long, device=device)
    for i in range(n_samples):
        k          = int(target_counts[i].item())
        topk_pos   = p_mutated[i].topk(k).indices          # (k,)
        final[i, topk_pos] = x[i, topk_pos]

    return final.cpu().numpy().astype(np.int8)


# ── Independence baseline ─────────────────────────────────────────────────────

def independence_baseline_generate(posfreq, n_samples, seed=42):
    """
    Baseline: sample each position independently from its empirical distribution.
    posfreq: (P, 21) float — per-position residue frequencies.
    This is what the model must outperform to claim it captures co-occurrence.
    """
    rng = np.random.default_rng(seed)
    P   = posfreq.shape[0]
    out = np.zeros((n_samples, P), dtype=np.int8)
    for j in range(P):
        p = posfreq[j].astype(np.float64)
        p = p / p.sum()
        out[:, j] = rng.choice(N_RESIDUES, n_samples, p=p)
    return out
