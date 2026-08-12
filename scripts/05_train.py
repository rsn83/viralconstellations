"""
Step 5: Train the mutation constellation model.

Training pipeline per triple (month_t, month_{t+h}, h):

  1. Filter     : GRU([c_{t-W},...,c_t]) → h_t
  2. Transition : TransitionModel × h steps → h_{t+1},...,h_{t+h}
  3. Freq loss  : FreqHead(h_{t+k}) vs real posfreq_{t+k}  [at each k]
  4. Denoise    : corrupt sequences from month t+h, train ConstellationTransformer
                  conditioned on h_{t+h}
  5. Length     : LengthToGoHead(h_{t+h}, h) vs observed mutation count

Multi-step frequency supervision (step 3) is the key signal that forces
the TransitionModel to learn realistic intermediate trajectories.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from viralconstellations.data.utils import load_vocab, train_test_split_months
from viralconstellations.model.model import (
    ConstellationTransformer, PopulationEncoder,
    LengthToGoHead, FrequencyRegressionHead,
    TrajectoryEmbeddingCache, reversion_noising, N_RESIDUES,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
)


def load_mat(d, m):     return np.load(d / f"{m}.npy")
def load_posfreq(d, m): return np.load(d / f"{m}_posfreq.npy")

def all_npy_months(matrix_dir):
    return sorted(p.stem for p in matrix_dir.glob("*.npy")
                  if "_posfreq" not in p.stem)


def get_hidden_state(traj_encoder, transition, cache, month_t, h_val, mode, W):
    """
    Filter → Transition → return final hidden state h_{t+h} and all states.
    For velocity mode, transition is None; return the encoder output directly.
    """
    if mode == "gru":
        window = cache.get_window(month_t, W)          # (W+1, d_model)
        h_t    = traj_encoder(window)                  # (d_model,)
        h_final, states = transition(h_t, h_val)       # (d_model,), list
    else:
        pf_t    = cache.get_posfreq(month_t)
        pf_prev = cache.get_posfreq_prev(month_t)
        h_t    = traj_encoder(pf_t, pf_prev)
        h_final, states = h_t, [h_t]
    return h_final, states


def one_triple_loss(
    model, traj_encoder, transition, length_head, freq_head,
    cache, triple, mode, W, T, device,
    denoise_w, length_w, freq_w, inter_w, batch_size=256,
):
    """Compute composite loss for one (month_t, mat_{t+h}, h) triple."""
    month_t, mat_th, h_val = triple
    all_m = cache._sorted_months

    # ── 1 & 2: Filter + Transition ────────────────────────────────────────
    h_final, states = get_hidden_state(
        traj_encoder, transition, cache, month_t, h_val, mode, W
    )

    # ── 3: Frequency loss at each intermediate step ───────────────────────
    t_idx = all_m.index(month_t)
    freq_losses = []
    for k in range(1, h_val + 1):
        if t_idx + k >= len(all_m): continue
        month_tk = all_m[t_idx + k]
        if month_tk not in cache._freq_cache: continue
        pf_target = cache.get_posfreq(month_tk)                  # (P, 21)
        h_k       = states[k] if k < len(states) else h_final    # (d_model,)
        fl        = freq_head.loss(h_k.unsqueeze(0), pf_target.unsqueeze(0))
        w         = 1.0 if k == h_val else inter_w
        freq_losses.append(w * fl)
    freq_loss = (torch.stack(freq_losses).mean()
                 if freq_losses else torch.tensor(0.0, device=device))

    # ── 4: Denoising loss (ConstellationTransformer, DILM-M) ──────────────
    n   = len(mat_th)
    idx = torch.randint(0, n, (min(batch_size, n),))
    x0  = torch.tensor(mat_th[idx.numpy()].astype(np.int64), device=device)
    B   = x0.shape[0]
    t   = torch.randint(1, T + 1, (B,), device=device)
    xt  = reversion_noising(x0, t, T)
    ctx = h_final.unsqueeze(0).expand(B, -1)
    h_t = torch.full((B,), h_val, dtype=torch.long, device=device)
    logits = model(xt, t, ctx, h_t)
    Bp, P_, C = logits.shape
    denoise_loss = F.cross_entropy(
        logits.reshape(Bp*P_, C), x0.reshape(Bp*P_), label_smoothing=0.05
    )

    # ── 5: Length-to-go loss (molecular clock) ────────────────────────────
    true_count  = torch.tensor(
        [(mat_th > 0).sum(axis=1).mean()], dtype=torch.float32, device=device
    )
    pred_count  = length_head(h_final.unsqueeze(0),
                              torch.tensor([h_val], dtype=torch.long, device=device))
    length_loss = F.mse_loss(pred_count, true_count)

    total = denoise_w * denoise_loss + length_w * length_loss + freq_w * freq_loss
    return {
        "total":   total,
        "denoise": denoise_loss.detach(),
        "length":  length_loss.detach(),
        "freq":    freq_loss.detach(),
    }


@torch.no_grad()
def val_loss(model, traj_encoder, transition, length_head, freq_head,
             cache, val_triples, mode, W, T, device,
             denoise_w, length_w, freq_w, inter_w):
    for m in [model, traj_encoder, length_head, freq_head]:
        m.eval()
    if transition: transition.eval()
    total, n = 0.0, 0
    for triple in val_triples:
        l = one_triple_loss(
            model, traj_encoder, transition, length_head, freq_head,
            cache, triple, mode, W, T, device,
            denoise_w, length_w, freq_w, inter_w,
        )
        total += l["total"].item(); n += 1
    return total / max(n, 1)


def main():
    cfg        = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    vocab_dir  = ROOT / cfg["paths"]["vocab_dir"]
    ckpt_dir   = ROOT / cfg["train"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    P         = len(pd.read_csv(vocab_dir / "position_vocab.tsv", sep="\t"))
    max_h     = cfg["horizon"]["max_h"]
    T         = cfg["model"]["diffusion_T"]
    d         = cfg["model"]["d_model"]
    denoise_w = cfg["horizon"]["denoising_weight"]
    length_w  = cfg["horizon"]["length_weight"]
    freq_w    = cfg["freq_head"]["freq_weight"]
    inter_w   = cfg["freq_head"]["intermediate_weight"]
    mode      = cfg["trajectory"]["mode"]
    W         = cfg["trajectory"]["window_size"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"P={P}  max_h={max_h}  T={T}  mode={mode}  W={W}")
    print(f"Device: {device}")

    avail = all_npy_months(matrix_dir)
    train_months, test_month = train_test_split_months(
        avail,
        cfg["train"]["train_months"] or None,
        cfg["train"]["test_month"]   or None,
    )
    print(f"Train: {train_months}")
    print(f"Test : {test_month}")

    # Build training triples: (month_t, mat_{t+h}, h)
    train_triples = []
    for i, mt in enumerate(train_months):
        for h in range(1, max_h + 1):
            j = i + h
            if j < len(train_months):
                train_triples.append((mt, load_mat(matrix_dir, train_months[j]), h))
    print(f"Training triples: {len(train_triples)}")

    val_triples = [(train_months[-1], load_mat(matrix_dir, test_month), 1)]

    # Build cache: all context months + window history + intermediate months
    cache_months = set()
    for mt, _, h in train_triples + val_triples:
        idx = avail.index(mt)
        for w in range(W + 1):
            cache_months.add(avail[max(0, idx - w)])
        for k in range(1, h + 1):
            if idx + k < len(avail):
                cache_months.add(avail[idx + k])
    cache_months = sorted(cache_months)
    ctx_mats  = {m: load_mat(matrix_dir, m)     for m in cache_months}
    ctx_freqs = {m: load_posfreq(matrix_dir, m) for m in cache_months}
    print(f"Cache months: {len(cache_months)}")

    # Build models
    encoder     = PopulationEncoder(P, d, cfg["model"]["phi_hidden"]).to(device)
    traj_enc    = build_trajectory_encoder(
        mode, d, cfg["trajectory"]["gru_hidden"], W, P
    ).to(device)
    transition  = TransitionModel(d).to(device) if mode == "gru" else None
    model       = ConstellationTransformer(
        P, d, cfg["model"]["n_heads"], cfg["model"]["n_layers"],
        cfg["model"]["dropout"], T, max_h,
    ).to(device)
    length_head = LengthToGoHead(d, max_h).to(device)
    freq_head   = FrequencyRegressionHead(d, P).to(device)

    modules = [model, encoder, traj_enc, length_head, freq_head]
    if transition: modules.append(transition)
    n_params = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    cache = TrajectoryEmbeddingCache(
        encoder, ctx_mats, ctx_freqs, device,
        cfg["model"]["deepsets_batch_size"],
    )
    all_params = [p for m in modules for p in m.parameters()]
    optimizer  = torch.optim.Adam(
        all_params, lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["n_epochs"]
    )

    best_val = float("inf")
    print("\nTraining...")

    for epoch in range(1, cfg["train"]["n_epochs"] + 1):
        cache.refresh()
        for m in modules: m.train()

        ep = {"total":0.0, "denoise":0.0, "freq":0.0, "length":0.0}
        perm = torch.randperm(len(train_triples)).tolist()
        for i in perm:
            losses = one_triple_loss(
                model, traj_enc, transition, length_head, freq_head,
                cache, train_triples[i], mode, W, T, device,
                denoise_w, length_w, freq_w, inter_w,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            for k in ep: ep[k] += losses[k].item() if torch.is_tensor(losses[k]) else losses[k]

        scheduler.step()
        n = len(train_triples)
        vl = val_loss(model, traj_enc, transition, length_head, freq_head,
                      cache, val_triples, mode, W, T, device,
                      denoise_w, length_w, freq_w, inter_w)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:3d}  "
                  f"denoise={ep['denoise']/n:.4f}  "
                  f"freq={ep['freq']/n:.4f}  "
                  f"length={ep['length']/n:.4f}  "
                  f"val={vl:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}")

        if vl < best_val:
            best_val = vl
            ckpt = dict(
                epoch=epoch, best_val=best_val,
                model_state=model.state_dict(),
                encoder_state=encoder.state_dict(),
                traj_state=traj_enc.state_dict(),
                length_state=length_head.state_dict(),
                freq_state=freq_head.state_dict(),
                n_positions=P, train_months=train_months,
                test_month=test_month, model_cfg=cfg["model"],
                traj_cfg=cfg["trajectory"], max_h=max_h,
            )
            if transition: ckpt["transition_state"] = transition.state_dict()
            torch.save(ckpt, ckpt_dir / "best_model.pt")

        if epoch % cfg["train"]["save_every"] == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")

    print(f"\nBest val: {best_val:.5f}  →  {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
