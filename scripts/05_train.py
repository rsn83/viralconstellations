"""
Step 5: Train the mutation constellation model.

Usage:
  python scripts/05_train.py                              # uses configs/default.yaml
  python scripts/05_train.py --config configs/colab_2022_test.yaml

Training losses printed every epoch with flush=True so Colab shows them immediately.
Any error is caught and printed with full traceback before exiting.
"""

import sys, argparse, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml, numpy as np, pandas as pd, torch, torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()

cfg        = yaml.safe_load(open(ROOT / args.config))
matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
vocab_dir  = ROOT / cfg["paths"]["vocab_dir"]
ckpt_dir   = ROOT / cfg["train"]["checkpoint_dir"]
ckpt_dir.mkdir(parents=True, exist_ok=True)

from viralconstellations.data.utils import train_test_split_months
from viralconstellations.model.model import (
    ConstellationTransformer, PopulationEncoder,
    LengthToGoHead, FrequencyRegressionHead, CooccurrenceRegressionHead,
    TrajectoryEmbeddingCache, reversion_noising, N_RESIDUES,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
)


def log(msg):
    """Print with flush so Colab shows it immediately."""
    print(msg, flush=True)


def load_mat(m):     return np.load(matrix_dir / f"{m}.npy")
def load_pf(m):      return np.load(matrix_dir / f"{m}_posfreq.npy")
def mat_exists(m):   return (matrix_dir / f"{m}.npy").exists()

def month_plus(s, h):
    y,mo=int(s[:4]),int(s[5:7]); mo+=h; y+=(mo-1)//12; mo=(mo-1)%12+1
    return f"{y:04d}-{mo:02d}"

def month_minus(s, k=1):
    y,mo=int(s[:4]),int(s[5:7]); mo-=k
    while mo<=0: mo+=12; y-=1
    return f"{y:04d}-{mo:02d}"


def main():
    log("="*60)
    log(f"Starting training | config: {args.config}")
    log("="*60)

    try:
        P      = len(pd.read_csv(vocab_dir / "position_vocab.tsv", sep="\t"))
        max_h  = cfg["horizon"]["max_h"]
        T      = cfg["model"]["diffusion_T"]
        d      = cfg["model"]["d_model"]
        mode   = cfg["trajectory"]["mode"]
        W      = cfg["trajectory"]["window_size"]
        dw     = cfg["horizon"]["denoising_weight"]
        lw     = cfg["horizon"]["length_weight"]
        fw     = cfg["freq_head"]["freq_weight"]
        iw     = cfg["freq_head"]["intermediate_weight"]
        cw     = cfg["cooc_head"]["cooc_weight"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log(f"P={P}  d={d}  T={T}  mode={mode}  W={W}  max_h={max_h}")
        log(f"Device: {device}")
        log(f"Losses: denoise×{dw}  freq×{fw}  length×{lw}  cooc×{cw}")

        # Available months
        all_avail = sorted(p.stem for p in matrix_dir.glob("*.npy")
                           if "_posfreq" not in p.stem)
        log(f"Available months: {len(all_avail)}  ({all_avail[0]} → {all_avail[-1]})")

        train_months, test_month = train_test_split_months(
            all_avail,
            cfg["train"]["train_months"] or None,
            cfg["train"]["test_month"]   or None,
        )
        log(f"Train months: {len(train_months)}  ({train_months[0]} → {train_months[-1]})")
        log(f"Test month:   {test_month}")

        # Build training triples
        triples = []
        for i, mt in enumerate(train_months):
            for h in range(1, max_h + 1):
                j = i + h
                if j < len(train_months) and mat_exists(train_months[j]):
                    triples.append((mt, load_mat(train_months[j]), h))
        log(f"Training triples: {len(triples)}")

        val_triples = [(train_months[-1], load_mat(test_month), 1)]

        # Cache months
        cache_months = set()
        for mt, _, h in triples + val_triples:
            idx = all_avail.index(mt) if mt in all_avail else -1
            for w in range(W + 1):
                cache_months.add(all_avail[max(0, idx - w)])
            for k in range(1, h + 1):
                tk = month_plus(mt, k)
                if mat_exists(tk): cache_months.add(tk)
        cache_months = sorted(cache_months)
        log(f"Cache months: {len(cache_months)}")

        ctx_mats  = {m: load_mat(m) for m in cache_months}
        ctx_freqs = {m: load_pf(m)  for m in cache_months}

        # Build models
        encoder     = PopulationEncoder(P, d, cfg["model"]["phi_hidden"]).to(device)
        traj_enc    = build_trajectory_encoder(
            mode, d, cfg["trajectory"]["gru_hidden"], W, P).to(device)
        transition  = TransitionModel(d).to(device)
        model       = ConstellationTransformer(
            P, d, cfg["model"]["n_heads"], cfg["model"]["n_layers"],
            cfg["model"]["dropout"], T, max_h).to(device)
        length_head = LengthToGoHead(d, max_h).to(device)
        freq_head   = FrequencyRegressionHead(d, P).to(device)
        cooc_head   = CooccurrenceRegressionHead(d, P).to(device)

        modules    = [model, encoder, traj_enc, transition,
                      length_head, freq_head, cooc_head]
        all_params = [p for m in modules for p in m.parameters()]
        n_params   = sum(p.numel() for p in all_params if p.requires_grad)
        log(f"Parameters: {n_params:,}")

        cache = TrajectoryEmbeddingCache(
            encoder, ctx_mats, ctx_freqs, device,
            cfg["model"]["deepsets_batch_size"],
        )
        optimizer = torch.optim.Adam(
            all_params, lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["n_epochs"]
        )
        all_m = sorted(cache_months)

        best_val = float("inf")
        log("\nTraining started...")
        log(f"{'Epoch':>5} {'denoise':>9} {'freq':>9} {'cooc':>9} {'length':>9} {'val':>9} {'lr':>9}")
        log("-"*65)

        for epoch in range(1, cfg["train"]["n_epochs"] + 1):
            cache.refresh()
            for m in modules: m.train()

            ep = {k: 0.0 for k in ["denoise","freq","cooc","length","total"]}
            n_tri = 0

            for mt, mat_th, h_val in triples:
                try:
                    # ── Filter + Transition ───────────────────────────────
                    window  = cache.get_window(mt, W)
                    h_t     = traj_enc(window)
                    h_final, states = transition(h_t, h_val)

                    # ── Frequency loss at each step ───────────────────────
                    t_idx = all_m.index(mt) if mt in all_m else -1
                    freq_losses = []
                    for k in range(1, h_val + 1):
                        if t_idx + k >= len(all_m): continue
                        mk = all_m[t_idx + k]
                        if mk not in cache._freq_cache: continue
                        h_k = states[k] if k < len(states) else h_final
                        fl  = freq_head.loss(h_k.unsqueeze(0),
                                             cache.get_posfreq(mk).unsqueeze(0))
                        freq_losses.append((1.0 if k==h_val else iw) * fl)
                    freq_loss = (torch.stack(freq_losses).mean()
                                 if freq_losses else torch.tensor(0.0, device=device))

                    # ── Co-occurrence loss at final step ──────────────────
                    cooc_loss = cooc_head.loss(h_final.unsqueeze(0), mat_th)

                    # ── Denoising loss ────────────────────────────────────
                    n   = len(mat_th)
                    idx = torch.randint(0, n, (min(cfg["train"]["batch_size"], n),))
                    x0  = torch.tensor(mat_th[idx.numpy()].astype(np.int64), device=device)
                    B   = x0.shape[0]
                    t   = torch.randint(1, T+1, (B,), device=device)
                    xt  = reversion_noising(x0, t, T)
                    ctx = h_final.unsqueeze(0).expand(B, -1)
                    ht2 = torch.full((B,), h_val, dtype=torch.long, device=device)
                    logits = model(xt, t, ctx, ht2)
                    Bp, Pv, C = logits.shape
                    denoise_loss = F.cross_entropy(
                        logits.reshape(Bp*Pv, C), x0.reshape(Bp*Pv),
                        label_smoothing=0.05
                    )

                    # ── Length loss (Poisson NLL) ──────────────────────────
                    # LengthToGoHead outputs log(count).
                    # Poisson NLL stays stable for counts 5-55.
                    if lw > 0:
                        true_cnt = torch.tensor(
                            [(mat_th>0).sum(1).mean()], dtype=torch.float32,
                            device=device
                        ).clamp(min=1.0)
                        log_pred = length_head(
                            h_final.unsqueeze(0),
                            torch.tensor([h_val], dtype=torch.long, device=device)
                        )   # (1,) log(count)
                        length_loss = F.poisson_nll_loss(
                            log_pred, true_cnt, log_input=True, full=False
                        )
                    else:
                        length_loss = torch.tensor(0.0, device=device)

                    # ── Combined loss ─────────────────────────────────────
                    loss = (dw * denoise_loss + fw * freq_loss +
                            cw * cooc_loss   + lw * length_loss)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, cfg["train"]["grad_clip"])
                    optimizer.step()

                    ep["denoise"] += denoise_loss.item()
                    ep["freq"]    += freq_loss.item()
                    ep["cooc"]    += cooc_loss.item()
                    ep["length"]  += length_loss.item()
                    ep["total"]   += loss.item()
                    n_tri += 1

                except Exception as e:
                    log(f"\nERROR in triple ({mt}, h={h_val}): {e}")
                    traceback.print_exc()
                    raise

            scheduler.step()
            n = max(n_tri, 1)

            # Validation loss
            val_loss = 0.0
            try:
                for m in modules: m.eval()
                with torch.no_grad():
                    for mt_v, mat_v, h_v in val_triples:
                        window_v = cache.get_window(mt_v, W)
                        h_t_v    = traj_enc(window_v)
                        h_f_v, _ = transition(h_t_v, h_v)
                        x0_v = torch.tensor(mat_v[:64].astype(np.int64), device=device)
                        B_v  = len(x0_v)
                        t_v  = torch.randint(1, T+1, (B_v,), device=device)
                        xt_v = reversion_noising(x0_v, t_v, T)
                        ctx_v = h_f_v.unsqueeze(0).expand(B_v, -1)
                        ht_v  = torch.full((B_v,), h_v, dtype=torch.long, device=device)
                        lg_v  = model(xt_v, t_v, ctx_v, ht_v)
                        Bv,Pv2,Cv = lg_v.shape
                        val_loss = F.cross_entropy(
                            lg_v.reshape(Bv*Pv2,Cv), x0_v.reshape(Bv*Pv2)
                        ).item()
            except Exception as e:
                log(f"  Val loss error: {e}")
                val_loss = -1.0
            for m in modules: m.train()

            # Print every epoch
            log(f"{epoch:>5d} {ep['denoise']/n:>9.4f} {ep['freq']/n:>9.4f} "
                f"{ep['cooc']/n:>9.4f} {ep['length']/n:>9.4f} "
                f"{val_loss:>9.4f} {scheduler.get_last_lr()[0]:>9.2e}")

            if val_loss > 0 and val_loss < best_val:
                best_val = val_loss
                ckpt = dict(
                    epoch=epoch, best_val=best_val,
                    model_state=model.state_dict(),
                    encoder_state=encoder.state_dict(),
                    traj_state=traj_enc.state_dict(),
                    transition_state=transition.state_dict(),
                    length_state=length_head.state_dict(),
                    freq_state=freq_head.state_dict(),
                    cooc_state=cooc_head.state_dict(),
                    n_positions=P,
                    train_months=train_months,
                    test_month=test_month,
                    model_cfg=cfg["model"],
                    traj_cfg=cfg["trajectory"],
                    max_h=max_h,
                )
                torch.save(ckpt, ckpt_dir / "best_model.pt")
                log(f"  → Saved best model (val={best_val:.4f})")

            if epoch % cfg["train"]["save_every"] == 0:
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")

        log(f"\nTraining complete. Best val loss: {best_val:.5f}")
        log(f"Checkpoint: {ckpt_dir / 'best_model.pt'}")

    except KeyboardInterrupt:
        log("\nTraining interrupted by user.")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
