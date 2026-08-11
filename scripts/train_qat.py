"""
ESP32LLM Quantization-Aware Training (QAT) Pipeline
Trains Ternary (1.58-bit) models with proper convergence:
  - Cosine LR schedule with warmup
  - Gradient accumulation for effective larger batch
  - Validation loop with best-checkpoint saving
  - Periodic generation samples
  - Mixed precision (AMP) for speed
  - Full logging to a training log file

Usage:
  python scripts/train_qat.py                   # Option C (default, 3.1M)
  python scripts/train_qat.py --config option_b  # Option B (11.4M)
  python scripts/train_qat.py --epochs 50        # Override epoch count
"""
import os
import sys
import math
import time
import json
import struct
import argparse
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.backends.cudnn as cudnn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from micro_lm.tokenizer import BPETokenizer
from micro_lm.model import ESP32LLM, ESP32LLMConfig
from micro_lm.quantization import replace_linear_with_ternary
from micro_lm.train import BPEDataset, get_lr


# --- Logging ---
class TrainingLogger:
    def __init__(self, log_dir="checkpoints", config_name="option_c"):
        self.config_dir = os.path.join(log_dir, config_name.lower().replace(" ", "_"))
        os.makedirs(self.config_dir, exist_ok=True)
        # Force UTF-8 stdout on Windows
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        self.log_path = os.path.join(self.config_dir, f"train_qat.log")
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        # Add a visual separator for new sessions
        self.log_file.write("\n" + "="*65 + "\n")
        self.log_file.write(f"  SESSION RESUMED: {datetime.datetime.now().isoformat()}\n")
        self.log_file.write("="*65 + "\n")
        self.print(f"Training log: {self.log_path}")

    def print(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ─── Ternary Binary Export ─────────────────────────────────
def pack_ternary_weights(weight_tensor):
    """Pack {-1, 0, 1} weights: 4 per byte. Mapping: -1->00, 0->01, 1->10"""
    flat = weight_tensor.flatten().cpu().numpy()
    num_weights = len(flat)
    bytes_len = (num_weights + 3) // 4
    packed = bytearray(bytes_len)
    mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
    for i in range(bytes_len):
        b = 0
        for j in range(4):
            idx = i * 4 + j
            if idx < num_weights:
                val = round(float(flat[idx]))
                bits = mapping.get(float(val), 1)
                b |= (bits << (j * 2))
        packed[i] = b
    return packed


def export_qat_model(model, config, tokenizer, out_bin_path):
    """Export Ternary model weights for the ESP32 C Engine."""
    model.eval()
    state_dict = model.state_dict()
    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    bin_file = open(out_bin_path, "wb")

    header = struct.pack(
        "<8I",
        0x54504f45,  # 'TPOE'
        3,           # version 3 = QAT Ternary
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        config.n_embd,
        config.block_size,
        config.vocab_size
    )
    bin_file.write(header)
    total_bytes = len(header)

    def write_fp32(tensor):
        nonlocal total_bytes
        data = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
        bin_file.write(data)
        total_bytes += len(data)

    def write_ternary_linear(prefix):
        nonlocal total_bytes
        weight = state_dict[f"{prefix}.weight"]
        bias = state_dict.get(f"{prefix}.bias", None)
        scale = weight.abs().mean().clamp(min=1e-8)
        weight_norm = weight / scale
        weight_q = torch.round(weight_norm).clamp(-1, 1)
        write_fp32(scale)
        packed = pack_ternary_weights(weight_q)
        bin_file.write(packed)
        total_bytes += len(packed)
        if bias is not None:
            write_fp32(bias)
        else:
            write_fp32(torch.zeros(weight.size(0)))

    # 1. Embeddings
    write_fp32(state_dict["transformer.wte.weight"])

    # 2. Layers
    for l in range(config.n_layer):
        prefix = f"transformer.h.{l}"
        write_fp32(state_dict[f"{prefix}.ln_1.weight"])
        write_fp32(state_dict[f"{prefix}.ln_1.bias"])
        write_ternary_linear(f"{prefix}.attn.q_proj")
        write_ternary_linear(f"{prefix}.attn.k_proj")
        write_ternary_linear(f"{prefix}.attn.v_proj")
        write_ternary_linear(f"{prefix}.attn.c_proj")
        write_fp32(state_dict[f"{prefix}.ln_2.weight"])
        write_fp32(state_dict[f"{prefix}.ln_2.bias"])
        write_ternary_linear(f"{prefix}.mlp.c_fc")
        write_ternary_linear(f"{prefix}.mlp.c_proj")

    # 3. Final layernorm
    write_fp32(state_dict["transformer.ln_f.weight"])
    write_fp32(state_dict["transformer.ln_f.bias"])

    bin_file.close()
    return total_bytes


# ─── Validation ────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        total_loss += loss.item()
        total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    bpc = avg_loss / math.log(2)
    ppl = math.exp(min(avg_loss, 20))
    model.train()
    return avg_loss, bpc, ppl


# ─── Generation Sample ────────────────────────────────────
@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="I ", max_tokens=60):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=8)[0].tolist()
    text = tokenizer.decode(out)
    model.train()
    return text


# ─── Main Training Loop ───────────────────────────────────
def train_qat():
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="ESP32LLM QAT Training")
    parser.add_argument("--config", choices=["option_c", "option_b", "option_a_plus", "option_pico"], default="option_c",
                        help="Model architecture config (default: option_c)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Physical batch size per step (default: 8)")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective_bs = batch_size * accum) (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    log = TrainingLogger("checkpoints", args.config)
    config_name = args.config.upper().replace("_", " ")

    log.print("=" * 65)
    log.print(f"  ESP32LLM QAT Training — {config_name}")
    log.print(f"  Device: {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    log.print(f"  Started: {datetime.datetime.now().isoformat()}")
    log.print("=" * 65)

    # ─── Load corpus ───
    corpus_path = os.path.join("datasets", "raw", "lyrics_corpus.txt")
    if not os.path.exists(corpus_path):
        log.print(f"ERROR: {corpus_path} not found!")
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    if args.config == "option_pico":
        corpus = corpus[:50000] # Super fast training
    log.print(f"\nCorpus: {len(corpus):,} chars ({len(corpus)/1024/1024:.2f} MB)")

    # ─── Tokenizer ───
    VOCAB_SIZE = 256
    tok_path = os.path.join("datasets", "bpe_tokenizer.json")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load(tok_path)

    log.print("Tokenizing corpus...")
    t0 = time.time()
    tokens = tokenizer.encode(corpus)
    data = torch.tensor(tokens, dtype=torch.long)
    log.print(f"  BPE tokens: {len(data):,} (compression: {len(corpus)/len(data):.2f}x) [{time.time()-t0:.1f}s]")

    # Train/Val split (5% val)
    n_val = int(len(data) * 0.05)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    log.print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")

    # ─── Model ───
    if args.config == "option_c":
        config = ESP32LLMConfig.option_c()
    elif args.config == "option_b":
        config = ESP32LLMConfig.option_b()
    elif args.config == "option_pico":
        config = ESP32LLMConfig.option_pico()
    else:
        config = ESP32LLMConfig.option_a_plus()

    model = ESP32LLM(config)
    model = replace_linear_with_ternary(model)
    model = model.to(device)

    n_params = model.count_parameters()
    log.print(f"\nArchitecture: {config.n_layer}L / {config.n_head}H / {config.n_embd}D (MQA kv_head={config.n_kv_head})")
    log.print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    log.print(f"Context: {config.block_size} tokens")
    log.print(f"Precision: Ternary QAT (1.58-bit weights via STE)")

    # ─── Dataloaders ───
    train_ds = BPEDataset(train_data, config.block_size)
    val_ds = BPEDataset(val_data, config.block_size)

    use_pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=use_pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=use_pin, num_workers=0)

    # ─── Optimizer & Schedule ───
    MAX_LR = args.lr
    MIN_LR = MAX_LR / 30  # ~1e-5
    WEIGHT_DECAY = 0.05
    EPOCHS = args.epochs
    ACCUM = args.accum_steps

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = EPOCHS * len(train_loader) // ACCUM
    warmup_steps = int(total_steps * 0.05)

    effective_bs = args.batch_size * ACCUM
    log.print(f"\nTraining Config:")
    log.print(f"  Epochs: {EPOCHS}")
    log.print(f"  Batch size: {args.batch_size} × {ACCUM} accum = {effective_bs} effective")
    log.print(f"  Steps/epoch: {len(train_loader)} ({len(train_loader)//ACCUM} optimizer steps)")
    log.print(f"  Total optimizer steps: {total_steps}")
    log.print(f"  Warmup: {warmup_steps} steps")
    log.print(f"  LR: {MAX_LR} → {MIN_LR} (cosine)")
    log.print(f"  Weight decay: {WEIGHT_DECAY}")
    log.print(f"  Mixed precision (AMP): {use_amp}")

    # ─── Resume ───
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
        log.print(f"\n  Resumed from {args.resume} (epoch {start_epoch})")

    # ─── Initial eval ───
    val_loss, val_bpc, val_ppl = evaluate(model, val_loader, device, use_amp)
    log.print(f"\nPre-training eval: Val Loss={val_loss:.4f} | BPC={val_bpc:.4f} | PPL={val_ppl:.2f}")

    # ─── Training ───
    ckpt_name = f"esp32_llm_qat_{args.config}.pth"
    ckpt_path = os.path.join("checkpoints", ckpt_name)
    global_step = 0
    train_start = time.time()

    log.print(f"\n{'='*65}")
    log.print(f"  Starting Training")
    log.print(f"{'='*65}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / ACCUM  # scale for accumulation

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * ACCUM  # unscale for logging
            epoch_batches += 1

            if (step + 1) % ACCUM == 0:
                # Update LR
                lr = get_lr(global_step, warmup_steps, total_steps, MAX_LR, MIN_LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

        avg_train_loss = epoch_loss / max(1, epoch_batches)
        epoch_time = time.time() - epoch_start

        # ─── Validation ───
        val_loss, val_bpc, val_ppl = evaluate(model, val_loader, device, use_amp)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_bpc': val_bpc,
                'val_ppl': val_ppl,
                'best_val_loss': best_val_loss,
                'global_step': global_step,
                'args': vars(args),
            }, ckpt_path)
            improved = " ★ BEST"

        # Log every epoch
        elapsed = time.time() - train_start
        log.print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"BPC: {val_bpc:.4f} | "
            f"PPL: {val_ppl:.2f} | "
            f"LR: {lr:.6f} | "
            f"Time: {epoch_time:.0f}s | "
            f"Total: {elapsed/60:.1f}m{improved}"
        )

        # Dump progress report
        progress_report = {
            "config": args.config,
            "epoch": epoch,
            "total_epochs": EPOCHS,
            "val_loss": round(val_loss, 4),
            "val_bpc": round(val_bpc, 4),
            "val_ppl": round(val_ppl, 2),
            "best_val_loss": round(best_val_loss, 4),
            "best_ppl": round(math.exp(min(best_val_loss, 20)), 2),
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(os.path.join(log.config_dir, "training_progress.json"), "w") as f:
            json.dump(progress_report, f, indent=2)

        # ─── Generation sample every epoch ───
        if True:
            prompts = ["I ", "the night ", "love is "]
            for p in prompts:
                sample = generate_sample(model, tokenizer, device, prompt=p, max_tokens=40)
                log.print(f"  [{p.strip()}] → {sample[:120]}")
            log.print("")

    # ─── Final Summary ───
    total_time = time.time() - train_start
    best_bpc = best_val_loss / math.log(2)
    best_ppl = math.exp(min(best_val_loss, 20))

    log.print(f"\n{'='*65}")
    log.print(f"  Training Complete!")
    log.print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
    log.print(f"  Best Val Loss: {best_val_loss:.4f}")
    log.print(f"  Best Val BPC:  {best_bpc:.4f}")
    log.print(f"  Best Val PPL:  {best_ppl:.2f}")
    log.print(f"  Checkpoint: {ckpt_path}")
    log.print(f"{'='*65}")

    # ─── Export best model binary ───
    log.print("\nLoading best checkpoint for export...")
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    bin_path = f"firmware/src/model_weights_{args.config}.bin"
    total_bytes = export_qat_model(model, config, tokenizer, bin_path)
    log.print(f"Exported: {bin_path} ({total_bytes:,} bytes / {total_bytes/1024/1024:.2f} MB)")

    # Final generation samples
    log.print("\n--- Final Generation Samples ---")
    test_prompts = [
        "I ", "the night ", "love is ", "in the dark ",
        "money on my ", "she said ", "city lights "
    ]
    for p in test_prompts:
        sample = generate_sample(model, tokenizer, device, prompt=p, max_tokens=80)
        log.print(f"  [{p.strip()}] → {sample[:200]}")

    # Save summary JSON
    summary = {
        "model": f"esp32_llm_qat_{args.config}",
        "config": args.config,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "n_embd": config.n_embd,
        "block_size": config.block_size,
        "vocab_size": config.vocab_size,
        "n_params": n_params,
        "precision": "ternary_qat_1.58bit",
        "epochs": EPOCHS,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_bpc": round(best_bpc, 4),
        "best_val_ppl": round(best_ppl, 2),
        "training_time_min": round(total_time / 60, 1),
        "corpus_size_mb": round(len(corpus) / 1024 / 1024, 2),
        "binary_size_bytes": total_bytes,
        "binary_size_mb": round(total_bytes / 1024 / 1024, 2),
    }
    summary_path = os.path.join(log.config_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.print(f"\nSummary saved: {summary_path}")

    log.print(f"\nNext: test interactively with:")
    log.print(f"  .venv\\Scripts\\python scripts/interactive_gpu.py")
    log.close()


if __name__ == "__main__":
    train_qat()
