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
  python scripts/train_qat.py                   # Micro-LM-Pro (default)
  python scripts/train_qat.py --config micro_lm_ultra  # Ultra config
  python scripts/train_qat.py --epochs 50        # Override epoch count
"""
import os
import sys
import math
import time
import json
import struct
import shutil
import argparse
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.backends.cudnn as cudnn

# Enable TF32 for NVIDIA Tensor Cores (Tesla T4 / Ampere / Ada)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from micro_lm.tokenizer import BPETokenizer
from micro_lm.model import ESP32LLM, ESP32LLMConfig
from micro_lm.quantization import replace_linear_with_ternary
from micro_lm.train import BPEDataset, get_lr


CONFIG_TO_NAME = {
    "micro_lm_pico": "Micro-LM-Pico",
    "micro_lm_pro": "Micro-LM-Pro",
    "micro_lm_ultra": "Micro-LM-Ultra",
    "micro_lm_mega": "Micro-LM-Mega",
    "micro_lm_s3_large": "Micro-LM-S3-Large",
    "micro_lm_colossus": "Micro-LM-Colossus"
}

DEFAULT_BATCH_SIZES = {
    "micro_lm_pico": 64,
    "micro_lm_pro": 32,
    "micro_lm_ultra": 32,
    "micro_lm_s3_large": 16,
    "micro_lm_mega": 16,
    "micro_lm_colossus": 8
}

DEFAULT_MAX_CHARS = {
    "micro_lm_pico": 4_000_000,      # 4 MB (~1M tokens) - optimal for 207K params
    "micro_lm_pro": 12_000_000,      # 12 MB (~3M tokens) - optimal for 3M params
    "micro_lm_ultra": 25_000_000,    # 25 MB (~6M tokens) - optimal for 11M params
    "micro_lm_s3_large": 45_000_000, # 45 MB (~11M tokens) - optimal for 26M params
    "micro_lm_mega": 60_000_000,     # 60 MB
    "micro_lm_colossus": 80_000_000  # 80 MB
}

# --- Logging ---
class TrainingLogger:
    def __init__(self, log_dir="checkpoints", config_name="micro_lm_pro"):
        self.model_name = CONFIG_TO_NAME.get(config_name, config_name)
        self.config_dir = os.path.join(log_dir, self.model_name)
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

    import zlib
    crc = 0

    header_offset = bin_file.tell()
    bin_file.write(b'\x00' * 40) # Dummy 10-int header
    total_bytes = 40

    def write_fp32(tensor):
        nonlocal total_bytes, crc
        data = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
        crc = zlib.crc32(data, crc)
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
        crc = zlib.crc32(packed, crc)
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

    # Rewrite header with checksum
    bin_file.seek(header_offset)
    header = struct.pack(
        "<10I",
        0x54504f45,  # 'TPOE'
        3,           # version 3 = QAT Ternary
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        config.n_embd,
        config.block_size,
        config.vocab_size,
        crc & 0xFFFFFFFF,
        (total_bytes - 40) # payload size
    )
    bin_file.write(header)

    bin_file.close()
    return total_bytes


# ─── GPU-Accelerated Batching & Validation ──────────────────
def get_batch(data_tensor, block_size, batch_size, device):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,), device=device)
    x = torch.stack([data_tensor[i : i + block_size] for i in ix])
    y = torch.stack([data_tensor[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    num_samples = (len(val_data) - 1) // block_size
    total_batches = max(1, min(50, num_samples // batch_size))
    for step in range(total_batches):
        start = step * batch_size * block_size
        indices = torch.arange(start, start + batch_size * block_size, block_size, device=device)
        indices = indices[indices + block_size < len(val_data)]
        if len(indices) == 0:
            break
        x = torch.stack([val_data[i : i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1 : i + 1 + block_size] for i in indices])
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        total_loss += loss.item()
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
    parser = argparse.ArgumentParser(description="ESP32LLM QAT Training")
    parser.add_argument("--config", choices=["micro_lm_pro", "micro_lm_ultra", "micro_lm_mega", "micro_lm_pico", "micro_lm_s3_large", "micro_lm_colossus"], default="micro_lm_pro",
                        help="Model architecture config (default: micro_lm_pro)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Physical batch size per step (auto-tuned for GPU Tensor Cores if None)")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective_bs = batch_size * accum) (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Path to teacher checkpoint for Knowledge Distillation (e.g. checkpoints/Micro-LM-Ultra/Micro-LM-Ultra_qat.pth)")
    parser.add_argument("--kd_alpha", type=float, default=0.5,
                        help="Weight for KD loss (0.0 = CE only, 1.0 = KD only, default: 0.5)")
    parser.add_argument("--kd_temp", type=float, default=2.0,
                        help="Softmax temperature for Knowledge Distillation (default: 2.0)")
    parser.add_argument("--max_chars", type=int, default=None,
                        help="Maximum characters to load from corpus (auto-scaled per model if None)")
    parser.add_argument("--gdrive_dir", type=str, default="/content/drive/MyDrive/esp32_llm_checkpoints",
                        help="Destination folder on Google Drive for continuous backup")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    # Auto-tune batch size for 100% GPU utilization if not provided
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZES.get(args.config, 32)

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
    
    max_chars = args.max_chars if args.max_chars is not None else DEFAULT_MAX_CHARS.get(args.config, 25_000_000)
    if max_chars > 0 and len(corpus) > max_chars:
        corpus = corpus[:max_chars]
        log.print(f"Corpus scaled for {config_name}: {len(corpus):,} chars ({len(corpus)/1024/1024:.2f} MB)")
    else:
        log.print(f"\nCorpus: {len(corpus):,} chars ({len(corpus)/1024/1024:.2f} MB)")

    # ─── Tokenizer ───
    VOCAB_SIZE = 256
    tok_path = os.path.join("datasets", "bpe_tokenizer.json")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load(tok_path)

    log.print("Tokenizing corpus...")
    t0 = time.time()
    tokens = tokenizer.encode(corpus)
    # Move token dataset directly to GPU memory for zero-copy training
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    log.print(f"  BPE tokens: {len(data):,} (compression: {len(corpus)/len(data):.2f}x) [{time.time()-t0:.1f}s]")
    log.print(f"  Dataset VRAM: {data.element_size() * data.nelement() / 1024 / 1024:.2f} MB on {device.upper()}")

    # Train/Val split (5% val)
    n_val = int(len(data) * 0.05)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    log.print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")

    # ─── Model ───
    if args.config == "micro_lm_pro":
        config = ESP32LLMConfig.micro_lm_pro()
    elif args.config == "micro_lm_ultra":
        config = ESP32LLMConfig.micro_lm_ultra()
    elif args.config == "micro_lm_pico":
        config = ESP32LLMConfig.micro_lm_pico()
    elif args.config == "micro_lm_s3_large":
        config = ESP32LLMConfig.micro_lm_s3_large()
    elif args.config == "micro_lm_colossus":
        config = ESP32LLMConfig.micro_lm_colossus()
    else:
        config = ESP32LLMConfig.micro_lm_mega()

    model = ESP32LLM(config)
    model = replace_linear_with_ternary(model)
    model = model.to(device)

    n_params = model.count_parameters()
    log.print(f"\nArchitecture: {config.n_layer}L / {config.n_head}H / {config.n_embd}D (MQA kv_head={config.n_kv_head})")
    log.print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    log.print(f"Context: {config.block_size} tokens")
    log.print(f"Precision: Ternary QAT (1.58-bit weights via STE)")

    # ─── Steps Calculation ───
    steps_per_epoch = max(1, (len(train_data) - config.block_size) // (batch_size * config.block_size))

    # ─── Optimizer & Schedule ───
    MAX_LR = args.lr
    MIN_LR = MAX_LR / 30  # ~1e-5
    WEIGHT_DECAY = 0.05
    EPOCHS = args.epochs
    ACCUM = args.accum_steps

    use_fused = (device == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY, fused=use_fused)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = EPOCHS * steps_per_epoch // ACCUM
    warmup_steps = int(total_steps * 0.05)

    effective_bs = batch_size * ACCUM
    log.print(f"\nTraining Config:")
    log.print(f"  Epochs: {EPOCHS}")
    log.print(f"  Batch size: {batch_size} × {ACCUM} accum = {effective_bs} effective")
    log.print(f"  Steps/epoch: {steps_per_epoch} ({steps_per_epoch//ACCUM} optimizer steps)")
    log.print(f"  Total optimizer steps: {total_steps}")
    log.print(f"  Warmup: {warmup_steps} steps")
    log.print(f"  LR: {MAX_LR} → {MIN_LR} (cosine)")
    log.print(f"  Weight decay: {WEIGHT_DECAY}")
    log.print(f"  Mixed precision (AMP): {use_amp}")
    log.print(f"  Fused AdamW: {use_fused}")

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

    # ─── Knowledge Distillation (Teacher) ───
    teacher_model = None
    if args.teacher and os.path.exists(args.teacher):
        log.print(f"\n[KD] Loading Teacher Model from {args.teacher}...")
        t_ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
        t_config = t_ckpt.get('config')
        if t_config is None:
            t_config = ESP32LLMConfig.micro_lm_ultra()
        teacher_model = ESP32LLM(t_config)
        teacher_model = replace_linear_with_ternary(teacher_model)
        teacher_model.load_state_dict(t_ckpt['model_state_dict'])
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        log.print(f"  [KD] Teacher loaded: {teacher_model.count_parameters():,} params ({teacher_model.count_parameters()/1e6:.2f}M)")
        log.print(f"  [KD] Distillation Alpha={args.kd_alpha} | Temp={args.kd_temp}")

    # ─── Initial eval ───
    val_loss, val_bpc, val_ppl = evaluate(model, val_data, config.block_size, batch_size, device, use_amp)
    log.print(f"\nPre-training eval: Val Loss={val_loss:.4f} | BPC={val_bpc:.4f} | PPL={val_ppl:.2f}")

    # ─── Training ───
    model_name = CONFIG_TO_NAME.get(args.config, args.config)
    ckpt_name = f"{model_name}_qat.pth"
    ckpt_path = os.path.join(log.config_dir, ckpt_name)
    global_step = 0
    train_start = time.time()

    log.print(f"\n{'='*65}")
    log.print(f"  Starting Training" + (" [with Knowledge Distillation]" if teacher_model else ""))
    log.print(f"{'='*65}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        for step in range(steps_per_epoch):
            x, y = get_batch(train_data, config.block_size, batch_size, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                s_logits, loss_ce = model(x, y)

                if teacher_model is not None:
                    with torch.no_grad():
                        t_logits, _ = teacher_model(x)
                    T = args.kd_temp
                    loss_kd = F.kl_div(
                        F.log_softmax(s_logits / T, dim=-1),
                        F.softmax(t_logits / T, dim=-1),
                        reduction="batchmean"
                    ) * (T * T)
                    loss = (1.0 - args.kd_alpha) * loss_ce + args.kd_alpha * loss_kd
                else:
                    loss = loss_ce

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
        val_loss, val_bpc, val_ppl = evaluate(model, val_data, config.block_size, batch_size, device, use_amp)

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

        # Resilient GDrive sync on every epoch
        if args.gdrive_dir and os.path.exists(args.gdrive_dir):
            try:
                gdrive_ckpt_dir = os.path.join(args.gdrive_dir, "checkpoints", model_name)
                os.makedirs(gdrive_ckpt_dir, exist_ok=True)
                if os.path.exists(ckpt_path):
                    shutil.copy2(ckpt_path, gdrive_ckpt_dir)
                shutil.copy2(os.path.join(log.config_dir, "training_progress.json"), gdrive_ckpt_dir)
                shutil.copy2(log.log_path, gdrive_ckpt_dir)
            except Exception:
                pass

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

    bin_path = f"firmware/src/{model_name}.bin"
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
        "model": model_name,
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

    # Final full GDrive sync
    if args.gdrive_dir and os.path.exists(args.gdrive_dir):
        try:
            gdrive_fw_dir = os.path.join(args.gdrive_dir, "firmware_binaries")
            gdrive_ckpt_dir = os.path.join(args.gdrive_dir, "checkpoints", model_name)
            os.makedirs(gdrive_fw_dir, exist_ok=True)
            os.makedirs(gdrive_ckpt_dir, exist_ok=True)
            shutil.copy2(bin_path, gdrive_fw_dir)
            shutil.copy2(summary_path, gdrive_ckpt_dir)
            shutil.copy2(log.log_path, gdrive_ckpt_dir)
            if os.path.exists(ckpt_path):
                shutil.copy2(ckpt_path, gdrive_ckpt_dir)
            log.print(f"  [GDrive Sync]: Checkpoint & {model_name}.bin exported to {args.gdrive_dir}")
        except Exception as e:
            log.print(f"  [GDrive Sync Warning: {e}]")

    log.print("\nNext: test interactively with:")
    log.print(r"  .venv\Scripts\python scripts/interactive_gpu.py")
    log.close()


if __name__ == "__main__":
    train_qat()
