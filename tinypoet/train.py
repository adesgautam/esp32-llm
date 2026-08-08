"""
Complete retraining pipeline for TinyPoet v2 with:
  - 10+ MB poetry corpus
  - BPE tokenizer (256 tokens)
  - Improved hyperparameters (warmup + cosine decay, dropout, etc.)
  - Exports model + BPE tables for ESP32
"""
import os
import sys
import math
import time
import json
import struct
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tinypoet.tokenizer import BPETokenizer
from tinypoet.model import TinyPoet, TinyPoetConfig


class BPEDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        x = self.data[start_idx : start_idx + self.block_size]
        y = self.data[start_idx + 1 : start_idx + 1 + self.block_size]
        return x, y


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def export_bpe_model(model, config, tokenizer, out_bin_path):
    """Export model weights + BPE vocab table as a single binary for ESP32."""
    model_state = model.state_dict()

    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    bin_file = open(out_bin_path, "wb")

    # Header: magic, version, n_layer, n_head, n_embd, block_size, vocab_size
    header = struct.pack(
        "<7I",
        0x54504f45,  # 'TPOE'
        2,  # version 2 = BPE model
        config.n_layer,
        config.n_head,
        config.n_embd,
        config.block_size,
        config.vocab_size
    )
    bin_file.write(header)

    total_bytes = len(header)
    for k, v in model_state.items():
        arr = v.detach().cpu().numpy().astype(np.float32)
        data_bytes = arr.tobytes()
        bin_file.write(data_bytes)
        total_bytes += len(data_bytes)

    bin_file.close()
    print(f"Exported BPE model binary: {out_bin_path} ({total_bytes} bytes / {total_bytes/1024:.2f} KB)")
    return total_bytes


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("  TinyPoet v2 Retraining Pipeline")
    print(f"  Device: {device.upper()}")
    print("=" * 60)

    # ─── Step 1: Load corpus ───
    corpus_path = os.path.join("datasets", "raw", "lyrics_corpus.txt")
    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found! Run scripts/prepare_lyrics_data.py first.")
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    print(f"\nCorpus loaded: {len(corpus):,} chars ({len(corpus)/1024/1024:.2f} MB)")

    # ─── Step 2: Train BPE tokenizer ───
    VOCAB_SIZE = 256
    tok_path = os.path.join("datasets", "bpe_tokenizer.json")

    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    if os.path.exists(tok_path):
        print("Loading cached BPE tokenizer...")
        tokenizer.load(tok_path)
    else:
        tokenizer.train(corpus)
        tokenizer.save(tok_path)

    # Export C header for ESP32
    tokenizer.export_c_decode_table(os.path.join("firmware", "src", "bpe_vocab.h"))

    # ─── Step 3: Tokenize corpus ───
    print("\nTokenizing corpus with BPE...")
    tokens = tokenizer.encode(corpus)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"  Total BPE tokens: {len(data):,} (compression ratio: {len(corpus)/len(data):.2f}x)")

    # Train/Val split
    n_val = int(len(data) * 0.05)  # 5% validation (we have lots of data now)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    # ─── Step 4: Setup model ───
    BLOCK_SIZE = 64
    config = TinyPoetConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=2,
        n_head=4,
        n_embd=80,
        dropout=0.1,
        bias=True
    )

    model = TinyPoet(config).to(device)
    n_params = model.count_parameters()
    print(f"\nModel: {config.n_layer}L / {config.n_head}H / {config.n_embd}D")
    print(f"Parameters: {n_params:,}")
    print(f"Vocab size: {VOCAB_SIZE} (BPE)")

    # ─── Step 5: Training ───
    BATCH_SIZE = 64
    MAX_LR = 3e-4
    MIN_LR = 1e-5
    WEIGHT_DECAY = 5e-2
    EPOCHS = 150

    train_ds = BPEDataset(train_data, BLOCK_SIZE)
    val_ds = BPEDataset(val_data, BLOCK_SIZE)

    use_pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              pin_memory=use_pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            pin_memory=use_pin, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * 0.05)
    print(f"\nTraining: {EPOCHS} epochs, {len(train_loader)} batches/epoch, {total_steps} total steps")
    print(f"Warmup: {warmup_steps} steps, LR: {MAX_LR} -> {MIN_LR}")

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    global_step = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(global_step, warmup_steps, total_steps, MAX_LR, MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1
            global_step += 1

        avg_train_loss = total_train_loss / max(1, train_batches)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        val_bpc = avg_val_loss / math.log(2)
        val_ppl = math.exp(min(avg_val_loss, 20))  # cap to avoid overflow

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join("checkpoints", "tinypoet_v2_bpe.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_loss': avg_val_loss,
                'val_bpc': val_bpc,
                'val_ppl': val_ppl,
                'params': n_params,
                'tokenizer_path': tok_path,
            }, ckpt_path)

        if epoch % 10 == 0 or epoch <= 3 or epoch == EPOCHS:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"BPC: {val_bpc:.4f} | PPL: {val_ppl:.2f} | LR: {lr:.6f} | Time: {elapsed:.0f}s")

    elapsed = time.time() - start_time
    best_bpc = best_val_loss / math.log(2)
    best_ppl = math.exp(min(best_val_loss, 20))

    print(f"\n{'='*60}")
    print(f"  Training Complete! ({elapsed:.1f}s)")
    print(f"  Best Val Loss: {best_val_loss:.4f} | BPC: {best_bpc:.4f} | PPL: {best_ppl:.2f}")
    print(f"{'='*60}")

    # ─── Step 6: Generate sample poem ───
    model.eval()
    ckpt = torch.load("checkpoints/tinypoet_v2_bpe.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    prompts = ["the light ", "in the night ", "love is "]
    for prompt in prompts:
        prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(prompt_tokens, max_new_tokens=120, temperature=0.8, top_k=8)[0].tolist()
        poem = tokenizer.decode(out)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: {poem[:200]}")

    # ─── Step 7: Export for ESP32 ───
    print("\n\nExporting model binary for ESP32...")
    export_bpe_model(model, config, tokenizer, "firmware/src/model_weights.bin")

    # Save summary
    summary = {
        "model": "tinypoet_v2_bpe",
        "vocab_size": VOCAB_SIZE,
        "tokenizer": "BPE",
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "block_size": BLOCK_SIZE,
        "n_params": n_params,
        "epochs": EPOCHS,
        "corpus_size_mb": len(corpus) / 1024 / 1024,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_bpc": round(best_bpc, 4),
        "best_val_ppl": round(best_ppl, 2),
        "training_time_s": round(elapsed, 1),
    }
    with open("checkpoints/tinypoet_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone! Next steps:")
    print("  1. Update ESP32 main.c to use BPE vocab")
    print("  2. Rebuild firmware with: cd firmware && platformio run")
    print("  3. Flash with: python scripts/flash_firmware.py")


if __name__ == "__main__":
    train()
