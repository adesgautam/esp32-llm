import os
import sys
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from micro_lm.tokenizer import BPETokenizer
from micro_lm.model import ESP32LLM, ESP32LLMConfig
from micro_lm.quantization import replace_linear_with_ternary, apply_ptq
from micro_lm.train import BPEDataset

def calculate_perplexity(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Option C Formal Benchmarking ({device}) ---")

    # Load Data
    corpus_path = os.path.join("datasets", "raw", "lyrics_corpus.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.load(os.path.join("datasets", "bpe_tokenizer.json"))
    tokens = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)

    # 5% Validation set
    n_val = int(len(tokens) * 0.05)
    train_data = tokens[:-n_val]
    val_data = tokens[-n_val:]

    config = ESP32LLMConfig.option_c()
    train_ds = BPEDataset(train_data, config.block_size)
    val_ds = BPEDataset(val_data, config.block_size)
    
    use_pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=use_pin)

    print(f"\n1. Training FP32 Baseline (Option C)...")
    fp32_model = ESP32LLM(config).to(device)
    optimizer = torch.optim.AdamW(fp32_model.parameters(), lr=3e-4)
    fp32_model.train()
    
    # Train FP32 briefly to get a baseline structure
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = fp32_model(x, y)
        loss.backward()
        optimizer.step()
        if step > 200: # Brief training
            break
            
    fp32_ppl, fp32_loss = calculate_perplexity(fp32_model, val_loader, device)
    print(f"[FP32 Baseline] PPL: {fp32_ppl:.2f} | Loss: {fp32_loss:.4f}")

    print(f"\n2. Applying PTQ (Ternary 1.58-bit)...")
    # Clone model for PTQ
    ptq_model = ESP32LLM(config)
    ptq_model.load_state_dict(fp32_model.state_dict())
    ptq_model = apply_ptq(ptq_model, bits="ternary").to(device)
    
    ptq_ppl, ptq_loss = calculate_perplexity(ptq_model, val_loader, device)
    print(f"[Ternary PTQ] PPL: {ptq_ppl:.2f} | Loss: {ptq_loss:.4f}")

    print(f"\n3. Running Quantization-Aware Training (QAT)...")
    # Start fresh for QAT but copy embeddings from FP32 for a head start
    qat_model = ESP32LLM(config)
    qat_model = replace_linear_with_ternary(qat_model)
    qat_model.transformer.wte.weight.data.copy_(fp32_model.transformer.wte.weight.data)
    qat_model = qat_model.to(device)
    
    optimizer_qat = torch.optim.AdamW(qat_model.parameters(), lr=1e-3)
    qat_model.train()
    
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer_qat.zero_grad()
        _, loss = qat_model(x, y)
        loss.backward()
        optimizer_qat.step()
        if step > 200: # Same training duration as FP32
            break
            
    qat_ppl, qat_loss = calculate_perplexity(qat_model, val_loader, device)
    print(f"[Ternary QAT] PPL: {qat_ppl:.2f} | Loss: {qat_loss:.4f}")

    # Export the QAT model
    print(f"\nExporting QAT Model to firmware/src/model_weights.bin...")
    from train_qat import export_qat_model
    export_qat_model(qat_model, config, tokenizer, "firmware/src/model_weights.bin")
    
    # Generate some text for reference
    print("\n[PyTorch Reference Output]")
    qat_model.eval()
    context = torch.tensor([tokenizer.encode("I ")], dtype=torch.long).to(device)
    out_tokens = qat_model.generate(context, max_new_tokens=20, temperature=0.8)
    print("Output:", tokenizer.decode(out_tokens[0].tolist()))

if __name__ == "__main__":
    main()
