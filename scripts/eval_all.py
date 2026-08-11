import os
import sys
import glob
import torch
import numpy as np
import argparse
from bin_eval import evaluate_bin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from micro_lm.model import ESP32LLM, ESP32LLMConfig
from micro_lm.quantization import replace_linear_with_ternary

def eval_pth(ckpt_path, is_ternary=False, tokenizer_type="bpe"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "config" in ckpt and isinstance(ckpt["config"], ESP32LLMConfig):
        config = ckpt["config"]
    else:
        if "Pico" in ckpt_path:
            config = ESP32LLMConfig.option_pico()
        elif "Pro" in ckpt_path:
            config = ESP32LLMConfig.option_c()
        elif "Ultra" in ckpt_path:
            config = ESP32LLMConfig.option_b()
        elif "Base" in ckpt_path:
            # Reconstruct the original Option C dimensions for V2 Base
            config = ESP32LLMConfig(vocab_size=256, block_size=128, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)
        else:
            config = ESP32LLMConfig.option_pico()
            
    if not hasattr(config, 'n_kv_head'):
        config.n_kv_head = config.n_head

    model = ESP32LLM(config)
    if is_ternary:
        model = replace_linear_with_ternary(model)
        
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    from esp32_llm.tokenizer import BPETokenizer
    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.load("datasets/bpe_tokenizer.json")

    with open(os.path.join("datasets", "raw", "lyrics_corpus.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    n = len(tokens)
    val_data = tokens[int(n*0.9):]

    seq_len = config.block_size
    total_loss = 0.0
    iters = 0

    with torch.no_grad():
        for i in range(0, len(val_data) - seq_len, seq_len):
            x = torch.tensor(val_data[i:i+seq_len], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(val_data[i+1:i+1+seq_len], dtype=torch.long).unsqueeze(0)
            logits, loss = model(x, y)
            total_loss += loss.item()
            iters += 1
            if iters >= 5: # Only 5 batches to quickly verify parity!
                break

    avg_loss = total_loss / iters
    ppl = np.exp(avg_loss)
    return ppl

def main():
    models = {
        "Micro-LM-Pro": {"ternary": True},
        "Micro-LM-Ultra": {"ternary": True},
        "Micro-LM-Base": {"ternary": False},
        "Micro-LM-Pico": {"ternary": True},
    }

    print("Model | PTH PPL (5 batch) | BIN PPL (5 batch) | Exact Match?")
    print("-" * 65)

    for name, info in models.items():
        pth_path = os.path.join("models", name, "model.pth")
        bin_path = os.path.join("models", name, "model.bin")
        
        if not os.path.exists(pth_path) or not os.path.exists(bin_path):
            print(f"{name:15} | Missing PTH or BIN!")
            continue

        pth_ppl = eval_pth(pth_path, is_ternary=info["ternary"])
        
        # We need to hack evaluate_bin to return ppl instead of printing, or just rewrite a small snippet
        import subprocess
        result = subprocess.run([sys.executable, "scripts/bin_eval.py", "--bin", bin_path], capture_output=True, text=True)
        bin_ppl = None
        for line in result.stdout.split('\n'):
            if "BIN Model PPL:" in line:
                bin_ppl = float(line.split(":")[1].strip())
                
        if bin_ppl is None:
            print(f"{name:15} | {pth_ppl:17.4f} | Error             | N/A")
        else:
            match = "Yes" if abs(pth_ppl - bin_ppl) < 0.01 else "No"
            print(f"{name:15} | {pth_ppl:17.4f} | {bin_ppl:17.4f} | {match}")

if __name__ == "__main__":
    main()
