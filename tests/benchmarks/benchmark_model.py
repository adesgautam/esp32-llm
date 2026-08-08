import torch
import time
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tinypoet.model import TinyPoet, TinyPoetConfig
from tinypoet.tokenizer import BPETokenizer

def benchmark():
    device = "cpu"
    ckpt_path = "checkpoints/tinypoet_v2_bpe.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    print("Loading model for CPU benchmarking...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    model = TinyPoet(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load(ckpt['tokenizer_path'])

    prompt = "the "
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    print("Warming up...")
    with torch.no_grad():
        model.generate(prompt_tokens, max_new_tokens=10, temperature=0.8, top_k=8)

    print("Benchmarking generation speed...")
    max_new_tokens = 100
    start_time = time.time()
    with torch.no_grad():
        model.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=0.8, top_k=8)
    end_time = time.time()

    elapsed = end_time - start_time
    tok_per_sec = max_new_tokens / elapsed

    report = {
        "model": "TinyPoet V2 (Lyrics Pivot)",
        "parameters": model.count_parameters(),
        "host_cpu_inference_seconds": round(elapsed, 4),
        "host_cpu_tokens_per_sec": round(tok_per_sec, 2),
        "validation_loss": ckpt['val_loss'],
        "perplexity": ckpt['val_ppl']
    }

    os.makedirs("tests/benchmarks", exist_ok=True)
    out_file = "tests/benchmarks/host_benchmarks.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nBenchmarking Complete!")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {out_file}")

if __name__ == "__main__":
    benchmark()
