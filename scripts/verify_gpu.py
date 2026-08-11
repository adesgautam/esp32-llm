import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from micro_lm.model import ESP32LLM, ESP32LLMConfig
from micro_lm.tokenizer import BPETokenizer

def verify():
    device = "cpu"
    print("Loading PyTorch model...")
    ckpt_path = "checkpoints/esp32_llm_v2_bpe.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    config = ckpt['config']
    model = ESP32LLM(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("Loading BPE Tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load(ckpt['tokenizer_path'])

    prompts = ["love ", "my god ", "love is "]
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*50}")
        
        # Greedy search (argmax)
        prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            out_greedy = model.generate(prompt_tokens, max_new_tokens=100, temperature=1.0, top_k=1)[0].tolist()
        print("\n--- GPU OUTPUT (Greedy / Argmax) ---")
        print(tokenizer.decode(out_greedy))
        
        # Sampling (temp 0.8, top-k 8) - identical to ESP32 setup
        with torch.no_grad():
            out_sample = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8, top_k=8)[0].tolist()
        print("\n--- GPU OUTPUT (Sampling T=0.8, TopK=8) ---")
        print(tokenizer.decode(out_sample))

if __name__ == "__main__":
    verify()
