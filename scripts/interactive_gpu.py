import os
import sys
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import esp32_llm.model as model_mod
model_mod.TinyPoetConfig = model_mod.ESP32LLMConfig
model_mod.TinyPoet = model_mod.ESP32LLM

sys.modules['training'] = sys.modules['esp32_llm']
sys.modules['training.model'] = model_mod
sys.modules['tinypoet'] = sys.modules['esp32_llm']
sys.modules['tinypoet.model'] = model_mod

from esp32_llm.tokenizer import BPETokenizer, BASE_CHARS

class CharTokenizer:
    def __init__(self):
        self.base_chars = BASE_CHARS
        self.char_to_id = {ch: i for i, ch in enumerate(self.base_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.base_chars)}

    def encode(self, text):
        return [self.char_to_id.get(c.lower(), 0) for c in text if c.lower() in self.char_to_id]

    def decode(self, ids):
        return "".join([self.id_to_char.get(i, "") for i in ids])

from esp32_llm.model import ESP32LLM
from esp32_llm.quantization import replace_linear_with_ternary

VARIANTS = {
    "0": {
        "name": "Micro-LM-Pro 3.1M (Ternary QAT)",
        "ckpt": "models/Micro-LM-Pro/model.pth",
        "type": "bpe",
        "is_ternary": True
    },
    "1": {
        "name": "Micro-LM-Ultra 11.4M (Ternary QAT)",
        "ckpt": "models/Micro-LM-Ultra/model.pth",
        "type": "bpe",
        "is_ternary": True
    },
    "2": {
        "name": "Micro-LM-Base 181K (FP32)",
        "ckpt": "models/Micro-LM-Base/model.pth",
        "type": "bpe",
        "is_ternary": False
    },
    "3": {
        "name": "TinyPoet-FP32 164K (Char Level)",
        "ckpt": "models/TinyPoet-FP32/model.pth",
        "type": "char",
        "is_ternary": False
    },
    "4": {
        "name": "TinyPoet-INT4 164K (Char Level)",
        "ckpt": "models/TinyPoet-INT4/model.pth",
        "type": "char",
        "is_ternary": False
    },
    "5": {
        "name": "TinyPoet-Nano 50K (Ternary STE)",
        "ckpt": "models/TinyPoet-Nano/model.pth",
        "type": "char",
        "is_ternary": True
    },
    "6": {
        "name": "TinyPoet-Pico 50K (INT4 QAT)",
        "ckpt": "models/TinyPoet-Pico/model.pth",
        "type": "char",
        "is_ternary": False
    }
}

def load_variant(variant_key, device):
    var_info = VARIANTS[variant_key]
    ckpt_path = var_info["ckpt"]
    print(f"\n[Loading Variant {variant_key}] {var_info['name']}...")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file {ckpt_path} not found.")
        return None, None
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    
    if var_info["type"] == "bpe":
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        tokenizer.load(os.path.join("datasets", "bpe_tokenizer.json"))
    else:
        tokenizer = CharTokenizer()
        
    model = ESP32LLM(config)
    if var_info["is_ternary"]:
        model = replace_linear_with_ternary(model)
        
    state_dict = ckpt["model_state_dict"]
    
    # Convert old c_attn state dict keys to MQA (q_proj, k_proj, v_proj)
    new_sd = {}
    for k, v in state_dict.items():
        if ".c_attn.weight" in k:
            prefix = k.rsplit(".c_attn.weight", 1)[0]
            q, k_w, v_w = v.chunk(3, dim=0)
            new_sd[f"{prefix}.q_proj.weight"] = q
            new_sd[f"{prefix}.k_proj.weight"] = k_w
            new_sd[f"{prefix}.v_proj.weight"] = v_w
        elif ".c_attn.bias" in k:
            prefix = k.rsplit(".c_attn.bias", 1)[0]
            q, k_b, v_b = v.chunk(3, dim=0)
            new_sd[f"{prefix}.q_proj.bias"] = q
            new_sd[f"{prefix}.k_proj.bias"] = k_b
            new_sd[f"{prefix}.v_proj.bias"] = v_b
        elif ".wpe.weight" in k:
            continue # RoPE is used dynamically
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=False)
    model = model.to(device)
    model.eval()
    
    params = model.count_parameters()
    print(f"-> Model Loaded on {device.upper()}!")
    print(f"   Architecture: {config.n_layer}L / {config.n_head}H / {config.n_embd}D")
    print(f"   Context Length: {config.block_size} | Parameters: {params:,}")
    return model, tokenizer

def run_interactive():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("  ESP32-LLM GPU Interactive Testing Suite")
    print(f"  Device: {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print("=" * 65)
    
    print("\nAvailable Model Variants:")
    for k, v in VARIANTS.items():
        print(f"  [{k}] {v['name']}")
        
    current_key = "1"
    model, tokenizer = load_variant(current_key, device)
    
    temperature = 0.8
    top_k = 8
    max_tokens = 100
    
    print("\nCommands:")
    print("  /switch <1-6>  - Switch to a different model variant")
    print("  /temp <val>    - Set temperature (e.g., /temp 0.7)")
    print("  /topk <val>    - Set top_k (e.g., /topk 10)")
    print("  /len <val>     - Set max new tokens (e.g., /len 150)")
    print("  /exit          - Exit interactive session\n")
    
    while True:
        try:
            prompt = input(f"[{VARIANTS[current_key]['name'].split()[0]}] Enter prompt > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not prompt:
            continue
            
        if prompt == "/exit":
            break
            
        if prompt.startswith("/switch"):
            parts = prompt.split()
            if len(parts) > 1 and parts[1] in VARIANTS:
                current_key = parts[1]
                model, tokenizer = load_variant(current_key, device)
            else:
                print("Invalid variant key. Choose 1-6.")
            continue
            
        if prompt.startswith("/temp"):
            parts = prompt.split()
            if len(parts) > 1:
                temperature = float(parts[1])
                print(f"Temperature set to {temperature}")
            continue

        if prompt.startswith("/topk"):
            parts = prompt.split()
            if len(parts) > 1:
                top_k = int(parts[1])
                print(f"Top-K set to {top_k}")
            continue

        if prompt.startswith("/len"):
            parts = prompt.split()
            if len(parts) > 1:
                max_tokens = int(parts[1])
                print(f"Max new tokens set to {max_tokens}")
            continue

        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        start_t = time.time()
        with torch.no_grad():
            out_tokens = model.generate(
                x, 
                max_new_tokens=max_tokens, 
                temperature=temperature, 
                top_k=top_k
            )[0].tolist()
        elapsed = time.time() - start_t
        
        gen_tokens = out_tokens[len(input_ids):]
        generated_text = tokenizer.decode(gen_tokens)
        tps = len(gen_tokens) / elapsed if elapsed > 0 else 0
        
        print("-" * 50)
        print(f"Prompt: {prompt}")
        print(f"Generated Text:\n{prompt}{generated_text}")
        print("-" * 50)
        print(f"Metrics: {len(gen_tokens)} tokens generated in {elapsed:.3f}s ({tps:.2f} tokens/sec on GPU)\n")

if __name__ == "__main__":
    run_interactive()
