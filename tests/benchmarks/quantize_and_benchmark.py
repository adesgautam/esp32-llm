import sys
import os
import torch
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from esp32_llm.model import ESP32LLMConfig, ESP32LLM
from esp32_llm.quantization import apply_ptq
from esp32_llm.tokenizer import BPETokenizer
from esp32_llm.train import BPEDataset
from torch.utils.data import DataLoader

def evaluate_loss(model, val_loader, device="cpu", eval_iters=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (X, Y) in enumerate(val_loader):
            if i >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def benchmark_ptq():
    print("="*60)
    print(" PTQ Benchmarking: Perplexity Degradation")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the Phase 3 Baseline Model
    ckpt_path = "checkpoints/esp32_llm_v2_bpe.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Please train the Phase 3 model first.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # We must construct the model exactly as it was during Phase 3 training.
    # Phase 3 Config: block_size=64, n_layer=2, n_head=4, n_kv_head=4, n_embd=80, vocab_size=256
    config = ESP32LLMConfig(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=4, 
        n_embd=80
    )
    
    baseline_model = ESP32LLM(config).to(device)
    state_dict = checkpoint["model_state_dict"]
    
    # Convert old c_attn weights to q_proj, k_proj, v_proj
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'c_attn.weight' in k:
            q, k_w, v_w = v.chunk(3, dim=0)
            prefix = k.replace('c_attn.weight', '')
            new_state_dict[prefix + 'q_proj.weight'] = q
            new_state_dict[prefix + 'k_proj.weight'] = k_w
            new_state_dict[prefix + 'v_proj.weight'] = v_w
        elif 'c_attn.bias' in k:
            q, k_b, v_b = v.chunk(3, dim=0)
            prefix = k.replace('c_attn.bias', '')
            new_state_dict[prefix + 'q_proj.bias'] = q
            new_state_dict[prefix + 'k_proj.bias'] = k_b
            new_state_dict[prefix + 'v_proj.bias'] = v_b
        else:
            new_state_dict[k] = v
            
    baseline_model.load_state_dict(new_state_dict)
    
    # 2. Load dataset directly
    corpus_path = "datasets/raw/lyrics_corpus.txt"
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found.")
        return
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()
        
    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.load("datasets/bpe_tokenizer.json")
    tokens = tokenizer.encode(corpus)
    data = torch.tensor(tokens, dtype=torch.long)
    
    n_val = int(len(data) * 0.05)
    val_data = data[-n_val:]
    
    val_ds = BPEDataset(val_data, config.block_size)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Baseline FP32
    fp32_loss = evaluate_loss(baseline_model, val_loader, device)
    print(f"FP32 Baseline   - Loss: {fp32_loss:.4f} | PPL: {torch.exp(torch.tensor(fp32_loss)):.2f}")
    
    # INT8
    int8_model = copy.deepcopy(baseline_model)
    apply_ptq(int8_model, "int8")
    int8_loss = evaluate_loss(int8_model, val_loader, device)
    print(f"INT8 PTQ        - Loss: {int8_loss:.4f} | PPL: {torch.exp(torch.tensor(int8_loss)):.2f}")
    
    # INT4
    int4_model = copy.deepcopy(baseline_model)
    apply_ptq(int4_model, "int4")
    int4_loss = evaluate_loss(int4_model, val_loader, device)
    print(f"INT4 PTQ        - Loss: {int4_loss:.4f} | PPL: {torch.exp(torch.tensor(int4_loss)):.2f}")
    
    # Ternary
    ternary_model = copy.deepcopy(baseline_model)
    apply_ptq(ternary_model, "ternary")
    ternary_loss = evaluate_loss(ternary_model, val_loader, device)
    print(f"Ternary PTQ     - Loss: {ternary_loss:.4f} | PPL: {torch.exp(torch.tensor(ternary_loss)):.2f}")

def calculate_footprint(config, name):
    print(f"\n--- {name} Footprint ---")
    model = ESP32LLM(config)
    total_params = model.count_parameters()
    
    fp32_flash = (total_params * 4) / (1024*1024)
    ternary_flash = (total_params * 0.25) / (1024*1024) # 4 weights per byte
    
    # KV Cache Size = 2 * n_layers * n_kv_head * max_seq_len * head_dim
    head_dim = config.n_embd // config.n_head
    kv_elements = 2 * config.n_layer * config.n_kv_head * config.block_size * head_dim
    
    fp32_sram = (kv_elements * 4) / 1024
    int4_sram = (kv_elements * 0.5) / 1024
    
    print(f"Parameters:      {total_params/1e6:.2f} Million")
    print(f"Flash (FP32):    {fp32_flash:.2f} MB")
    print(f"Flash (Ternary): {ternary_flash:.2f} MB")
    print(f"KV SRAM (FP32):  {fp32_sram:.2f} KB")
    print(f"KV SRAM (INT4):  {int4_sram:.2f} KB")

def benchmark_architectures():
    print("\n" + "="*60)
    print(" Architecture Target Benchmarking")
    print("="*60)
    
    calculate_footprint(ESP32LLMConfig.option_a(), "Option A (Max Context)")
    calculate_footprint(ESP32LLMConfig.option_b(), "Option B (Balanced)")

if __name__ == "__main__":
    benchmark_ptq()
    benchmark_architectures()
