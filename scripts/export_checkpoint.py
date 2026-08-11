import torch
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from esp32_llm.model import ESP32LLM, ESP32LLMConfig
from esp32_llm.tokenizer import BPETokenizer
from esp32_llm.quantization import replace_linear_with_ternary
from scripts.train_qat import export_qat_model

def pack_int4_weights(weight_q):
    w = weight_q.cpu().numpy().astype(np.uint8)
    w = w & 0x0F
    packed = np.zeros((w.size + 1) // 2, dtype=np.uint8)
    for i in range(w.size):
        if i % 2 == 0:
            packed[i // 2] |= (w.flatten()[i] & 0x0F)
        else:
            packed[i // 2] |= (w.flatten()[i] & 0x0F) << 4
    return packed.tobytes()

def export_model(ckpt_path, bin_path, config_name="option_c", fmt="ternary"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "config" in ckpt and isinstance(ckpt["config"], ESP32LLMConfig):
        config = ckpt["config"]
    else:
        if hasattr(ESP32LLMConfig, config_name):
            config = getattr(ESP32LLMConfig, config_name)()
        else:
            config = ESP32LLMConfig.option_c()

    # Ensure config object has all required new fields if loading older checkpoint
    if not hasattr(config, 'n_kv_head'):
        config.n_kv_head = config.n_head

    model = ESP32LLM(config)
    
    if fmt == "ternary":
        model = replace_linear_with_ternary(model)
        
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.load("datasets/bpe_tokenizer.json")

    # Export Logic
    import struct
    f = open(bin_path, "wb")
    
    if fmt == "ternary":
        magic = 0x54504f45 # TPOE
        version = 3
    elif fmt == "int4":
        magic = 0x54504934 # TPI4
        version = 1
    else:
        magic = 0x54504632 # TPF2
        version = 2
        
    if fmt in ["fp32", "int4"]:
        # Write old-style 7-int header or new 8-int header?
        # Let's write the new 8-int header but keep the magic.
        # Actually our C engine checks magic, so we can just write 8-int header universally for FP32 and INT4 since we updated C engine!
        # Wait, for INT4 we set header_bytes=28 in C engine if magic is TPI4! So we MUST write 7-int header!
        if fmt == "int4" or fmt == "fp32":
            header = struct.pack("<7I", magic, version, config.n_layer, config.n_head, config.n_embd, config.block_size, config.vocab_size)
        else:
            header = struct.pack("<8I", magic, version, config.n_layer, config.n_head, config.n_kv_head, config.n_embd, config.block_size, config.vocab_size)
    else:
        header = struct.pack("<8I", magic, version, config.n_layer, config.n_head, config.n_kv_head, config.n_embd, config.block_size, config.vocab_size)
        
    f.write(header)
    total_bytes = len(header)
    
    state_dict = model.state_dict()
    
    def write_fp32(tensor):
        nonlocal total_bytes
        data = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
        f.write(data)
        total_bytes += len(data)
        
    def write_layer_weights(prefix):
        nonlocal total_bytes
        weight = state_dict[f"{prefix}.weight"]
        bias = state_dict.get(f"{prefix}.bias", None)
        if bias is None:
            bias = torch.zeros(weight.size(0))
            
        if fmt == "ternary":
            scale = weight.abs().mean().clamp(min=1e-8)
            weight_norm = weight / scale
            weight_q = torch.round(weight_norm).clamp(-1, 1)
            write_fp32(scale)
            from scripts.train_qat import pack_ternary_weights
            packed = pack_ternary_weights(weight_q)
            f.write(packed)
            total_bytes += len(packed)
            write_fp32(bias)
        elif fmt == "int4":
            # Quantize per row
            scales = weight.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / 7.0
            weight_q = torch.round(weight / scales).clamp(-8, 7)
            # Map -8..7 to 0..15
            weight_q = (weight_q + 8).to(torch.uint8)
            write_fp32(scales.flatten())
            packed = pack_int4_weights(weight_q)
            f.write(packed)
            total_bytes += len(packed)
            write_fp32(bias)
        else: # fp32
            write_fp32(weight)
            write_fp32(bias)
            
    # 1. Embeddings
    write_fp32(state_dict["transformer.wte.weight"])
    
    # 2. Layers
    for l in range(config.n_layer):
        prefix = f"transformer.h.{l}"
        write_fp32(state_dict[f"{prefix}.ln_1.weight"])
        write_fp32(state_dict[f"{prefix}.ln_1.bias"])
        
        write_layer_weights(f"{prefix}.attn.q_proj")
        write_layer_weights(f"{prefix}.attn.k_proj")
        write_layer_weights(f"{prefix}.attn.v_proj")
        write_layer_weights(f"{prefix}.attn.c_proj")
        
        write_fp32(state_dict[f"{prefix}.ln_2.weight"])
        write_fp32(state_dict[f"{prefix}.ln_2.bias"])
        
        write_layer_weights(f"{prefix}.mlp.c_fc")
        write_layer_weights(f"{prefix}.mlp.c_proj")
        
    # 3. Final LN
    write_fp32(state_dict["transformer.ln_f.weight"])
    write_fp32(state_dict["transformer.ln_f.bias"])
    
    f.close()
    
    print(f"Exported to {bin_path} ({total_bytes} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a PyTorch .pth checkpoint to a C-compatible .bin file")
    parser.add_argument("--ckpt", type=str, required=True, help="Input .pth checkpoint path")
    parser.add_argument("--out", type=str, required=True, help="Output .bin file path")
    parser.add_argument("--config", type=str, default="option_c", help="Config name (e.g. option_b, option_c)")
    parser.add_argument("--fmt", type=str, default="ternary", choices=["fp32", "int4", "ternary"], help="Export format")
    args = parser.parse_args()
    
    export_model(args.ckpt, args.out, config_name=args.config, fmt=args.fmt)
