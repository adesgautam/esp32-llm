import os
import sys
import struct
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from esp32_llm.model import ESP32LLM, ESP32LLMConfig
from esp32_llm.tokenizer import BPETokenizer

def unpack_ternary_weights(packed, num_weights):
    mapping = {0: -1.0, 1: 0.0, 2: 1.0, 3: 0.0}
    out = np.zeros(num_weights, dtype=np.float32)
    bytes_data = np.frombuffer(packed, dtype=np.uint8)
    
    idx = 0
    for b in bytes_data:
        if idx < num_weights: out[idx] = mapping[b & 0x03]; idx += 1
        if idx < num_weights: out[idx] = mapping[(b >> 2) & 0x03]; idx += 1
        if idx < num_weights: out[idx] = mapping[(b >> 4) & 0x03]; idx += 1
        if idx < num_weights: out[idx] = mapping[(b >> 6) & 0x03]; idx += 1
    return out

def unpack_int4_weights(packed, num_weights):
    out = np.zeros(num_weights, dtype=np.float32)
    bytes_data = np.frombuffer(packed, dtype=np.uint8)
    
    idx = 0
    for b in bytes_data:
        if idx < num_weights: out[idx] = (b & 0x0F) - 8; idx += 1
        if idx < num_weights: out[idx] = ((b >> 4) & 0x0F) - 8; idx += 1
    return out

def evaluate_bin(bin_path, tokenizer_type="bpe"):
    with open(bin_path, "rb") as f:
        # Read magic
        magic_bytes = f.read(4)
        magic = struct.unpack("<I", magic_bytes)[0]
        
        fmt = "unknown"
        if magic == 0x54504f45:
            fmt = "ternary"
        elif magic == 0x54504934:
            fmt = "int4"
        elif magic == 0x54504632:
            fmt = "fp32"
        else:
            print(f"Invalid magic number! {hex(magic)}")
            return
            
        header_len = 24 if fmt in ["fp32", "int4"] else 28
        header_data = f.read(header_len)
        
        if header_len == 24:
            version, n_layer, n_head, n_embd, block_size, vocab_size = struct.unpack("<6I", header_data)
            n_kv_head = n_head
        else:
            version, n_layer, n_head, n_kv_head, n_embd, block_size, vocab_size = struct.unpack("<7I", header_data)
            
        print(f"Loaded Header: format={fmt} ver={version} layers={n_layer} head={n_head} kv_head={n_kv_head} embd={n_embd}")
        
        config = ESP32LLMConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            n_embd=n_embd,
            bias=True
        )
        model = ESP32LLM(config)
        state_dict = {}
        
        def read_fp32(shape):
            numel = np.prod(shape)
            data = f.read(int(numel * 4))
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
            return torch.tensor(arr.copy())
            
        def read_layer(out_features, in_features):
            numel = out_features * in_features
            if fmt == "ternary":
                scale = read_fp32((1,)).item()
                packed_len = (numel + 3) // 4
                packed = f.read(packed_len)
                weight = unpack_ternary_weights(packed, numel).reshape(out_features, in_features)
                weight = torch.tensor(weight.copy()) * scale
                bias = read_fp32((out_features,))
                return weight, bias
            elif fmt == "int4":
                scales = read_fp32((out_features,))
                packed_len = (numel + 1) // 2
                packed = f.read(packed_len)
                weight = unpack_int4_weights(packed, numel).reshape(out_features, in_features)
                weight = torch.tensor(weight.copy()) * scales.unsqueeze(1)
                bias = read_fp32((out_features,))
                return weight, bias
            else:
                weight = read_fp32((out_features, in_features))
                bias = read_fp32((out_features,))
                return weight, bias
                
        # 1. Embeddings
        state_dict["transformer.wte.weight"] = read_fp32((vocab_size, n_embd))
        
        head_size = n_embd // n_head
        # 2. Layers
        for l in range(n_layer):
            prefix = f"transformer.h.{l}"
            state_dict[f"{prefix}.ln_1.weight"] = read_fp32((n_embd,))
            state_dict[f"{prefix}.ln_1.bias"] = read_fp32((n_embd,))
            
            w, b = read_layer(n_head * head_size, n_embd)
            state_dict[f"{prefix}.attn.q_proj.weight"] = w; state_dict[f"{prefix}.attn.q_proj.bias"] = b
            
            w, b = read_layer(n_kv_head * head_size, n_embd)
            state_dict[f"{prefix}.attn.k_proj.weight"] = w; state_dict[f"{prefix}.attn.k_proj.bias"] = b
            
            w, b = read_layer(n_kv_head * head_size, n_embd)
            state_dict[f"{prefix}.attn.v_proj.weight"] = w; state_dict[f"{prefix}.attn.v_proj.bias"] = b
            
            w, b = read_layer(n_embd, n_embd)
            state_dict[f"{prefix}.attn.c_proj.weight"] = w; state_dict[f"{prefix}.attn.c_proj.bias"] = b
            
            state_dict[f"{prefix}.ln_2.weight"] = read_fp32((n_embd,))
            state_dict[f"{prefix}.ln_2.bias"] = read_fp32((n_embd,))
            
            w, b = read_layer(4 * n_embd, n_embd)
            state_dict[f"{prefix}.mlp.c_fc.weight"] = w; state_dict[f"{prefix}.mlp.c_fc.bias"] = b
            
            w, b = read_layer(n_embd, 4 * n_embd)
            state_dict[f"{prefix}.mlp.c_proj.weight"] = w; state_dict[f"{prefix}.mlp.c_proj.bias"] = b

        state_dict["transformer.ln_f.weight"] = read_fp32((n_embd,))
        state_dict["transformer.ln_f.bias"] = read_fp32((n_embd,))
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print("Model loaded successfully from .bin file.")
        
        if tokenizer_type == "bpe":
            tokenizer = BPETokenizer(vocab_size=256)
            tokenizer.load("datasets/bpe_tokenizer.json")
        else:
            from esp32_llm.tokenizer import CharTokenizer
            tokenizer = CharTokenizer()
            tokenizer.load("datasets/char_tokenizer.json")
        
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
                if iters >= 50:
                    break
        
        avg_loss = total_loss / iters
        ppl = np.exp(avg_loss)
        print(f"BIN Model PPL: {ppl:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"])
    args = parser.parse_args()
    evaluate_bin(args.bin, args.tokenizer)
