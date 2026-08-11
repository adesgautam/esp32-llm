import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from micro_lm.tokenizer import BPETokenizer
from micro_lm.model import ESP32LLM, ESP32LLMConfig
from micro_lm.quantization import replace_linear_with_ternary

def generate():
    device = "cpu"
    print("--- PyTorch QAT Inference (Option C) ---")
    
    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.load(os.path.join("datasets", "bpe_tokenizer.json"))
    
    config = ESP32LLMConfig.option_c()
    model = ESP32LLM(config)
    model = replace_linear_with_ternary(model)
    
    # Load the QAT checkpoint
    checkpoint_path = "checkpoints/esp32_llm_qat_option_b.pth"
    # Actually wait, benchmark_qat.py didn't save a checkpoint! It only exported the .bin!
    # Let me just run generation directly from benchmark_qat if I can, but the script exited.
    pass

if __name__ == "__main__":
    generate()
