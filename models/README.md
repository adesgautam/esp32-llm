# ESP32-LLM Model Zoo

This directory contains all the pre-trained checkpoints and exported binaries for the various ESP32-LLM variants. The models are named intuitively based on their size and precision.

| Friendly Name | Technical Name | Architecture | Parameters | Precision | Tokenizer | Target Hardware (RAM) | PPL | Hardware TPS | Emulator TPS |
|---------------|----------------|--------------|------------|-----------|-----------|-----------------------|-----|--------------|--------------|
| **Micro-LM-Pro** | Option C (Ternary QAT) | 4L / 4H / 256D (MQA) | 3.1M | 1.58-bit (Ternary) | BPE | Standard ESP32 (520KB SRAM) | ~15.5 | ~1.5 TPS | 7.43 TPS |
| **Micro-LM-Ultra** | Option B (Ternary QAT) | 4L / 8H / 512D (MQA) | 11.4M | 1.58-bit (Ternary) | BPE | ESP32-S3 (8MB+ Flash/PSRAM) | ~18.2 | N/A (>4MB Flash) | 3.99 TPS |
| **Micro-LM-Base** | V2 Baseline FP32 | 4L / 4H / 64D | 181K | 32-bit (FP32) | BPE | Standard ESP32 (520KB SRAM) | ~220.5 | ~8.7 TPS | 27.62 TPS |
| **Micro-LM-Pico** | Pico (Ternary QAT) | 4L / 4H / 64D (MQA) | 207K | 1.58-bit (Ternary) | BPE | Standard ESP32 (<120KB SRAM) | 234.50 | ~15.8 TPS | 14.03 TPS |


## Directory Structure

Inside each model's directory, you will find:
- `model.pth`: The PyTorch checkpoint containing the state dict and configuration used for training/inference in Python.
- `model.bin`: (If generated) The exported C-compatible flat binary containing the model configuration and weights ready to be flashed to the ESP32 partition.

## How to Flash

You can use the universal flashing script in the root directory to flash any of these models. 

For example, to flash **Micro-LM-Pro** (requires ESP32-S3):
```bash
.venv\Scripts\python scripts/flash_firmware.py --port COM3 --model models/Micro-LM-Pro/model.bin
```

To flash **Micro-LM-Pico** (perfect for Standard ESP32):
```bash
.venv\Scripts\python scripts/flash_firmware.py --port COM3 --model models/Micro-LM-Pico/model.bin
```
