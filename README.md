<div align="center">
  <h1>🚀 Micro-LM</h1>
  <p><strong>A ternary, zero-heap tiny language model that runs inside a $2 microcontroller</strong></p>
  
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Platform](https://img.shields.io/badge/Platform-ESP32-orange.svg)](https://www.espressif.com/en/products/socs/esp32)
  [![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
  [![Language](https://img.shields.io/badge/Language-C++-green.svg)]()
</div>

---

Micro-LM (formerly ESP32-LLM) is an incredibly optimized, ultra-low-memory language model architecture designed to run entirely inside the 520KB SRAM of a standard ESP32 microcontroller, without requiring external PSRAM. By utilizing **1.58-bit Ternary Quantization (QAT)** and custom RoPE (Rotary Position Embeddings), we achieved an end-to-end NLP pipeline operating on a dual-core 240MHz MCU.

## 🌟 Features
- **Zero-Heap Allocations:** The C inference engine operates entirely on statically allocated memory, preventing fragmentation and maximizing stability on edge devices.
- **Ternary Weights:** Model weights are quantized to `[-1, 0, 1]`, mapping 4 weights into a single byte!
- **Bit-Exact Parity:** Output from the C++ inference engine on the ESP32 matches the PyTorch FP32 model exactly.

## 🗄️ Model Zoo

We provide multiple variations of the architecture in the `models/` directory, spanning from 50K up to 11.4M parameters depending on your hardware limits (Standard ESP32 vs ESP32-S3). Check the [Models README](models/README.md) for full config details.

| Friendly Name | Params | Architecture | Precision | Target Device | PPL | Hardware TPS | Location |
|---------------|--------|--------------|-----------|---------------|-----|--------------|----------|
| **Micro-LM-Pro** | 3.1M | 4L / 4H / 256D | 1.58-bit | Standard ESP32 | **15.52** | ~1.5 TPS | `models/Micro-LM-Pro/model.bin` |
| **Micro-LM-Ultra**| 11.4M | 4L / 8H / 512D | 1.58-bit | ESP32-S3 (8MB+ Flash) | ~18.2 | N/A (>4MB Flash) | `models/Micro-LM-Ultra/model.bin` |
| **Micro-LM-Base** | 181K | 4L / 4H / 64D | FP32 | Standard ESP32 | ~220.5 | ~8.7 TPS | `models/Micro-LM-Base/model.bin` |
| **Micro-LM-Pico** | 207K | 4L / 4H / 64D | 1.58-bit | Standard ESP32 (<120KB RAM) | 234.50 | ~15.8 TPS | `models/Micro-LM-Pico/model.bin` |

## 🚀 Quickstart

### 1. Flash the ESP32

Ensure you have [PlatformIO](https://platformio.org/) installed and your ESP32 connected via USB (e.g. `COM3` or `/dev/ttyUSB0`).

```bash
# Activate virtual environment (Windows: .\.venv\Scripts\Activate.ps1 | Linux/Mac: source .venv/bin/activate)

# 1. Build the C++ firmware
cd firmware
pio run
cd ..

# 2. Flash C++ firmware + model weights (e.g. Micro-LM-Pico for Standard ESP32)
.\.venv\Scripts\python.exe scripts/flash_firmware.py --port COM3 --model models/Micro-LM-Pico/model.bin
```

### 2. Monitor Hardware Serial Output

To view real-time token generation, memory logs, and TPS (Tokens Per Second) stats from the ESP32:

**Option A (Using Python Serial Miniterm - Recommended):**
```bash
.\.venv\Scripts\python.exe -m serial.tools.miniterm COM3 115200
```
*(To exit miniterm, press `Ctrl + ]`)*

**Option B (Using PlatformIO Direct Binary):**
```bash
~/.platformio/penv/Scripts/pio.exe device monitor --port COM3 --baud 115200
```

*Note: Press the **RESET** (EN) button on your ESP32 board to restart execution and stream text output.*

### 3. Manual GPU / CPU Testing

Want to test PyTorch models natively on your PC before flashing to hardware? Use our interactive suite:

```bash
.\.venv\Scripts\python.exe scripts/interactive_gpu.py
```
*You can seamlessly switch between FP32 and Ternary models on-the-fly to compare output quality.*

## 📈 Performance & Milestones

The current champion is **Micro-LM-Pro (3.1M QAT)**. After diagnosing a GPU bandwidth bottleneck (resolved by accumulating gradients and shrinking batch sizes), the model was trained for 35 epochs. It broke past multiple plateaus, reaching a final, staggering Perplexity of **15.52** on our custom Lyrics dataset!

*For detailed technical context, architecture logs, and progress reports, check the `docs/` folder.*

## ⚖️ License
Released under the MIT License.
