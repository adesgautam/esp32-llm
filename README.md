<div align="center">
  <h1>🚀 Micro-LM</h1>
  <p><strong>A zero-allocation, 1.58-bit ternary language model running natively on microcontroller-class hardware ($2 ESP32)</strong></p>
  
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Platform](https://img.shields.io/badge/Platform-ESP32-orange.svg)](https://www.espressif.com/en/products/socs/esp32)
  [![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
  [![Language](https://img.shields.io/badge/Language-C%2B%2B%2FC99-green.svg)]()
</div>

---

> A reference implementation of an ultra-low-memory, zero-heap tiny language model architecture designed to run entirely inside the 520KB SRAM of a standard ESP32 microcontroller, without requiring external PSRAM.

Micro-LM combines **1.58-bit Ternary Quantization (QAT)** (after BitNet b1.58), custom RoPE (Rotary Position Embeddings), Multi-Query Attention (MQA), and a byte-level BPE tokenizer into a single lightweight kit. **The core contribution is full integration**: a complete PyTorch training → ternary weight export → 2-bit packing → C99 zero-allocation inference engine executing in-place (XIP) via flash MMU mapping.

**Quick Links:**
- 🖥️ **QEMU Local Emulator Guide**: [`docs/emulator.md`](docs/emulator.md)
- 🗄️ **Model Zoo Specifications**: [`models/README.md`](models/README.md)
- 📜 **Training Runs Log**: [`docs/training_runs.md`](docs/training_runs.md)

---

## ⚡ MCU Status & Real-Hardware Metrics

All models run offline on physical ESP32 microcontrollers using standard SPI Flash memory mapping (Execute-In-Place / XIP):

- **Micro-LM-Pico (207K)**: **~15.8 TPS** on bare-metal Hardware (14.03 TPS QEMU)
- **Micro-LM-Base (181K)**: **~8.7 TPS** on bare-metal Hardware (27.62 TPS QEMU)
- **Micro-LM-Pro (3.1M)**: **~1.5 TPS** on bare-metal Hardware (7.43 TPS QEMU)
- **Micro-LM-Ultra (11.4M)**: **3.99 TPS** in QEMU Emulator *(Requires 8MB+ Flash for physical hardware)*

---

## 📊 Dataset & Training Specifications

### 1. Corpus & Data Sources
Micro-LM models are trained on structured text datasets optimized for low-vocabulary tokenization and tiny-scale memory constraints:
- **Primary Source**: Cleaned TinyStories and text poetry corpus.
- **Preprocessing**: Normalized text stripped of invalid Unicode sequences, mapped into standard ASCII / UTF-8 byte ranges.

### 2. Data Splits & Tokenization
- **Split Ratio**: **90% Training / 10% Validation** split.
- **Tokenizer**: Custom Byte-Pair Encoding (BPE) tokenizer with a fixed **vocabulary size of 256 tokens** (`vocab_size=256`), matching byte boundaries for minimal lookup overhead.

### 3. Quantization & Training Pipeline
- **Method**: Quantization-Aware Training (QAT) using a Straight-Through Estimator (STE).
- **Weight Precision**: Ternary weights restricted to `{-α, 0, +α}` per tensor, allowing 4 weights to be packed into a single byte.
- **Weight Tying**: Embeddings (`wte`) and output head (`lm_head`) share identical FP32 weight pointers to guarantee zero redundant memory during C-engine inference.
- **Optimizer**: AdamW with Cosine Annealing learning rate schedule, gradient accumulation, and Automatic Mixed Precision (AMP).

---

## 🗄️ Model Zoo

We provide pre-trained PyTorch checkpoints (`.pth`) and flat C-binary exports (`.bin`) in the `models/` directory:

| Friendly Name | Technical Config | Parameters | Precision | Target Hardware | PPL | Hardware TPS | QEMU TPS | Location |
|---------------|------------------|------------|-----------|-----------------|-----|--------------|----------|----------|
| **Micro-LM-Pro** | 4L / 4H / 256D (MQA) | 3.1M | 1.58-bit (Ternary) | Standard ESP32 | **15.52** | ~1.5 TPS | 7.43 TPS | [`models/Micro-LM-Pro/`](models/Micro-LM-Pro/) |
| **Micro-LM-Ultra**| 4L / 8H / 512D (MQA) | 11.4M | 1.58-bit (Ternary) | ESP32-S3 (8MB+ Flash) | ~18.2 | N/A (>4MB Flash) | 3.99 TPS | [`models/Micro-LM-Ultra/`](models/Micro-LM-Ultra/) |
| **Micro-LM-Base** | 4L / 4H / 64D | 181K | 32-bit (FP32) | Standard ESP32 | ~220.5 | ~8.7 TPS | 27.62 TPS | [`models/Micro-LM-Base/`](models/Micro-LM-Base/) |
| **Micro-LM-Pico** | 4L / 4H / 64D (MQA) | 207K | 1.58-bit (Ternary) | Standard ESP32 (<120KB RAM) | 234.50 | ~15.8 TPS | 14.03 TPS | [`models/Micro-LM-Pico/`](models/Micro-LM-Pico/) |

---

## ⚙️ Key Architectural Features

- **Zero-Heap Allocations**: The C99 inference engine operates entirely on statically allocated memory buffers (`micro_lm_engine.c`), eliminating memory fragmentation risks.
- **Execute-In-Place (XIP)**: Flash partition is mapped directly to MMU memory space via `esp_partition_mmap`, running models without loading weights into RAM.
- **Ternary Bit Packing**: 1.58-bit ternary weights mapped to 2 bits per weight (4 weights per byte), drastically reducing flash footprint.

---

## 🚀 Quickstart

### 1. Flash the Firmware & Model
Ensure you have [PlatformIO](https://platformio.org/) installed and your ESP32 connected via USB:

```bash
# 1. Build C++ firmware
cd firmware
pio run
cd ..

# 2. Flash firmware and model binary (e.g. Micro-LM-Pico)
.venv\Scripts\python scripts/flash_firmware.py --port COM3 --model models/Micro-LM-Pico/model.bin
```

### 2. Monitor Hardware Output
To stream real-time generation and TPS stats from your physical ESP32:
```bash
.venv\Scripts\python -m serial.tools.miniterm COM3 115200
```
*(Press `Ctrl + ]` to exit).*

### 3. Local Evaluation & Emulation
- **Evaluate All Models**: `.venv\Scripts\python scripts/eval_all.py`
- **Run Local QEMU Simulator**: Follow instructions in [`docs/emulator.md`](docs/emulator.md).
- **Interactive PyTorch Testing**: `.venv\Scripts\python scripts/interactive_gpu.py`

---

## 📚 Documentation Index

Here is a list of detailed documentation guides available in this repository:

- 📄 [`docs/emulator.md`](docs/emulator.md): Comprehensive guide on setting up QEMU for ESP32 locally, merging partitions, and testing binaries automatically without physical hardware.
- 🗄️ [`models/README.md`](models/README.md): Detailed specs, directory layout, and binary flashing instructions for all model variants.
- 📊 [`docs/training_runs.md`](docs/training_runs.md): Training logs, validation loss history, and BPC/PPL progression records.

---

## ⚖️ License
Released under the [MIT License](LICENSE).
