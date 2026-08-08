# ESP32LLM ESP32: Zero-Allocation On-Device LLM Engine

[![ESP32](https://img.shields.io/badge/Hardware-ESP32--D0WD--V3-orange.svg)](https://www.espressif.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/Framework-PlatformIO%20%2F%20ESP--IDF-green.svg)](https://platformio.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**ESP32LLM** is a complete, production-grade implementation of an on-device Transformer Language Model running on the **ESP32-D0WD-V3 (ESP32-WROOM-32)** microcontroller without requiring external PSRAM.

It features a custom zero-allocation C inference engine, Flash Memory-Mapping (XIP), Byte Pair Encoding (BPE) subword tokenization, and hardware-seeded Temperature + Top-K sampling.

---

## ⚡ Technical Highlights & Hardware Feasibility

| Parameter / Dimension | Specification | Architectural Design Rationale |
| :--- | :--- | :--- |
| **Microcontroller** | ESP32-D0WD-V3 (240 MHz Dual-Core) | Dual Xtensa LX6 CPU core @ 80 MHz SPI bus speed |
| **Model Parameters** | 181,440 Floating-Point Weights | 2 Layers / 4 Attention Heads / 80 Embedding Dim ($N_{ctx}=64$) |
| **Subword Vocabulary** | 256 Tokens (BPE) | Learned 212 subword merges (1.98x token compression) |
| **Peak SRAM Allocation** | **113.92 KB** ($\le 120\text{ KB}$ Heap Limit) | **Zero dynamic `malloc()` at runtime**; static ring KV-Cache |
| **Weight Storage** | 820.78 KB in 2.5MB Flash Partition | **Mapped directly into CPU MMU memory space (XIP)** |
| **Sampling Engine** | Temp = 0.8 / Top-K = 8 | Hardware-seeded `xorshift32` PRNG (`esp_timer_get_time()`) |
| **Perplexity (PPL)** | **16.22 PPL** | Trained on 10.48 MB public domain poetry corpus |

---

## 🛠️ Step-by-Step E2E Reproduction Guide

Follow these steps to train, export, and flash ESP32LLM onto your physical ESP32 board from scratch.

### Step 1: Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/esp32-llm.git
   cd esp32-llm
   ```

2. **Create & Activate Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 2: Download & Prepare the 10MB Training Corpus

Run the dataset acquisition script to download public domain poetry (Shakespeare, Whitman, Dickinson, Poe, Keats, Shelley, Byron, Wordsworth, Blake, Frost):
```bash
python data/prepare_expanded_data.py
```
*Output: `data/raw/poetry_corpus.txt` (10.48 MB, ~10.98 million characters).*

---

### Step 3: Train BPE Tokenizer & Retrain Model on GPU

Run the complete training pipeline. This script trains the 256-token BPE vocabulary, exports the C header `esp32/main/bpe_vocab.h`, trains the 181K-parameter Transformer on GPU for 150 epochs, and exports the binary weight file `esp32/main/model_weights.bin`:
```bash
python training/train_v2_bpe.py
```
*Outputs:*
- `data/bpe_tokenizer.json`: BPE vocabulary and merge rules.
- `esp32/main/bpe_vocab.h`: C decode array and merge pairs.
- `esp32/main/model_weights.bin`: FP32 serialized weight binary (820.78 KB).

---

### Step 4: Verify GPU Inference Output (Optional)

Run the verification script to check the PyTorch GPU model output with Top-K sampling:
```bash
python training/verify_gpu.py
```

---

### Step 5: Build ESP32 Firmware

Build the PlatformIO C/C++ firmware containing the zero-allocation inference engine:
```bash
platformio run
```
*Build artifact: `.pio/build/esp32dev/firmware.bin`.*

---

### Step 6: Flash Firmware & Model Weights to ESP32

Connect your ESP32 board via USB (default port `COM3`). Run the multi-partition automated flasher:
```bash
python esp32/flash_all.py
```
*This script erases and flashes:*
- `0x00001000`: Bootloader (`bootloader.bin`)
- `0x00008000`: Custom Partition Table (`partitions.bin`)
- `0x00010000`: ESP32 Application Firmware (`firmware.bin`)
- `0x00110000`: Model Weights Binary (`model_weights.bin`, 820.78 KB into 2.5 MB partition)

---

### Step 7: Interactive Hardware Serial Console

Start the bidirectional interactive monitor to prompt the on-device model live:
```bash
python esp32/monitor.py
```

Type a prompt (e.g. `love is` or `in the night`) and press **Enter** to watch the ESP32 generate verse in real time!

---

## 🧮 Core SRAM Budget Mathematics

To guarantee that ESP32LLM never triggers an Out-Of-Memory (OOM) heap panic on internal SRAM:

$$\text{KV Cache} = 2 \times N_{layer} \times N_{head} \times N_{ctx} \times d_{head} \times 4\text{ bytes}$$
$$\text{For } 2\text{L} / 4\text{H} / 80\text{D } (d_{head}=20), N_{ctx}=64: \quad 2 \times 2 \times 4 \times 64 \times 20 \times 4 = 81.92\text{ KB}$$

$$\text{Total SRAM Footprint} = \text{KV Cache (81.92 KB)} + \text{Activations (16.0 KB)} + \text{Stack (16.0 KB)} = \mathbf{113.92\text{ KB}}$$

$$\mathbf{113.92\text{ KB}} \le \mathbf{120.0\text{ KB DRAM Heap Safety Limit}}$$

---

## 📖 Detailed Documentation

- 📄 **Technical Architecture Markdown:** [`docs/architecture_and_workflow.md`](docs/architecture_and_workflow.md)
- 🎨 **Interactive Visual Flowchart HTML:** [`docs/architecture_and_workflow.html`](docs/architecture_and_workflow.html)
- 📊 **Empirical Research Progress:** [`project_progress.md`](project_progress.md)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
