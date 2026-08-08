# ESP32LLM ESP32 On-Device LLM: End-to-End Architecture & Workflow

## Executive Summary & Hardware Constraints

**ESP32LLM** is a zero-allocation, memory-mapped C inference engine and transformer language model designed to run completely on-device on the **ESP32-D0WD-V3 (ESP32-WROOM-32)** microcontroller without external PSRAM.

### Core Hardware Specifications
- **MCU:** ESP32-D0WD-V3 (Dual-core Xtensa LX6 @ 240 MHz)
- **Total Physical SRAM:** 520 KB (System boot & FreeRTOS leave **~280 KB usable DRAM heap**)
- **Target DRAM Heap Allocation Limit:** **120 KB peak** (to guarantee zero OOM panics)
- **SPI Flash:** 4 MB Flash in DIO Mode @ 80 MHz
- **Partition Table:** 2.5 MB model XIP partition at offset `0x110000`

---

## 1. End-to-End Project Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Corpus    │ ──> │  BPE Tokenizer   │ ──> │ PyTorch Training │ ──> │ Binary Export    │
│  (10.48 MB text)│     │  (256 Vocab)     │     │ (2L/4H/80D Model)│     │ (.bin with Hdr)  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
                                                                                   │
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐              │
│ ESP32 Console   │ <── │ Hardware Sampling│ <── │ Zero-Alloc C     │ <────────────┘
│ (Serial COM3)   │     │ (Top-K / Temp)   │     │ Engine (XIP)     │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

### Step 1: Architectural Planning & Memory Bound Formulation
Before writing any code, we established strict bounds for memory allocation on the ESP32:

1. **Zero Dynamic Allocation (`malloc`) at Runtime:** Dynamic allocation leads to heap fragmentation and fatal OOM crashes on embedded systems.
2. **eXecute-In-Place (XIP) Model Mapping:** Model weights reside in SPI Flash mapped into memory space via the ESP32's MMU (`esp_partition_mmap`), consuming **0 bytes of SRAM for weight storage**.
3. **Internal SRAM Footprint Calculation:**
   $$\text{SRAM Footprint} = \text{KV Cache} + \text{Layer Activations} + \text{Call Stack}$$
   $$\text{KV Cache} = 2 \times N_{layer} \times N_{head} \times N_{ctx} \times d_{head} \times 4\text{ bytes}$$
   $$\text{For } 2\text{L}/4\text{H}/80\text{D } (d_{head}=20), N_{ctx}=64: \quad 2 \times 2 \times 4 \times 64 \times 20 \times 4 = 81.92\text{ KB}$$
   Adding activation buffer (~16 KB) and stack (~16 KB) gives a total peak SRAM allocation of **~113.92 KB**, comfortably under our 120 KB threshold.

---

### Step 2: Training Data Expansion (10.48 MB Corpus)
To prevent character-level repetition, we constructed a comprehensive public-domain poetry dataset (`data/raw/poetry_corpus.txt`):

- **Sources:** Project Gutenberg (Shakespeare Complete Works, Walt Whitman, Emily Dickinson, Edgar Allan Poe, John Keats, Percy Shelley, Lord Byron, William Wordsworth, William Blake, Robert Frost) + Karpathy's TinyShakespeare.
- **Corpus Size:** **10,988,705 characters (10.48 MB)**.
- **Normalization:** Cleaned unicode characters, standardized quotes/dashes, restricted to a core allowed character set.

---

### Step 3: Byte Pair Encoding (BPE) Tokenization
Character-level models require many steps to predict single words and suffer from high loss. We designed a custom BPE Tokenizer (`training/bpe_tokenizer.py`):

- **Base Vocabulary:** 44 individual characters (a-z, 0-9, space, newline, punctuation).
- **Subword Merges:** Learned **212 subword merge rules** from the 10.4MB corpus.
- **Final Vocabulary Size:** **256 tokens** (fitting nicely into `uint8_t` token representation).
- **Compression Ratio:** **1.98x** (10.48M chars compressed into 5.56M BPE tokens).
- **C Header Export:** Exported C decode table (`esp32/main/bpe_vocab.h`) containing string literals and merge lookup pairs.

---

### Step 4: Model Architecture & Retraining Pipeline
We built and trained the **ESP32LLM v2** architecture (`training/train_v2_bpe.py`):

- **Parameters:** 181,440 parameters (2 Layers, 4 Attention Heads, 80 Embedding Dim, Context Window 64).
- **Training Setup:** 150 Epochs on NVIDIA GPU over 5.5M BPE tokens (~193,500 steps).
- **Optimizer:** AdamW ($LR_{max} = 3 \times 10^{-4}$, Weight Decay = $0.05$).
- **Learning Rate Schedule:** Linear warmup (5% of steps) followed by Cosine Decay down to $1 \times 10^{-5}$.
- **Final Metrics:** 
  - **Validation Loss:** `2.7865`
  - **Bits Per Character (BPC):** `4.0200`
  - **Perplexity (PPL):** **`16.22 PPL`**

---

### Step 5: Weight Export & Binary Serialization
The PyTorch `state_dict` is exported into a unified packed binary (`esp32/main/model_weights.bin`):

- **36-byte Magic Header:** Contains magic `0x54504f45` ('TPOE'), version 2, $N_{layer}$, $N_{head}$, $N_{embd}$, $N_{ctx}$, and $V_{size}$.
- **Weight Skipping:** Crucially, the PyTorch `state_dict` contains an `attn.bias` lower-triangular causal mask buffer. The exporter skips this buffer so binary offsets strictly match the C engine expectations.
- **Binary Size:** **820.78 KB** (FP32 precision), which fits comfortably within the 2.5 MB model flash partition.

---

### Step 6: ESP32 Zero-Allocation C Engine & Hardware Sampling
The ESP32 runtime (`esp32/main/esp32_llm_engine.c` and `model_runner.c`) executes inference with zero heap allocations:

1. **Flash Mapping:** `esp_partition_mmap` maps `model_weights.bin` directly into Xtensa CPU address space.
2. **On-Device BPE Encoding:** Prompts entered by the user via the interactive serial console are encoded directly on the ESP32 using the embedded BPE merge rules (`bpe_merges`).
3. **On-Device Generation & Sampling:** Replaced deterministic greedy `argmax` with **Temperature (0.8) & Top-K (8) Sampling**:
   - Uses a hardware-seeded `xorshift32` PRNG (`esp_timer_get_time()`).
   - Scales logits by temperature, sorts top-8 choices, computes softmax, and samples probabilistically.
4. **On-Device BPE Decoding:** As tokens are generated, the token string (`bpe_vocab[token_id]`) is printed directly to `stdout`.

---

### Step 7: Flashing & Verification
We developed an automated single-command flash script (`esp32/flash_all.py`):
- Flashes **bootloader** (`0x1000`), **partition table** (`0x8000`), **firmware** (`0x10000`), and **model weights** (`0x110000`) using `esptool`.
- Verification verified that output on ESP32 matches the PyTorch GPU reference model output structure.
