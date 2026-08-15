# ESP32-LLM Emulator Setup Guide (QEMU)

While deploying to physical hardware is the primary goal of this project, you can rapidly test firmware changes, model binary (`.bin`) structures, and text generation logic directly on your computer using QEMU configured for ESP32.

This allows you to verify that the ternary 1.58-bit C engine and memory-mapped model binaries function exactly as expected before flashing to a real device.

## 1. Download QEMU for ESP32

Espressif maintains a custom fork of QEMU specifically designed for Xtensa architectures (ESP32, ESP32-S3, etc.).

1. Navigate to the [Espressif QEMU Releases Page](https://github.com/espressif/qemu/releases).
2. Download the latest release for your operating system (e.g., `qemu-xtensa-softmmu-esp_develop_*_x86_64-w64-mingw32.tar.xz` for Windows).
3. Extract the archive. On Windows, you can use `tar` in PowerShell:
   ```powershell
   tar -xf qemu-xtensa-softmmu-esp_develop_9.2.2-20260417-x86_64-w64-mingw32.tar.xz
   ```
4. Note the path to the `qemu-system-xtensa` binary located in the extracted `qemu\bin` directory.

## 2. Compile the Firmware and Export the Model

Make sure you have compiled the firmware for the **Standard ESP32** target via PlatformIO:
```bash
pio run -e esp32dev
```
This generates the bootloader, partition table, and the main firmware binary inside `firmware/.pio/build/esp32dev/`.

Next, ensure you have exported your `.pth` model to a `.bin` format using the export script:
```bash
python scripts/export_checkpoint.py --ckpt checkpoints/esp32_llm_qat_micro_lm_pro.pth --config micro_lm_pro --format ternary
```

## 3. Merge and Pad the Flash Image

QEMU requires a single, monolithic `.bin` file that represents the entire SPI flash memory of the ESP32. This includes the bootloader, partitions, firmware, and the model itself mapped to `0x290000`.

Additionally, QEMU strictly requires the flash image to be an exact standard size (e.g., 2MB, 4MB, 8MB, or 16MB). For the Standard ESP32, we will create a 4MB image.

### Step 3a: Merge the Binaries
Use `esptool` to merge the components together:
```bash
python -m esptool --chip esp32 merge_bin -o flash_image.bin \
    0x1000 firmware/.pio/build/esp32dev/bootloader.bin \
    0x8000 firmware/.pio/build/esp32dev/partitions.bin \
    0x10000 firmware/.pio/build/esp32dev/firmware.bin \
    0x00290000 models/Micro-LM-Pro/model.bin
```

### Step 3b: Pad to 4MB
Pad the resulting `flash_image.bin` with `0xFF` bytes to exactly 4MB (4,194,304 bytes). You can use this quick Python script:
```python
import os
# Pad to 4MB (4 * 1024 * 1024)
f = open('flash_image.bin', 'ab')
f.write(b'\xff' * (4 * 1024 * 1024 - os.path.getsize('flash_image.bin')))
f.close()
```

## 4. Run the Emulator

Once your 4MB `flash_image.bin` is ready, run QEMU. 
*(Replace the path to `qemu-system-xtensa.exe` with your actual path)*:

```bash
qemu\bin\qemu-system-xtensa.exe -nographic -machine esp32 -drive file=flash_image.bin,if=mtd,format=raw
```

### What to Expect
1. QEMU will boot the ESP32 bootloader and load the partitions.
2. The firmware will initialize, search for the `model` partition, and map it into the MMU using XIP.
3. You will see the standard ESP32-LLM prompt:
   ```text
   ESP32-LLM Starting...
   Searching for 'model' partition in SPI Flash...
   Found partition 'model' at offset 0x00290000, size 1507328 bytes
   Mapping model partition into MMU memory space (XIP)...
   ...
   Ready! Type a prompt and press Enter:
   ```
4. Type your prompt directly into the terminal and hit Enter to watch the model generate text in real-time within the emulator!

> [!NOTE]  
> The Tokens Per Second (TPS) reported by the firmware while running in QEMU does **not** reflect real hardware performance. QEMU is highly dependent on your host CPU speed. Always rely on a physical ESP32 for true TPS metrics.
