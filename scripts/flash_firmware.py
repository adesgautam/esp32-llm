import os
import sys
import subprocess

def flash_all(port: str = "COM3", baud: int = 460800):
    binaries = [
        ("0x1000", ".pio/build/esp32dev/bootloader.bin"),
        ("0x8000", ".pio/build/esp32dev/partitions.bin"),
        ("0x10000", ".pio/build/esp32dev/firmware.bin"),
        ("0x110000", "esp32/main/model_weights.bin")
    ]
    
    # Check if all files exist
    for _, path in binaries:
        if not os.path.exists(path):
            print(f"Error: {path} not found! Have you run PlatformIO build yet?")
            return False

    print(f"\n=======================================================")
    print(f" Flashing Full Firmware & Model to ESP32 ({port})")
    print(f"=======================================================")
    print("Connecting to ESP32... (If it displays 'Connecting...', hold the BOOT button on your ESP32 board for 1 second)\n")

    cmd = [
        sys.executable, "-m", "esptool",
        "--port", port,
        "--baud", str(baud),
        "--before", "default-reset",
        "--after", "hard-reset",
        "write-flash"
    ]
    
    for offset, path in binaries:
        cmd.extend([offset, path])

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\nSuccessfully flashed all partitions to ESP32!")
        return True
    else:
        print(f"\nFlashing failed with exit code {res.returncode}")
        return False

if __name__ == "__main__":
    flash_all()
