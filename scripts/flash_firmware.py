import os
import sys
import subprocess
import argparse

def flash_all(port: str, baud: int, model_path: str):
    binaries = [
        ("0x1000", "firmware/.pio/build/esp32dev/bootloader.bin"),
        ("0x8000", "firmware/.pio/build/esp32dev/partitions.bin"),
        ("0x10000", "firmware/.pio/build/esp32dev/firmware.bin"),
        ("0x290000", model_path)
    ]
    
    # Check if all files exist
    for _, path in binaries:
        if not os.path.exists(path):
            if "model" in path:
                print(f"Error: Model binary '{path}' not found! Please check the --model argument.")
            else:
                print(f"Error: {path} not found! Have you run PlatformIO build yet?")
            return False

    # Ensure esptool is installed
    try:
        subprocess.run([sys.executable, "-m", "esptool", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        esptool_cmd = [sys.executable, "-m", "esptool"]
    except Exception:
        # Check if installed in .venv
        venv_python = os.path.join(".venv", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            try:
                subprocess.run([venv_python, "-m", "esptool", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                esptool_cmd = [venv_python, "-m", "esptool"]
            except Exception:
                print("Installing 'esptool' in environment...")
                subprocess.run([sys.executable, "-m", "pip", "install", "esptool"], check=True)
                esptool_cmd = [sys.executable, "-m", "esptool"]
        else:
            print("Installing 'esptool' in environment...")
            subprocess.run([sys.executable, "-m", "pip", "install", "esptool"], check=True)
            esptool_cmd = [sys.executable, "-m", "esptool"]

    print(f"\n=======================================================")
    print(f" Flashing Full Firmware & Model to ESP32 ({port})")
    print(f" Model: {model_path}")
    print(f"=======================================================")
    print("\n>>> HOLD THE 'BOOT' BUTTON ON YOUR ESP32 NOW! <<<")
    print("Connecting in 3 seconds...\n")
    import time
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("\nAttempting connection...\n")

    cmd = esptool_cmd + [
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
    parser = argparse.ArgumentParser(description="Flash Micro-LM Firmware and Model")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=460800, help="Baud rate for flashing")
    parser.add_argument("--model", type=str, default="models/Micro-LM-Pro/model.bin", help="Path to the model .bin file")
    
    args = parser.parse_args()
    flash_all(args.port, args.baud, args.model)
