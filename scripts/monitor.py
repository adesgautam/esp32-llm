import serial
import time
import sys

def monitor_esp32_continuous(port="COM3", baud=115200, reset_on_start=True):
    print(f"=======================================================")
    print(f"   PlatformIO-Style Continuous Serial Monitor ({port})  ")
    print(f"   Baud Rate: {baud} | Press Ctrl+C to stop monitoring")
    print(f"=======================================================\n", flush=True)

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        
        if reset_on_start:
            # Trigger RTS toggle reset to restart chip on monitor attach
            ser.dtr = False; ser.rts = True; time.sleep(0.1)
            ser.dtr = True; ser.rts = False; time.sleep(0.1)
            ser.dtr = False; time.sleep(0.2)

        import os
        is_windows = os.name == 'nt'
        if is_windows:
            import msvcrt

        while True:
            # Read from Serial and print to Terminal
            if ser.in_waiting:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                print(data, end='', flush=True)
            
            # Read from Terminal and write to Serial
            if is_windows and msvcrt.kbhit():
                char = msvcrt.getch()
                ser.write(char)
                # Note: main.cpp echoes characters back, so we don't print it here
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n=======================================================")
        print(" Serial monitor stopped by user (Ctrl+C). Port closed.")
        print("=======================================================")
    except Exception as e:
        print(f"\nSerial port error: {e}")
    finally:
        try:
            if 'ser' in locals() and ser.is_open:
                ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    port_arg = "COM3"
    baud_arg = 115200
    
    for arg in sys.argv[1:]:
        if arg.startswith("COM") or arg.startswith("/dev/"):
            port_arg = arg
        elif arg.isdigit() and int(arg) >= 1200:
            baud_arg = int(arg)

    monitor_esp32_continuous(port=port_arg, baud=baud_arg)
