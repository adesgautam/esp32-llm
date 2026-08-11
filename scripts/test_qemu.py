import sys
from pytest_embedded_qemu import Qemu
from pytest_embedded.dut import Dut
import tempfile
import time

def main():
    image_path = "flash_pro.bin"
    # pytest-embedded Qemu requires some params
    
    import pexpect
    
    print("Testing QEMU with pexpect directly using qemu-system-xtensa")
    # Actually wait, let's just see if qemu-system-xtensa is on the path if pytest_embedded installed it.
    
if __name__ == "__main__":
    main()
