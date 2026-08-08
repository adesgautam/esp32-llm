from setuptools import setup, find_packages

setup(
    name="esp32_llm",
    version="2.0.0",
    description="ESP32LLM: Zero-Allocation On-Device LLM for ESP32",
    author="ESP32LLM Contributors",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "tqdm",
        "datasets",
    ],
    python_requires=">=3.11",
)
