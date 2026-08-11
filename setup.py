from setuptools import setup, find_packages

setup(
    name="micro_lm",
    version="2.0.0",
    description="Micro-LM: Zero-Allocation On-Device LLM for ESP32",
    author="Micro-LM Contributors",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "tqdm",
        "datasets",
    ],
    python_requires=">=3.11",
)
