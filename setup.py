from setuptools import setup, find_packages

setup(
    name="tinypoet",
    version="2.0.0",
    description="TinyPoet: Zero-Allocation On-Device LLM for ESP32",
    author="TinyPoet Contributors",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "tqdm",
        "datasets",
    ],
    python_requires=">=3.11",
)
