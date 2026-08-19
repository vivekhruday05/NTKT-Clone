from setuptools import setup, find_packages

setup(
    name="ntkt",
    version="0.1.0",
    description="Next Token Knowledge Tracing (NTKT): Exploiting Pretrained LLM Representations to Decode Student Behaviour",
    author="Reproduction Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "baselines": ["sentence-transformers>=2.5.0"],
        "dev": ["pytest>=7.0.0", "matplotlib>=3.7.0"],
        "quantization": ["bitsandbytes>=0.42.0", "peft>=0.10.0"],
    },
)
