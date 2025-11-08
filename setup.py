"""
Adaptive Sparse Training - Setup Script

Developed by Oluwafemi Idiakhoa
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive-sparse-training",
    version="1.0.1",
    author="Oluwafemi Idiakhoa",
    author_email="",
    description="Energy-efficient deep learning with adaptive sample selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oluwafemidiakhoa/adaptive-sparse-training",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
    },
    keywords=[
        "deep learning",
        "machine learning",
        "energy efficient",
        "green ai",
        "adaptive training",
        "sparse training",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/oluwafemidiakhoa/adaptive-sparse-training/issues",
        "Source": "https://github.com/oluwafemidiakhoa/adaptive-sparse-training",
        "Documentation": "https://github.com/oluwafemidiakhoa/adaptive-sparse-training/blob/main/README.md",
    },
)
