import sys
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

# Check for minimum Python version if necessary
if sys.version_info < (3, 7):
    sys.exit("Sorry, Python >= 3.7 is required for RTorch.")

# Read README for long description
try:
    with open("../../README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "RTorch: A PyTorch-like library in Rust. (README not found)"


setup(
    name="rtorch",
    version="0.1.0", # Should match Cargo.toml or be managed separately
    author="Your Name", # Replace with your name/organisation
    author_email="you@example.com", # Replace with your email
    description="A simplified PyTorch-like library implemented in Rust",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brandonyee-cs/rtorch", # Optional: Link to your repo
    license="MIT",
    # Rust Extension Configuration
    rust_extensions=[
        RustExtension(
            # Target for the extension module (`rtorch.rtorch_lib`)
            # The name before the dot must match the python package directory name.
            # The name after the dot corresponds to the `lib.name` in Cargo.toml.
            "rtorch.rtorch_lib", # <package_name>.<rust_lib_name>
            # Path to the Cargo.toml file, relative to setup.py
            path="../../Cargo.toml",
            binding=Binding.PyO3, # Use PyO3 binding mode
            # Optional: Enable specific Cargo features
            # features=["rand"],
            debug=False, # Set to True for debug builds
        )
    ],
    # Define the Python package structure
    packages=find_packages(), # Automatically find packages (like 'rtorch' directory)
    # Specify requirements for running setup.py (setuptools-rust)
    setup_requires=["setuptools>=18.0", "setuptools-rust>=0.10.2"],
    # Specify runtime dependencies (numpy is needed for PyO3 ndarray integration)
    install_requires=["numpy>=1.16"],
    # zip_safe=False is often recommended for extensions
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha", # Adjust as appropriate
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
)