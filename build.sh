#!/bin/bash

# Build script for RTorch - PyTorch-like library in Rust
set -e  # Exit on error

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Please install Rust first: https://rustup.rs/"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3."
    exit 1
fi

# Ensure virtual environment
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Build the Rust library with PyO3 bindings
echo "Building RTorch..."
cd bindings/python
pip install -e .

echo ""
echo "RTorch has been built successfully!"
echo ""
echo "To use RTorch, activate the virtual environment with:"
echo "  source venv/bin/activate"
echo ""
echo "You can then run the example:"
echo "  python examples/simple_nn.py"
echo ""
echo "If you encounter any issues, please check that all dependencies are properly installed."