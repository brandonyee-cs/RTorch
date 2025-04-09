# RTorch

**RTorch** is an experimental deep learning library built with Rust, aiming to provide a PyTorch-like experience for Python users. It leverages Rust's performance, safety features, and ecosystem while exposing a familiar API through PyO3 bindings.

**⚠️ Disclaimer: Highly Experimental ⚠️**

This project is currently a **proof-of-concept** and is **NOT intended for production use**. It lacks many features, optimizations, and rigorous testing found in mature libraries like PyTorch. Its primary goals are exploration, learning, and demonstrating the potential of Rust in this domain.

## Goals

*   Explore building a deep learning framework core in Rust.
*   Provide a familiar Tensor API (`rtorch.Tensor`) similar to PyTorch.
*   Implement basic reverse-mode automatic differentiation (Autograd).
*   Offer common neural network building blocks (`rtorch.nn`).
*   Include standard optimization algorithms (`rtorch.optim`).
*   Expose these features efficiently to Python using PyO3.
*   Leverage Rust's strengths like memory safety and performance (CPU-focused initially).

## Features

*   **Tensor:**
    *   Multi-dimensional tensor (`rtorch.Tensor`) backed by Rust's `ndarray`.
    *   **CPU Only:** Currently computations run only on the CPU.
    *   Basic mathematical operations: `+`, `-`, `*`, `matmul`, `sum`, `mean`, `reshape`, etc.
    *   Interoperability with NumPy arrays for creation (`torch.tensor(np_array)`) and retrieval (`tensor.numpy()`).
    *   Creation functions: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.rand`, `torch.randn`.
*   **Autograd:**
    *   Automatic differentiation engine tracking tensor operations.
    *   Gradient computation via `.backward()` on scalar tensors.
    *   Access gradients through the `.grad` attribute.
*   **Neural Networks (`rtorch.nn`):**
    *   `nn.Module` base structure (conceptually similar to PyTorch).
    *   Basic Layers:
        *   `nn.Linear`: Fully connected layer with Kaiming initialization.
        *   `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`: Activation functions as modules.
        *   `nn.Dropout`: Dropout layer (respects `.train()`/`.eval()` modes).
        *   `nn.Sequential`: A container for chaining modules (Python-side implementation for simplicity).
    *   Functional API (`rtorch.nn.functional`): Stateless functions like `F.relu`, `F.mse_loss`, `F.cross_entropy_loss`.
    *   Placeholders for more complex layers (Conv, RNN, BatchNorm, LayerNorm) exist in the Rust code but are **not implemented**.
*   **Optimizers (`rtorch.optim`):**
    *   `optim.Optimizer` base structure.
    *   `optim.SGD`: Stochastic Gradient Descent with momentum, weight decay, and Nesterov support.
    *   `optim.Adam`: Adam optimizer with AMSGrad option.
    *   `optim.Adagrad`: Adagrad optimizer.
*   **Python Bindings:**
    *   Built using `PyO3`.
    *   Aims for an API familiar to PyTorch users (`import rtorch as torch`).
*   **Utilities (`rtorch.utils`):**
    *   Placeholders for model serialization (`save`, `load`) using `serde` and `bincode` (FFI challenges limit current Python usability).
    *   Conceptual placeholder for CPU data parallelism (`DataParallel`).

## Project Status

*   **Alpha / Experimental:** Core functionality is present but limited.
*   **CPU Only:** No GPU support.
*   **Missing Operations:** Many tensor operations (advanced indexing, slicing, FFT, comprehensive linear algebra, etc.) are missing.
*   **Missing Layers:** Convolutional, Recurrent, Normalization (BatchNorm, LayerNorm), Pooling, Embedding layers are not implemented.
*   **Broadcasting:** Basic broadcasting (tensor-scalar, some simple cases) might work, but comprehensive NumPy-style broadcasting is likely incomplete.
*   **Serialization/Parallelism:** Utilities for saving models and data parallelism are placeholders and not fully functional from Python.
*   **Error Handling:** Basic error handling exists, but could be more robust and informative.
*   **Performance:** Performance is not optimized and likely significantly slower than mature libraries.
*   **API Stability:** The API is subject to change.

## Installation

### Prerequisites

*   **Rust:** Latest stable toolchain (via [rustup](https://rustup.rs/)).
*   **Python:** Version 3.7 or higher.
*   **NumPy:** Required for tensor creation/conversion.
*   **Build Tools:** A C compiler might be needed for Python extension building. `setuptools-rust` is used for building.

### From Source (Editable Install for Development)

This is the recommended method for development or trying out the library.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/rtorch.git # Replace with your actual repo link
    cd rtorch
    ```
2.  **Navigate to the Python bindings directory:**
    ```bash
    cd bindings/python
    ```
3.  **Install in editable mode:**
    ```bash
    pip install -e .
    ```
    This command uses `setup.py` (which uses `setuptools-rust`) to compile the Rust extension module and link it into your Python environment so you can edit the Python/Rust code and test changes without reinstalling.

*(Note: If you plan to publish to PyPI, you would use tools like `maturin` or `setuptools-rust` to build wheels).*

## Usage Examples

```python
import numpy as np
import rtorch as torch
import rtorch.nn as nn
import rtorch.optim as optim

# --- Tensor Creation ---
print("--- Tensor Creation ---")
# From Python list
t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"From list:\n{t1}")

# From NumPy array
t2 = torch.tensor(np.array([5.0, 6.0], dtype=np.float32))
print(f"From NumPy:\n{t2}")

# Using factory functions
t_zeros = torch.zeros(2, 3)
t_ones = torch.ones(3, 1)
t_randn = torch.randn(2, 2) # Requires 'rand' feature during compilation
print(f"Zeros:\n{t_zeros}")
print(f"Randn:\n{t_randn}")
print(f"Shape: {t_randn.shape}, Dtype: {t_randn.dtype}") # dtype currently fixed

# --- Tensor Operations ---
print("\n--- Tensor Operations ---")
t_a = torch.tensor([[1., 2.], [3., 4.]])
t_b = torch.ones(2, 2)
print(f"Add:\n{t_a + t_b}")
print(f"Mul:\n{t_a * 2.0}") # Scalar multiplication
# print(f"Matmul:\n{t_a.matmul(t_b)}") # Check shapes for matmul

# --- Autograd ---
print("\n--- Autograd ---")
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([5.0], requires_grad=True)
b = torch.tensor([-1.0], requires_grad=True)

# Simple computation graph: y = w * x + b
y = w * x + b # y = 5*2 - 1 = 9.0

# Compute gradients
y.backward()

print(f"y = {y.numpy()[0]}")
print(f"Gradient dy/dw: {w.grad.numpy()[0]}") # Should be x = 2.0
print(f"Gradient dy/dx: {x.grad.numpy()[0]}") # Should be w = 5.0
print(f"Gradient dy/db: {b.grad.numpy()[0]}") # Should be 1.0

# --- Neural Network Example (Simple Regression) ---
print("\n--- Neural Network ---")
# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 5), # Input features=10, output features=5
    nn.ReLU(),
    nn.Linear(5, 1)   # Input features=5, output features=1
)
print(f"Model:\n{model}") # Basic repr

# Define an optimizer
# Needs access to parameters - model.parameters() might not work on PySequential yet
# Workaround: Get parameters from individual layers if needed
# params_list = list(model.modules[0].parameters().values()) + list(model.modules[2].parameters().values()) # Example manual param collection
# optimizer = optim.SGD(params_list, lr=0.01)
# For demonstration, let's create a standalone layer and optimizer
layer = nn.Linear(10, 1)
optimizer = optim.Adam(layer.parameters().values(), lr=0.001)


# Dummy data
input_data = torch.randn(4, 10) # Batch size=4, features=10
target = torch.randn(4, 1)      # Batch size=4, targets=1

# Simple training loop
for epoch in range(5): # Few dummy epochs
    optimizer.zero_grad()    # Zero gradients before backward pass
    output = layer(input_data) # Forward pass
    loss = nn.functional.mse_loss(output, target) # Calculate loss
    loss.backward()          # Compute gradients
    optimizer.step()         # Update parameters

    print(f"Epoch {epoch}, Loss: {loss.numpy()[0]:.4f}")

# Check parameter gradients after training
print("\nGradients after training loop (example):")
for name, param in layer.parameters().items():
    if param.grad is not None:
        print(f"Parameter '{name}', Grad shape: {param.grad.shape}")
    else:
        print(f"Parameter '{name}', Grad: None")
```

## Project Structure

```
rtorch/
├── Cargo.toml                  # Rust package manifest
├── README.md                   # This documentation
├── .gitignore                  # Git ignore file
├── src/                        # Rust source code
│   ├── lib.rs                  # Rust library entry point
│   ├── tensor/                 # Core Tensor & Autograd implementation
│   ├── nn/                     # Neural Network modules (layers, functional, loss)
│   ├── optim/                  # Optimization algorithms
│   ├── utils/                  # Utility functions (serialization, parallelism placeholders)
│   └── bindings/               # Rust-side code for Python bindings (PyO3 setup)
├── bindings/                   # Language binding files (Python package source)
│   └── python/                 # Python package 'rtorch'
│       ├── setup.py            # Python package build configuration (uses setuptools-rust)
│       ├── rtorch/             # The actual Python package directory
│       │   ├── __init__.py     # Exports Rust components into Python namespace
│       │   ├── tensor.py       # Tensor class re-export (potential Python helpers)
│       │   ├── nn/             # NN submodule structure
│       │   │   └── __init__.py
│       │   ├── optim/          # Optim submodule structure
│       │   │   └── __init__.py
│       │   └── utils/          # Utils submodule structure
│       │       └── __init__.py
│       └── tests/              # Python unit tests
├── examples/                   # Placeholder for example projects using RTorch
├── benches/                    # Placeholder for Rust benchmarks
└── tests/                      # Placeholder for Rust integration tests
```

## Development & Testing

### Rust Core

*   **Build:** `cargo build` (debug) or `cargo build --release` (optimized)
    *   Enable features: `cargo build --features "rand"`
*   **Test:** `cargo test`
*   **Lint:** `cargo clippy`
*   **Format:** `cargo fmt`

### Python Bindings & Package

1.  Ensure you have installed the package in editable mode (`pip install -e .` inside `bindings/python`).
2.  Changes to Rust code require recompilation. `pip install -e .` handles this, or you can use `maturin develop` inside the `bindings/python` directory.
3.  Run Python unit tests:
    ```bash
    cd bindings/python
    python -m unittest discover tests
    # Or use pytest: pip install pytest; pytest
    ```

## Contributing

Contributions are welcome, but please keep the experimental nature of the project in mind!

1.  **Open an Issue:** Discuss proposed changes or bug fixes by opening an issue first.
2.  **Fork & Branch:** Fork the repository and create a new branch for your changes.
3.  **Develop:** Implement your changes, adhering to Rust coding standards (`clippy`, `fmt`).
4.  **Test:** Add relevant tests (Rust and/or Python) and ensure all tests pass.
5.  **Pull Request:** Submit a pull request detailing your changes.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project is heavily inspired by [PyTorch](https://pytorch.org/) and aims to replicate parts of its API and functionality. It relies on fantastic Rust crates like `ndarray`, `PyO3`, `serde`, `rayon`, and `rand`.