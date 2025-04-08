"""
RTorch Neural Network Submodule (`rtorch.nn`)

Provides layers, containers, activation functions, and loss functions.
"""

# Import the nn submodule created by the Rust extension in the parent __init__
from .. import nn as _nn_rust # Access parent's imported 'nn'

# --- Re-export Modules ---
Linear = _nn_rust.PyLinear # Use the PyO3 class name
ReLU = _nn_rust.PyReLU
Sequential = _nn_rust.PySequential
# Add other modules like Conv2d, LSTM, BatchNorm2d etc. when their wrappers are created
# Dropout = _nn_rust.PyDropout # Assuming PyDropout wrapper exists

# --- Re-export Functional submodule ---
functional = _nn_rust.functional

# --- Re-export Loss classes (if defined as PyO3 classes) ---
# MSELoss = _nn_rust.MSELoss # If MSELoss PyO3 class was exposed
# CrossEntropyLoss = _nn_rust.CrossEntropyLoss # If CrossEntropyLoss PyO3 class was exposed
# NLLLoss = _nn_rust.NLLLoss # If NLLLoss PyO3 class was exposed

# --- Define __all__ for `from rtorch.nn import *` ---
__all__ = [
    # Modules
    "Linear",
    "ReLU",
    "Sequential",
    # "Dropout", # Uncomment when added
    # "Conv2d", # Uncomment when added
    # "LSTM",   # Uncomment when added
    # "BatchNorm2d", # Uncomment when added

    # Submodules
    "functional",

    # Loss Functions (If exposed as classes, otherwise use functional)
    # "MSELoss",
    # "CrossEntropyLoss",
    # "NLLLoss",
]

# Cleanup internal reference
del _nn_rust