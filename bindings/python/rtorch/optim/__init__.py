"""
RTorch Optimization Submodule (`rtorch.optim`)

Provides optimization algorithms like SGD, Adam, etc.
"""

# Import the optim submodule created by the Rust extension
from .. import optim as _optim_rust

# --- Re-export Optimizers ---
SGD = _optim_rust.PySGD # Use the PyO3 class name
Adam = _optim_rust.PyAdam
# Adagrad = _optim_rust.PyAdagrad # Assuming PyAdagrad wrapper exists

# --- Define __all__ ---
__all__ = [
    "SGD",
    "Adam",
    # "Adagrad", # Uncomment when added
]

# Cleanup internal reference
del _optim_rust