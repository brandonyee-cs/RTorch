"""
RTorch - A PyTorch-like library in Rust.

This package provides a Tensor object and neural network building blocks
implemented in Rust for performance and safety, exposed via Python bindings.
"""

import sys

# --- Check Dependencies ---
# Although setup.py requires numpy, it's good practice to check here too
# in case the environment is somehow broken.
try:
    import numpy
except ImportError as e:
    print("Error: NumPy is required for RTorch but could not be imported.", file=sys.stderr)
    print("Please install NumPy: pip install numpy", file=sys.stderr)
    # Re-raise the exception to prevent further loading if numpy is critical at import time
    raise e from None

# --- Import the Rust Extension Module ---
# The name 'rtorch_lib' must match the second part of the name in
# RustExtension("rtorch.rtorch_lib", ...) in setup.py
try:
    from . import rtorch_lib as _rust # Use _rust convention for the raw binding module
except ImportError as e:
    # Provide a more helpful error message if the extension failed to build or import
    print(f"Error: Could not import the RTorch Rust extension module.", file=sys.stderr)
    print("Ensure that the package was installed correctly (e.g., 'pip install -e .' from bindings/python).", file=sys.stderr)
    print(f"Original error: {e}", file=sys.stderr)
    raise e from None


# --- Re-export Core Components ---

# Tensor class
Tensor = _rust.Tensor

# Tensor creation functions
tensor = _rust.tensor
zeros = _rust.zeros
ones = _rust.ones
rand = _rust.rand
randn = _rust.randn

# --- Expose Submodules (`nn`, `optim`, `utils`) ---

# Directly assign the submodules created in Rust's #[pymodule]
nn = _rust.nn
optim = _rust.optim
utils = _rust.utils

# --- Version Information ---
# Try to get version from package metadata if installed
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("rtorch").version
except Exception:
    # Fallback if not installed or pkg_resources is unavailable
    # Should match version in setup.py / Cargo.toml
    __version__ = "0.1.0" # Placeholder

# --- Clean up namespace ---
# Remove imported modules that users don't need direct access to
# (optional, but good practice)
del sys
del numpy
# del pkg_resources # Keep if needed elsewhere? Unlikely for main __init__.
del _rust # Hide the raw Rust binding module

# --- Define what `from rtorch import *` imports (optional) ---
# List public API elements. Be selective.
__all__ = [
    # Core
    "Tensor",
    "__version__",
    # Creation Ops
    "tensor",
    "zeros",
    "ones",
    "rand",
    "randn",
    # Submodules
    "nn",
    "optim",
    "utils",
    # Potentially re-export specific common items from submodules?
    # e.g., "nn.Linear", "optim.SGD" - but direct access via submodule is standard.
]