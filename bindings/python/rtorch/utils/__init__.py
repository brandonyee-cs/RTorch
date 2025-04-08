"""
RTorch Utilities Submodule (`rtorch.utils`)

Provides utility functions, currently including serialization helpers.
"""

# Import the utils submodule created by the Rust extension
from .. import utils as _utils_rust

# --- Re-export Utilities ---
# Note: The Rust implementation had issues exposing these easily.
# These might remain non-functional until the FFI challenges are resolved.
save = _utils_rust.save
load = _utils_rust.load
# DataParallel = _utils_rust.DataParallel # If exposed

# --- Define __all__ ---
__all__ = [
    "save",
    "load",
    # "DataParallel",
]

# Cleanup internal reference
del _utils_rust