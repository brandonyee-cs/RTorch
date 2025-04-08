"""
Defines the Tensor object for RTorch.

This module primarily re-exports the Tensor class implemented in Rust
and provides a place for potential Python-specific additions or documentation.
"""

# Import the Tensor class from the underlying Rust extension module
# The main __init__.py handles the import path correctly.
from .rtorch_lib import Tensor as _RustTensor

# Re-export the Tensor class
Tensor = _RustTensor

# --- Python-Specific Enhancements (Optional) ---

# Example: Add a docstring to the class (though PyO3 might handle basic ones)
# Tensor.__doc__ = """
# RTorch Tensor object.

# Stores multi-dimensional arrays and supports automatic differentiation.
# Mirrors the basic functionality of PyTorch Tensors.

# Args:
#     data (array_like): Data for the tensor. Can be a list, tuple, NumPy array.
#     requires_grad (bool, optional): If True, track operations for autograd. Defaults to False.
# """

# Example: Add a Python-only helper method if needed
# def _tensor_plot_helper(self, *args, **kwargs):
#     """(Example) Helper to plot tensor data using matplotlib."""
#     try:
#         import matplotlib.pyplot as plt
#     except ImportError:
#         raise ImportError("matplotlib is required for plotting. Please install it.")
#
#     data = self.numpy() # Requires .numpy() method to be implemented
#     if data.ndim == 1:
#         plt.plot(data, *args, **kwargs)
#     elif data.ndim == 2:
#         plt.imshow(data, *args, **kwargs)
#     else:
#         print(f"Plotting not supported for {data.ndim}-dimensional tensors.")
#     plt.show()

# Attach the helper method to the Tensor class (Monkey Patching)
# setattr(Tensor, 'plot', _tensor_plot_helper)


# --- Type Hinting (Optional) ---
# You might want to define type aliases if using type checkers
# from typing import Union, List, Tuple
# from numpy import ndarray
# ArrayLike = Union[List, Tuple, ndarray, Tensor]


# --- Ensure attributes defined in Rust are discoverable (usually handled by PyO3) ---
# Example: Explicitly add properties if PyO3's detection isn't perfect (unlikely needed)
# @property
# def shape(self) -> Tuple[int, ...]:
#     return self._shape() # Assuming a private accessor if needed

# @property
# def grad(self) -> Optional['Tensor']:
#     return self._grad() # Assuming accessor


# Define __all__ for this submodule if needed
__all__ = [
    "Tensor",
]