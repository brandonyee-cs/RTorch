//! # Convolutional Layer Modules (Placeholders)

use crate::nn::Module;
use crate::tensor::{Tensor, TensorError};
use std::collections::BTreeMap;
use std::fmt::Debug;

// --- Conv2d Placeholder ---

/// Applies a 2D convolution over an input signal composed of several input planes.
/// (Not Implemented)
#[derive(Debug)]
pub struct Conv2d {
    // Parameters (weight, bias)
    // Attributes (in_channels, out_channels, kernel_size, stride, padding, etc.)
    _marker: (), // Placeholder field
}

impl Conv2d {
    /// Creates a new Conv2d module. (Not Implemented)
    pub fn new(
        _in_channels: usize,
        _out_channels: usize,
        _kernel_size: usize, // Or tuple (usize, usize)
        _stride: usize,      // Or tuple
        _padding: usize,     // Or tuple
        _bias: bool,
    ) -> Result<Self, TensorError> {
        // TODO: Implement parameter initialization (e.g., Kaiming uniform)
        Err(TensorError::Generic("Conv2d::new not implemented".to_string()))
    }
}

impl Module for Conv2d {
    fn forward(&self, _input: &Tensor) -> Result<Tensor, TensorError> {
        // Input shape: (N, C_in, H_in, W_in)
        // Output shape: (N, C_out, H_out, W_out)
        // TODO: Implement 2D convolution logic (likely using external crate like ndarray-conv or direct implementation)
        // TODO: Implement autograd for convolution (complex backward pass)
        Err(TensorError::Generic("Conv2d::forward not implemented".to_string()))
    }

    fn parameters(&self) -> BTreeMap<String, Tensor> {
        // TODO: Return weight and bias tensors
        BTreeMap::new()
    }

    // train/eval might be needed if using BatchNorm within a conv block later
}

// TODO: Implement Conv1d, Conv3d, Transposed Convolution etc.
// TODO: Implement pooling layers (MaxPool2d, AvgPool2d) often used with conv. These might go in a separate pooling.rs or stay here/functional.