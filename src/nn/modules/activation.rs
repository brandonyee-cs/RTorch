//! # Activation Function Modules

use crate::nn::Module;
use crate::nn::functional as F; // Use functional interface
use crate::tensor::{Tensor, TensorError};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Applies the Rectified Linear Unit function element-wise.
/// `ReLU(x) = max(0, x)`
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU; // No parameters, so can be simple struct

impl ReLU {
    /// Creates a new ReLU module.
    pub fn new() -> Self {
        ReLU {}
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        F::relu(input) // Delegate to functional interface
    }

    /// ReLU has no parameters.
    fn parameters(&self) -> BTreeMap<String, Tensor> {
        BTreeMap::new()
    }

    // No state, train/eval do nothing. Default impl is fine.
}


/// Applies the Sigmoid function element-wise.
/// `Sigmoid(x) = 1 / (1 + exp(-x))`
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid; // No parameters

impl Sigmoid {
    pub fn new() -> Self { Sigmoid {} }
}

impl Module for Sigmoid {
     fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        F::sigmoid(input) // Delegate to functional interface
    }

    fn parameters(&self) -> BTreeMap<String, Tensor> {
        BTreeMap::new()
    }
}

/// Applies the Tanh function element-wise.
/// `Tanh(x) = tanh(x)`
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh; // No parameters

impl Tanh {
     pub fn new() -> Self { Tanh {} }
}

impl Module for Tanh {
     fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        F::tanh(input) // Delegate to functional interface
    }

    fn parameters(&self) -> BTreeMap<String, Tensor> {
        BTreeMap::new()
    }
}

// Add other activation modules like Softmax if needed, although Softmax
// is often used functionally or as the last layer implicitly with CrossEntropyLoss.