//! # Dropout Layer Module

use crate::nn::Module;
use crate::tensor::{ops, Tensor, TensorData, TensorError};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::cell::Cell; // For interior mutability of the training flag
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray::ArrayD;


/// During training, randomly zeroes some elements of the input tensor with probability `p`.
/// The remaining elements are scaled up by `1 / (1 - p)`.
/// During evaluation, this layer does nothing and acts as an identity function.
#[derive(Debug)]
pub struct Dropout {
    p: f64, // Dropout probability
    // Use Cell for interior mutability for the training flag.
    // This allows `train()` and `eval()` to take `&self`.
    is_training: Cell<bool>,
}

impl Dropout {
    /// Creates a new Dropout module.
    /// # Arguments
    /// * `p`: Probability of an element to be zeroed. Default: 0.5
    pub fn new(p: f64) -> Result<Self, TensorError> {
        if !(0.0..=1.0).contains(&p) {
            // Technically p=1.0 is allowed but zeros everything. Check if < 1? PyTorch allows 1.
             return Err(TensorError::Generic("Dropout probability must be between 0 and 1".to_string()));
        }
        Ok(Dropout {
            p,
            is_training: Cell::new(true), // Default to training mode
        })
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        if !self.is_training.get() || self.p == 0.0 {
            // If not training or p=0, return input directly (identity function)
            // Clone necessary to maintain graph structure if input is used elsewhere.
            // A true identity op might be better.
            // For now, clone works but might be inefficient if input is large.
            Ok(input.clone())
            // Alternative: Define an identity op in tensor::ops
            // ops::identity(input)
        } else {
            // Apply dropout mask during training
            let scale = 1.0 / (1.0 - self.p);
             // Generate mask with same shape as input
            // Requires rand crate and ndarray-rand features
            let mask_data = ArrayD::random_using(input.shape(), Uniform::new(0.0f64, 1.0f64), &mut rand::thread_rng())
                .mapv(|x| if x < self.p { 0.0 } else { scale as TensorData }); // Convert scale to TensorData

            // Create a tensor from the mask. It does not require gradient.
            let mask = Tensor::new(mask_data, false);

            // Apply mask using element-wise multiplication
            ops::mul(input, &mask)
        }
    }

    /// Dropout has no parameters.
    fn parameters(&self) -> BTreeMap<String, Tensor> {
        BTreeMap::new()
    }

    /// Sets the module to training mode. Affects forward pass.
    // Takes &self due to using Cell for is_training.
    fn train(&self) { // Note: Changed signature to &self
        self.is_training.set(true);
    }

    /// Sets the module to evaluation mode. Affects forward pass.
    // Takes &self due to using Cell for is_training.
    fn eval(&self) { // Note: Changed signature to &self
        self.is_training.set(false);
    }
}