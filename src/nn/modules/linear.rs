//! # Linear Layer Module

use crate::nn::Module;
use crate::tensor::{ops, randn, zeros, Tensor, TensorData, TensorError};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Applies a linear transformation to the incoming data: `y = xA^T + b`.
///
/// Input shape: `(N, *, in_features)` where `*` means any number of additional dimensions.
/// Output shape: `(N, *, out_features)`
#[derive(Debug)]
pub struct Linear {
    // Parameters of the layer
    pub weight: Tensor, // Shape: (out_features, in_features)
    pub bias: Option<Tensor>, // Shape: (out_features)

    // Store dimensions for reference
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Creates a new Linear module.
    ///
    /// # Arguments
    /// * `in_features`: Size of each input sample.
    /// * `out_features`: Size of each output sample.
    /// * `bias`: Whether to include a bias term. Defaults to `true`.
    ///
    /// Parameters are initialized using Kaiming uniform initialization for weights
    /// and zeros or uniform for bias (similar to PyTorch defaults).
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self, TensorError> {
        // Kaiming uniform initialization for weights
        let k = (1.0 / in_features as TensorData).sqrt();
        // Weight shape: (out_features, in_features)
        let weight_data = crate::tensor::rand(&[out_features, in_features], false)? // Use internal rand if available, requires rand feature
             .data_clone() // Get ArrayD
            * (2.0 * k) // Scale for uniform range [-k, k]
            - k;
        let weight = Tensor::new(weight_data, true); // Weights require gradient

        let bias_tensor = if bias {
            // Bias shape: (out_features)
            // Initialize bias similar to PyTorch: uniform(-k, k)
            let bias_data = crate::tensor::rand(&[out_features], false)? // Use internal rand
                .data_clone()
                * (2.0 * k)
                - k;
             Some(Tensor::new(bias_data, true)) // Bias requires gradient if it exists
            // Or initialize with zeros:
            // Some(zeros(&[out_features], true))
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias: bias_tensor,
            in_features,
            out_features,
        })
    }
}

impl Module for Linear {
    /// Performs the forward pass: `input @ weight.T + bias`.
    /// Handles inputs with more than 2 dimensions by flattening extra dimensions.
    /// (Note: PyTorch Linear doesn't flatten automatically, it works on the last dim.
    /// Let's follow PyTorch behavior).
    fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        // Input shape: (..., in_features)
        // Weight shape: (out_features, in_features)
        // Output = Input @ Weight.T + Bias
        // Output shape: (..., out_features)

        // 1. Transpose weight: (in_features, out_features)
        let weight_t = ops::transpose(&self.weight, 0, 1)?; // Needs transpose op

        // 2. Matrix multiplication: Input @ Weight.T
        // Matmul should handle input shapes like (N, *, K) @ (K, M) -> (N, *, M)
        let output = ops::matmul(input, &weight_t)?;

        // 3. Add bias (if exists)
        if let Some(ref bias) = self.bias {
            // Bias shape: (out_features)
            // Output shape: (..., out_features)
            // Need broadcasting add: output + bias
            ops::add(&output, bias) // Assumes ops::add handles broadcasting correctly
        } else {
            Ok(output)
        }

         // TODO: Ensure ops::transpose and ops::matmul (with >2D handling) and ops::add (with broadcasting) are implemented.
        // Placeholder if ops not ready:
        // Err(TensorError::Generic("Linear forward requires transpose, matmul, add ops.".to_string()))
    }

    fn parameters(&self) -> BTreeMap<String, Tensor> {
        let mut params = BTreeMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    // tensors() defaults to parameters() here, which is correct as Linear has no buffers.

    // train() and eval() have no effect on Linear layer, so default implementation is fine.
}

// Helper for transpose (can be moved to ops later)
mod ops_local {
    use super::*;
     use ndarray::{Axis};

    pub fn transpose(tensor: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor, TensorError> {
        let data = tensor.data();
        let mut axes: Vec<usize> = (0..data.ndim()).collect();
        if dim0 >= axes.len() || dim1 >= axes.len() {
             return Err(TensorError::Generic(format!("Transpose dims out of bounds: {:?}, {:?} for ndim {}", dim0, dim1, data.ndim())));
        }
        axes.swap(dim0, dim1);
        let permuted_view = data.view().permuted_axes(axes);

        // Create a new tensor. For autograd, need to track this.
        // For now, let's make a copy. A proper transpose op needs a backward pass.
        // If we make a copy, the autograd link might be broken unless handled explicitly.
        let result_data = permuted_view.to_owned();

        // TODO: Implement TransposeBackward op and use create_op_result
        // For now, assume transpose doesn't require grad tracking itself,
        // relying on matmul backward to handle the transposed weight.
        // This is a simplification.
        // Ok(Tensor::new(result_data, tensor.requires_grad))

        // Let's assume a basic TransposeBackward exists for demonstration
         use crate::tensor::autograd::op_abstractions::TransposeBackward; // Assume this exists
         use crate::tensor::autograd::create_op_result; // Assume this helper is callable

         create_op_result(result_data, vec![tensor.clone()], Box::new(TransposeBackward::new(dim0, dim1)))

    }
}
// Use the local definition for now
use ops_local::transpose;

// --- We need the rand function from tensor::mod ---
// Ensure tensor::rand is public or move rand generation here.
// Or add `rand` crate dependency directly here for initialization.
use rand::{Rng, distributions::{Distribution, Uniform}};

fn kaiming_uniform_data(shape: &[usize], fan_in: usize) -> ArrayD<TensorData> {
    let k = (1.0 / fan_in as TensorData).sqrt();
    let range = Uniform::new(-k, k);
    let mut rng = rand::thread_rng();
    ArrayD::from_shape_simple_fn(IxDyn(shape), || range.sample(&mut rng))
}

impl Linear {
    /// Alternate constructor using direct ndarray initialization
     pub fn new_with_init(in_features: usize, out_features: usize, bias: bool) -> Self {
         let weight_data = kaiming_uniform_data(&[out_features, in_features], in_features);
         let weight = Tensor::new(weight_data, true);

         let bias_tensor = if bias {
             let k = (1.0 / in_features as TensorData).sqrt();
             let bias_data = kaiming_uniform_data(&[out_features], in_features); // Same range as weight bias
             Some(Tensor::new(bias_data, true))
         } else {
             None
         };

         Linear {
             weight,
             bias: bias_tensor,
             in_features,
             out_features,
         }
     }
}


// ---- TransposeBackward Placeholder (Should be in tensor/autograd/op_abstractions.rs) ----
namespace_placeholder_autograd! { // Faking module structure for the example
    #[derive(Debug)]
    pub struct TransposeBackward {
        dim0: usize,
        dim1: usize,
    }
    impl TransposeBackward { pub fn new(d0: usize, d1: usize) -> Self { Self { dim0: d0, dim1: d1 }}}
    impl BackwardOp for TransposeBackward {
        fn backward( &self, inputs: &[Tensor], output_grad: &Tensor ) -> Result<Vec<Tensor>, TensorError> {
            // Gradient of transpose is transpose of gradient
            Ok(vec![transpose(output_grad, self.dim0, self.dim1)?])
        }
    }
}
// Macro to avoid actual module dependencies for this example structure
macro_rules! namespace_placeholder_autograd {
    ($($item:item)*) => { $($item)* }
}
use crate::tensor::autograd::{BackwardOp, Tensor, TensorError}; // Add necessary imports for placeholder
// --- End Placeholder ---