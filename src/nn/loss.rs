//! # Loss Function Modules
//!
//! Provides modules for common loss functions, wrapping the functional implementations.
//! These modules typically don't have parameters but might hold configuration like reduction type.

use crate::nn::Module;
use crate::nn::functional as F; // Use functional interface
use crate::tensor::{Tensor, TensorError};
use std::collections::BTreeMap;
use std::fmt::Debug;

// --- Reduction Enum (Optional) ---
/// Specifies the reduction to apply to the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction applied. The loss is returned per element/batch item.
    None,
    /// The sum of the output will be divided by the number of elements.
    Mean,
    /// The output will be summed.
    Sum,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean // Default reduction is usually 'mean'
    }
}


// --- Base Loss Trait (Optional) ---
// We could define a dedicated Loss trait, but for simplicity,
// we can just use the existing Module trait as loss functions
// also perform a forward computation.

// --- Mean Squared Error Loss Module ---

/// Creates a criterion that measures the mean squared error (squared L2 norm) between
/// each element in the input `x` and target `y`.
/// The loss is `mean((x_i - y_i)^2)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MSELoss {
    // reduction: Reduction // TODO: Add reduction option later
}

impl MSELoss {
    pub fn new(/* reduction: Reduction */) -> Self {
        MSELoss { /* reduction */ }
    }
}

impl Module for MSELoss {
    /// Calculates the MSE loss.
    /// Expects input and target tensors of the same shape.
    /// The Module trait expects a single input, so we might need a different trait
    /// or accept a tuple/struct as input.
    /// Let's adapt by assuming `forward` takes the prediction, and the target is
    /// passed separately or handled differently in the training loop.
    ///
    /// A common pattern is `loss_fn(prediction, target)`. How to fit this in `Module`?
    /// 1. Modify Module trait: `fn forward(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor, TensorError>;` (Breaking change)
    /// 2. Accept a tuple: `fn forward(&self, inputs: (&Tensor, &Tensor)) -> Result<Tensor, TensorError>;`
    /// 3. Keep Module as is, and the training loop calls `F::mse_loss(pred, target)` directly.
    ///
    /// Option 3 is often preferred as loss calculation doesn't always fit the layer/module pattern.
    /// Let's keep the functional approach primary and make the Module version potentially less useful
    /// or demonstrate Option 2.
    ///
    /// Let's try Option 2 for demonstration, but using functional might be better practice.
    fn forward(&self, inputs: &Tensor) -> Result<Tensor, TensorError> {
        // This signature doesn't work for loss functions needing two inputs.
         Err(TensorError::Generic("MSELoss module requires two inputs (prediction, target). Use functional::mse_loss instead or adapt Module trait/input type.".to_string()))
    }

     // Alternative signature if Module trait was adapted or input is tuple:
     // fn forward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
     //    F::mse_loss(prediction, target) // Add reduction logic here later
     // }

    /// Loss functions typically don't have parameters.
    fn parameters(&self) -> BTreeMap<String, Tensor> {
        BTreeMap::new()
    }
}


// --- Negative Log Likelihood Loss Module ---

/// The negative log likelihood loss. Useful for training classification problems.
/// Expects log-probabilities as input and class indices as target.
#[derive(Debug, Clone, Copy, Default)]
pub struct NLLLoss {
     // reduction: Reduction // TODO: Add reduction option
     // weight: Option<Tensor> // TODO: Add per-class weighting
     // ignore_index: Option<i64> // TODO: Add ignore index
}

impl NLLLoss {
     pub fn new(/* ... */) -> Self { NLLLoss {} }
}

impl Module for NLLLoss {
    /// Calculates NLL Loss. See functional::nll_loss for details.
    fn forward(&self, _input: &Tensor) -> Result<Tensor, TensorError> {
         // Requires two inputs (log_probs, target). See MSELoss comments.
         Err(TensorError::Generic("NLLLoss module requires two inputs (log_probs, target). Use functional::nll_loss instead.".to_string()))
    }

    // Alternative signature:
    // fn forward(&self, log_probs: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    //     F::nll_loss(log_probs, target) // Add reduction, weight, ignore_index logic
    // }

    fn parameters(&self) -> BTreeMap<String, Tensor> {
        BTreeMap::new()
    }
}


// --- Cross Entropy Loss Module ---

/// This criterion computes the cross entropy loss between input logits and target.
/// It combines `LogSoftmax` and `NLLLoss` in one class.
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossEntropyLoss {
    // reduction: Reduction
    // weight: Option<Tensor>
    // ignore_index: Option<i64>
}

impl CrossEntropyLoss {
     pub fn new(/* ... */) -> Self { CrossEntropyLoss {} }
}

impl Module for CrossEntropyLoss {
     /// Calculates Cross Entropy Loss. See functional::cross_entropy_loss for details.
    fn forward(&self, _input: &Tensor) -> Result<Tensor, TensorError> {
         // Requires two inputs (logits, target). See MSELoss comments.
        Err(TensorError::Generic("CrossEntropyLoss module requires two inputs (logits, target). Use functional::cross_entropy_loss instead.".to_string()))
    }

     // Alternative signature:
     // fn forward(&self, logits: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
     //     F::cross_entropy_loss(logits, target) // Add reduction, weight, ignore_index logic
     // }

    fn parameters(&self) -> BTreeMap<String, Tensor> {
        BTreeMap::new()
    }
}


// --- Conclusion on Loss Modules ---
// Using the standard `Module` trait for loss functions is awkward because they
// typically require both predictions and targets as input.
// It's generally more idiomatic in PyTorch (and likely here) to:
// 1. Use the functional versions (`F::mse_loss`, `F::cross_entropy_loss`) directly in the training loop.
// 2. Define loss structs (like above) mainly for configuration (e.g., reduction, weights)
//    and then call a method like `compute(prediction, target)` on an instance of the struct,
//    or have the struct itself be callable if using a language feature like Rust's nightly `Fn` traits
//    or a custom trait for callability.

// For now, these module structs serve as placeholders demonstrating the structure,
// but using the `nn::functional` versions is recommended.