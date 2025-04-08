//! # Optimization Algorithms (`optim`)
//!
//! Provides implementations of common optimization algorithms used to update model parameters.

use crate::tensor::{Tensor, TensorError};
use std::collections::BTreeMap; // Or HashMap if parameter order isn't critical for state

// --- Submodules ---
pub mod sgd;
pub mod adam;
pub mod adagrad;

// Re-export optimizers
pub use sgd::SGD;
pub use adam::Adam;
pub use adagrad::Adagrad;

// --- Optimizer Trait ---

/// Base trait for all optimizers.
/// Defines the essential methods for updating parameters and managing gradients.
pub trait Optimizer {
    /// Performs a single optimization step (parameter update).
    /// This method should access the gradients of the parameters it manages
    /// and update their data based on the specific algorithm's rules.
    ///
    /// # Returns
    /// * `Result<(), TensorError>`: Ok(()) if the step was successful, Err otherwise.
    fn step(&mut self) -> Result<(), TensorError>;

    /// Zeros the gradients of all parameters managed by the optimizer.
    /// It's crucial to call this before starting gradient computation for a new batch/iteration.
    fn zero_grad(&mut self);

    /// Returns a reference to the parameters managed by the optimizer.
    /// Useful for inspection or potentially saving/loading optimizer state alongside parameters.
    // This might be tricky if the optimizer only stores weak refs or IDs.
    // Alternatively, the optimizer might not need to expose the params directly.
    // Let's omit this for now and assume the optimizer holds the necessary info internally.
    // fn parameters(&self) -> &[Tensor]; // Or some iterator type

    // TODO: Add methods for managing learning rate schedules?
    // TODO: Add methods for saving/loading optimizer state?
}

// --- Helper Struct for Parameter Groups (Optional like PyTorch) ---
// Could allow different learning rates or settings per parameter group.
// pub struct ParamGroup {
//     params: Vec<Tensor>,
//     lr: f64,
//     // other options like weight_decay, momentum etc. specific to this group
// }

// --- Parameter Storage ---
// Optimizers need a way to hold onto the parameters they are optimizing.
// Storing `Tensor` directly (which clones the Arc) is common.
// Need to ensure these are the *same* tensors as in the model.

// We'll store parameters directly in the concrete optimizer structs for now.
// Example: `params: Vec<Tensor>` in SGD struct.