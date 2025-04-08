//! # Neural Network Module (`nn`)
//!
//! Provides building blocks for creating neural networks, similar to `torch.nn`.
//! Includes modules (layers), loss functions, and functional interfaces.

use crate::tensor::{Tensor, TensorError};
use std::collections::OrderedDict; // Or BTreeMap for ordered parameters
use std::sync::{Arc, Mutex}; // For shared parameter access if needed
use std::fmt::Debug;

// --- Submodules ---
pub mod functional;
pub mod modules;
pub mod loss;

// Re-export common items
pub use modules::*; // Re-export all layer modules (Linear, Conv, etc.)
pub use loss::*; // Re-export loss functions
pub use functional::*; // Re-export functional interface

// --- Core Trait: Module ---

/// Base trait for all neural network modules (layers, containers, etc.).
/// Defines the essential `forward` method and parameter management.
/// Needs `Debug` for potential printing/inspection.
/// `'static` bound often needed for storing modules in collections or closures.
pub trait Module: Debug + Send + Sync + 'static {
    /// Performs the forward pass of the module.
    ///
    /// # Arguments
    /// * `input`: The input tensor(s) to the module.
    ///
    /// # Returns
    /// * `Result<Tensor, TensorError>`: The output tensor of the module.
    fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError>;

    /// Returns a collection of the module's parameters (tensors that require gradients).
    /// The key is a descriptive name (e.g., "weight", "bias").
    /// Uses BTreeMap for deterministic order, useful for optimizers.
    /// The Tensor itself should be returned (cloned Arc).
    fn parameters(&self) -> std::collections::BTreeMap<String, Tensor>;

     /// Returns a collection of all tensors within the module, including parameters and buffers.
     /// Buffers are tensors that are part of the module's state but are not optimized
     /// (e.g., running mean/variance in BatchNorm).
     /// Default implementation just returns parameters. Subclasses can override.
    fn tensors(&self) -> std::collections::BTreeMap<String, Tensor> {
        self.parameters()
    }

    /// Zeros the gradients of all parameters within the module.
    /// Iterates through parameters and calls `zero_grad` on each.
    /// Requires mutable access conceptually, handled via interior mutability of Tensor's grad field.
    fn zero_grad(&self) {
        for (_name, param) in self.parameters() {
             // Need a way to call zero_grad on the Tensor.
             // If zero_grad takes &mut self, this won't work directly.
             // If Tensor::zero_grad uses internal locking (like we implemented), this is fine.

             // Assuming Tensor::zero_grad works on &self with internal mutability:
             param.zero_grad(); // This internally locks and modifies the grad tensor

             // If Tensor::zero_grad requires &mut self, we'd need a redesign, maybe:
             // let mut grad_mut = param.grad_mut_lock(); // Hypothetical method to get lock guard
             // grad_mut.fill_(0.0);
        }
    }

    /// Sets the module and its submodules to training mode.
    /// Affects behavior of layers like Dropout and BatchNorm.
    /// Default implementation does nothing; modules with mode-dependent behavior override this.
    fn train(&mut self) {
        // Iterate through submodules if this is a container? Needs a way to access submodules.
        // For a single layer, might set an internal `is_training` flag.
    }

    /// Sets the module and its submodules to evaluation mode.
    /// Affects behavior of layers like Dropout and BatchNorm.
    /// Default implementation does nothing; modules with mode-dependent behavior override this.
    fn eval(&mut self) {
        // Iterate through submodules if this is a container? Needs a way to access submodules.
        // For a single layer, might set an internal `is_training` flag to false.
    }

    // Potential future additions:
    // fn to(&self, device: Device) -> Self; // Move module to device
    // fn load_state_dict(&mut self, state_dict: BTreeMap<String, Tensor>);
    // fn state_dict(&self) -> BTreeMap<String, Tensor>;
    // fn add_module(&mut self, name: &str, module: Arc<dyn Module>); // For Sequential/ModuleList
    // fn modules(&self) -> Vec<Arc<dyn Module>>; // Get submodules
}

// --- Common Module Structures (Examples) ---

/// A sequential container for modules.
/// Modules will be added sequentially to the container.
/// The forward pass applies each module in order.
#[derive(Debug)]
pub struct Sequential {
    modules: Vec<Arc<dyn Module>>, // Store modules dynamically
    // Or use an ordered map:
    // named_modules: OrderedDict<String, Arc<dyn Module>>,
}

impl Sequential {
    /// Creates a new empty Sequential container.
    pub fn new() -> Self {
        Sequential { modules: Vec::new() }
    }

    /// Creates a Sequential container from a vector of modules.
    pub fn from_modules(modules: Vec<Arc<dyn Module>>) -> Self {
        Sequential { modules }
    }

    /// Adds a module to the sequence.
    pub fn add_module(&mut self, module: Arc<dyn Module>) {
        self.modules.push(module);
    }

     // Consider adding `add_named_module` if using a map internally.
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        let mut current_tensor = input.clone(); // Start with the input
        for module in &self.modules {
            current_tensor = module.forward(¤t_tensor)?; // Apply each module
        }
        Ok(current_tensor)
    }

    fn parameters(&self) -> std::collections::BTreeMap<String, Tensor> {
        let mut params = std::collections::BTreeMap::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.parameters() {
                // Prefix parameter names with module index for uniqueness
                params.insert(format!("{}.{}", i, name), param);
            }
        }
        params
    }

     fn tensors(&self) -> std::collections::BTreeMap<String, Tensor> {
        let mut tensors = std::collections::BTreeMap::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, tensor) in module.tensors() {
                 tensors.insert(format!("{}.{}", i, name), tensor);
            }
        }
        tensors
    }

    fn train(&mut self) {
         for module_arc in &self.modules {
             // To call `train` which takes `&mut self`, we need mutable access to the module.
             // `Arc<dyn Module>` doesn't easily allow this. Common patterns:
             // 1. Use `Arc<Mutex<dyn Module>>`: Adds runtime overhead for locking.
             // 2. Redesign: Maybe `train`/`eval` take `&self` and use interior mutability
             //    for flags like `is_training`. This seems more idiomatic for Rust/PyTorch.
             // Let's assume Option 2: `train`/`eval` use interior mutability.
             // module_arc.train(); // Assuming train/eval take &self

             // If we stick to `&mut self`, this structure is harder to implement safely.
             // We might need unsafe code or a different container design.
             // For now, let's comment out the recursive call, assuming leaf modules handle it.
             // --- Let's assume train/eval take &self ---
             // module_arc.train(); // This would require Module::train(&self)
         }
         // If train takes `&mut self`, we cannot implement it for Arc<dyn Module> directly.
         // The container itself doesn't have mutable state to track training mode.
         // The contained modules need to handle their own state.
          eprintln!("Warning: Sequential::train() cannot recursively call train() on Arc<dyn Module> if it requires &mut self.");

    }

    fn eval(&mut self) {
         for module_arc in &self.modules {
             // Same issue as train()
             // module_arc.eval(); // Assuming eval takes &self
         }
          eprintln!("Warning: Sequential::eval() cannot recursively call eval() on Arc<dyn Module> if it requires &mut self.");
    }

    // Override zero_grad to iterate through contained modules
    fn zero_grad(&self) {
        for module in &self.modules {
            module.zero_grad(); // Call zero_grad on each submodule
        }
    }
}