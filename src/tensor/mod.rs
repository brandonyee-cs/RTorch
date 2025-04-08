//! # Tensor Module
//!
//! This module defines the core `Tensor` struct and related functionalities,
//! including storage, operations, and automatic differentiation.

use std::sync::{Arc, Mutex, RwLock}; // For shared ownership and mutability needed in autograd
use ndarray::{ArrayD, IxDyn}; // Using ndarray for underlying storage and ops
use pyo3::prelude::*; // Import PyO3 traits if Tensor will be exposed directly

// --- Submodules ---
pub mod storage;
pub mod ops;
pub mod autograd;

// --- Re-exports ---
pub use storage::Storage;
pub use autograd::{AutogradContext, GradFn, op_abstractions}; // Expose core autograd parts
// Potentially re-export common operations if defined as traits or functions here
// pub use ops::{add, mul, matmul};

// --- Error Handling ---
// Consider defining a specific TensorError enum
#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Incompatible shapes for operation {op}: {shape1:?} and {shape2:?}")]
    IncompatibleShapes {
        op: String,
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },
    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    #[error("Operation requires gradient but tensor does not have it")]
    RequiresGradNotSet,
    #[error("Tensor does not have a gradient computed")]
    NoGradient,
    #[error("Cannot perform operation on detached tensor")]
    DetachedTensor,
     #[error("ndarray error: {0}")]
    NdarrayError(#[from] ndarray::ShapeError), // Example: Wrap ndarray errors
    #[error("Autograd error: {0}")]
    AutogradError(String), // Generic autograd errors
    #[error("Generic error: {0}")]
    Generic(String),
}

// Define a type alias for the underlying data type (e.g., f32)
pub type TensorData = f32; // TODO: Make generic later? <T: DataType>

/// # Tensor
///
/// The core data structure for numerical computation, similar to PyTorch's Tensor.
/// It wraps an `ndarray::ArrayD` for storage and includes metadata for autograd.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Tensor {
    // Use Arc<RwLock<...>> for the data to allow sharing and interior mutability
    // needed for in-place operations and autograd gradient updates.
    // `data` holds the actual numerical values.
    #[pyo3(get)]
    data: Arc<RwLock<ArrayD<TensorData>>>,

    // Shape information (redundant with ndarray but useful for quick access)
    #[pyo3(get)]
    shape: Vec<usize>,

    // Metadata for automatic differentiation
    // `grad_context` stores information needed for the backward pass (computation graph).
    // `grad` stores the computed gradient after .backward() is called.
    // Using Arc<Mutex<...>> for grad_context and grad because they need to be
    // potentially modified by multiple tensors during backpropagation.
    grad_context: Option<Arc<Mutex<AutogradContext>>>,
    grad: Option<Arc<Mutex<Tensor>>>, // Gradient is also a Tensor

    // Flags controlling behavior
    #[pyo3(get, set)]
    requires_grad: bool, // Does this tensor need its gradient computed?
    #[pyo3(get)]
    is_leaf: bool,       // Is this a leaf node in the computation graph?
}

// --- Core Tensor Implementation (basic methods) ---
// We will implement PyO3 methods in `src/bindings/python.rs` or a dedicated file.
// Here we add inherent Rust methods.
impl Tensor {
    /// Creates a new Tensor from an ndarray::ArrayD.
    pub fn new(data: ArrayD<TensorData>, requires_grad: bool) -> Self {
        let shape = data.shape().to_vec();
        let tensor_data = Arc::new(RwLock::new(data));
        Tensor {
            data: tensor_data,
            shape,
            grad_context: None,
            grad: None,
            requires_grad,
            is_leaf: true, // Tensors created directly are leaf nodes
        }
    }

    /// Creates a Tensor that is not a leaf node (i.e., result of an operation).
    pub(crate) fn from_op(
        data: ArrayD<TensorData>,
        grad_context: Arc<Mutex<AutogradContext>>,
        requires_grad: bool, // Determined by inputs to the op
    ) -> Self {
        let shape = data.shape().to_vec();
        let tensor_data = Arc::new(RwLock::new(data));
        Tensor {
            data: tensor_data,
            shape,
            grad_context: Some(grad_context),
            grad: None,
            requires_grad,
            is_leaf: false,
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Provides read-only access to the underlying data.
    /// Note: This locks the RwLock for reading.
    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, ArrayD<TensorData>> {
        self.data.read().expect("Tensor data RwLock poisoned")
    }

    /// Provides mutable access to the underlying data.
    /// Use with extreme caution, especially regarding autograd implications.
    /// Often used internally or for specific non-gradient-tracked ops.
    /// Note: This locks the RwLock for writing.
    pub fn data_mut(&self) -> std::sync::RwLockWriteGuard<'_, ArrayD<TensorData>> {
        self.data.write().expect("Tensor data RwLock poisoned")
    }

    /// Clones the underlying data into a new ArrayD.
    pub fn data_clone(&self) -> ArrayD<TensorData> {
        self.data().clone()
    }

    /// Detaches the tensor from the computation graph.
    /// Returns a new tensor sharing the same data but without autograd history.
    pub fn detach(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data), // Share data
            shape: self.shape.clone(),
            grad_context: None, // Remove autograd info
            grad: None,
            requires_grad: false, // Detached tensors don't require grad by default
            is_leaf: true,       // A detached tensor is considered a leaf
        }
    }

    /// Sets the gradient for this tensor. Used internally by autograd.
    pub(crate) fn set_grad(&mut self, grad_tensor: Tensor) {
        if !self.requires_grad {
            // Maybe warn or error? For now, just ignore.
            // eprintln!("Warning: Setting gradient on tensor that does not require grad.");
            return;
        }
        self.grad = Some(Arc::new(Mutex::new(grad_tensor)));
    }

     /// Accumulates gradient. Used internally by autograd.
    pub(crate) fn accumulate_grad(&mut self, incoming_grad: Tensor) -> Result<(), TensorError> {
         if !self.requires_grad {
            // eprintln!("Warning: Accumulating gradient on tensor that does not require grad.");
            return Ok(());
        }
        // Ensure shapes match before accumulating
        if self.shape() != incoming_grad.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: incoming_grad.shape().to_vec(),
            });
        }

        // Lock the existing gradient (if any) and update it
        let mut existing_grad_lock = self.grad.get_or_insert_with(|| {
             // Initialize gradient with zeros if it doesn't exist
             let zeros_data = ArrayD::zeros(IxDyn(&self.shape));
             let zero_tensor = Tensor::new(zeros_data, false); // Grad tensor itself doesn't need grad
             Arc::new(Mutex::new(zero_tensor))
         }).lock().expect("Gradient Mutex poisoned");

        // Perform the accumulation (element-wise addition)
        // This requires locking the data of both the existing grad and incoming grad
        let mut existing_grad_data = existing_grad_lock.data_mut();
        let incoming_grad_data = incoming_grad.data();

        // Use ndarray's addition
        *existing_grad_data += &*incoming_grad_data; // Deref guards to get ArrayD

        Ok(())
    }


    /// Retrieves the gradient tensor, if it exists.
    /// Returns an `Option<Tensor>`. The returned Tensor is a clone
    /// (sharing data via Arc) of the internal gradient tensor.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.as_ref().map(|grad_arc| {
            let grad_mutex = grad_arc.lock().expect("Gradient Mutex poisoned");
            grad_mutex.clone() // Clone the Tensor struct (increments Arc counts)
        })
    }

    /// Initiates the backward pass to compute gradients.
    /// Starts from this tensor (usually a scalar loss).
    pub fn backward(&self) -> Result<(), TensorError> {
        if !self.requires_grad {
            return Err(TensorError::Generic(
                "Cannot call backward on tensor that does not require grad".to_string(),
            ));
        }

        if !self.is_scalar() {
             return Err(TensorError::Generic(
                "Backward can only be called on scalar tensors (for now)".to_string(),
            ));
            // TODO: Handle backward on non-scalar tensors (requires gradient argument)
        }

        // Initialize the gradient of the root tensor w.r.t itself to 1.0
        let initial_grad_data = ArrayD::ones(IxDyn(&self.shape)); // Should be shape [1] or [] for scalar
        let initial_grad = Tensor::new(initial_grad_data, false);

        // Use a Vec as a queue for topological sort traversal
        let mut queue: Vec<Arc<Mutex<AutogradContext>>> = Vec::new();
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new(); // Track visited nodes by context ID

        // Start the backward pass from the current tensor's context
        if let Some(ctx_arc) = &self.grad_context {
            // Set the initial gradient for the output tensor of this first context
             let mut ctx = ctx_arc.lock().expect("AutogradContext Mutex poisoned");
             ctx.set_output_gradient(initial_grad)?; // Set the initial dOutput

             // Add the context to the queue if not already visited
             if visited.insert(Arc::as_ptr(ctx_arc) as usize) { // Use pointer address as ID
                 queue.push(Arc::clone(ctx_arc));
             }

        } else if self.is_leaf && self.requires_grad {
             // If it's a leaf tensor, its gradient is the initial gradient.
             // No context to traverse back from, but need to store the grad.
             // We need mutable access to self here, which is tricky.
             // This suggests backward() might need `&mut self` or a different design.
             // For now, let's assume we handle leaf accumulation later.
             // A better approach: pass the initial gradient into the autograd::backward function.
             // Let's redesign slightly:
            return autograd::backward(self, initial_grad);

        } else {
             // If it's not a leaf and has no context, something is wrong (e.g., detached tensor)
             // Or it just doesn't require grad. Check made at the start.
             return Ok(()); // Nothing to do if no context and not a leaf requiring grad.
        }

        // Perform topological sort and backward pass (delegated)
        autograd::execute_backward_pass(queue, visited)

        // Alternative: Call a central backward function directly
        // autograd::backward(self, initial_grad)
    }

     /// Checks if the tensor represents a single scalar value.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty() || (self.shape.len() == 1 && self.shape[0] == 1) || self.size() == 1
        // More robust check considering shapes like [1, 1, 1]
        // self.size() == 1
    }

     /// Zeroes the gradient of the tensor if it exists.
     /// Commonly used by optimizers.
     pub fn zero_grad(&mut self) {
         // If a gradient exists, lock it and fill its data with zeros.
         if let Some(grad_arc) = &self.grad {
             let grad_tensor = grad_arc.lock().expect("Gradient Mutex poisoned");
             let mut grad_data = grad_tensor.data_mut();
             grad_data.fill(0.0 as TensorData);
         }
         // If no gradient exists yet, do nothing. It will be created with zeros
         // when needed by accumulate_grad if requires_grad is true.
     }
}

// --- Traits Implementation (Example: Basic Arithmetic) ---
// Using Rust's operator overloading traits
use std::ops::{Add, Sub, Mul}; // Div requires more care (type constraints)

// Tensor + Tensor
impl Add<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, other: &Tensor) -> Self::Output {
        ops::add(self, other) // Delegate to ops module
    }
}
impl Add<Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;
    fn add(self, other: Tensor) -> Self::Output {
        ops::add(&self, &other)
    }
}
// ... potentially add other combinations like &Tensor + Tensor, Tensor + &Tensor ...


// Tensor - Tensor
impl Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, other: &Tensor) -> Self::Output {
        ops::sub(self, other) // Delegate to ops module
    }
}
impl Sub<Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;
    fn sub(self, other: Tensor) -> Self::Output {
        ops::sub(&self, &other)
    }
}

// Tensor * Tensor (Element-wise multiplication)
impl Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, other: &Tensor) -> Self::Output {
        ops::mul(self, other) // Delegate to ops module
    }
}
impl Mul<Tensor> for Tensor {
    type Output = Result<Tensor, TensorError>;
    fn mul(self, other: Tensor) -> Self::Output {
        ops::mul(&self, &other)
    }
}

// TODO: Add Scalar operations (Tensor + f32, f32 + Tensor, etc.)
// TODO: Add Matrix Multiplication (matmul)
// TODO: Add Division, Power, Exp, Log, etc.
// TODO: Implement comparison operators


// --- Helper functions ---

/// Helper to create a tensor filled with zeros.
pub fn zeros(shape: &[usize], requires_grad: bool) -> Tensor {
    let data = ArrayD::zeros(IxDyn(shape));
    Tensor::new(data, requires_grad)
}

/// Helper to create a tensor filled with ones.
pub fn ones(shape: &[usize], requires_grad: bool) -> Tensor {
    let data = ArrayD::ones(IxDyn(shape));
    Tensor::new(data, requires_grad)
}

/// Helper to create a tensor with random values (uniform distribution).
/// Requires the `rand` crate feature.
#[cfg(feature = "rand")]
pub fn rand(shape: &[usize], requires_grad: bool) -> Tensor {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    let data = Array::random(IxDyn(shape), Uniform::new(0.0, 1.0));
    Tensor::new(data, requires_grad)
}

/// Helper to create a tensor with random values (standard normal distribution).
/// Requires the `rand` crate feature.
#[cfg(feature = "rand")]
pub fn randn(shape: &[usize], requires_grad: bool) -> Tensor {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;

    let data = Array::random(IxDyn(shape), StandardNormal);
    Tensor::new(data, requires_grad)
}

// Note: The PyO3 specific #[pymethods] block for the Tensor class
// will likely live in `src/bindings/python.rs` to keep core logic
// separate from binding details.