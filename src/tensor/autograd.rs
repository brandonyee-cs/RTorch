//! # Automatic Differentiation (Autograd)
//!
//! Implements the reverse-mode automatic differentiation engine.
//! It builds a computation graph and performs backpropagation to compute gradients.

use super::{Tensor, TensorError};
use std::sync::{Arc, Mutex, Weak}; // Weak pointers prevent reference cycles in the graph
use std::collections::{VecDeque, HashSet};
use std::fmt::Debug;

// --- Backward Operation Trait ---

/// Trait defining the backward pass for an operation.
/// Each operation (like Add, Mul, ReLU) will have a struct implementing this.
/// Needs `Send + Sync` to be usable across threads if parallelism is added later.
/// Needs `'static` because it's stored in `Box<dyn BackwardOp>`.
pub trait BackwardOp: Debug + Send + Sync + 'static {
    /// Computes the gradients with respect to the inputs of the operation.
    ///
    /// # Arguments
    /// * `inputs` - Weak references to the input Tensors of the original forward operation.
    ///              Needed to access input shapes or potentially values if required by the backward formula.
    /// * `output_grad` - The gradient flowing back into this operation's output tensor.
    ///
    /// # Returns
    /// * `Result<Vec<Tensor>, TensorError>` - A Vec containing the computed gradients
    ///                                        for each input tensor, in the same order as `inputs`.
    ///                                        Returns an error if calculation fails.
    fn backward(
        &self,
        inputs: &[Weak<Mutex<Tensor>>], // Use Weak<Mutex<Tensor>> to avoid cycles? Or just pass necessary info?
                                        // Let's pass owned Tensors from AutogradContext for simplicity now.
                                        // Cloning the Tensor struct is cheap (Arc increments).
        input_tensors: &[Tensor], // Pass clones of input tensors used in forward pass
        output_grad: &Tensor,
    ) -> Result<Vec<Tensor>, TensorError>;
}

// --- Autograd Context ---

/// Stores information needed for the backward pass of a single operation.
/// Associated with the output Tensor of that operation.
#[derive(Debug)]
pub struct AutogradContext {
    /// The specific backward logic for this operation.
    op: Box<dyn BackwardOp>,
    /// Input tensors to the forward operation. Stored to be passed to `op.backward`.
    /// We store full Tensors (clones) here. Cloning Tensor is cheap (Arc).
    /// This avoids issues with Weak pointers if inputs go out of scope prematurely,
    /// although it uses more memory. A Weak pointer approach is more memory efficient
    /// but requires careful handling of upgrades.
    inputs: Vec<Tensor>,
    /// References to the `AutogradContext` of the *input* tensors.
    /// Used for graph traversal during the backward pass.
    /// Weak pointers are crucial here to prevent reference cycles.
    /// If Tensor A depends on B, and B depends on A (cyclic graph - error usually),
    /// storing `Arc<Mutex<AutogradContext>>` directly would create a cycle.
    /// Storing `Weak<Mutex<AutogradContext>>` breaks the cycle.
    input_contexts: Vec<Option<Weak<Mutex<AutogradContext>>>>,

     /// Gradient accumulated for the output tensor associated with this context.
     /// This is the `dOutput` that gets passed into `op.backward`.
     /// Option used because it starts as None until a gradient flows back.
     accumulated_grad: Option<Tensor>,
}

impl AutogradContext {
    /// Creates a new AutogradContext.
    pub fn new(inputs: Vec<Tensor>, op: Box<dyn BackwardOp>) -> Self {
        let input_contexts = inputs
            .iter()
            .map(|t| t.grad_context.as_ref().map(Arc::downgrade)) // Get Weak refs to input contexts
            .collect();

        AutogradContext {
            op,
            inputs,
            input_contexts,
            accumulated_grad: None,
        }
    }

    /// Sets or accumulates the gradient for the output tensor of this operation.
    /// This function is called when gradients flow back to this node during backprop.
    pub fn set_output_gradient(&mut self, grad: Tensor) -> Result<(), TensorError> {
        if let Some(existing_grad) = &mut self.accumulated_grad {
            // Accumulate if gradient already exists (e.g., tensor used multiple times)
            // Need mutable access to the existing grad's data.
            // This requires the Tensor's data to be behind a lock, which it is (Arc<RwLock<...>>).
            let mut existing_data = existing_grad.data_mut();
            let incoming_data = grad.data();
            // TODO: Handle broadcasting if shapes don't match exactly but are compatible.
            if existing_data.shape() != incoming_data.shape() {
                return Err(TensorError::ShapeMismatch {
                     expected: existing_data.shape().to_vec(),
                     got: incoming_data.shape().to_vec()
                });
            }
            *existing_data += &*incoming_data;
        } else {
            // First time gradient arrives, just store it.
            self.accumulated_grad = Some(grad);
        }
        Ok(())
    }

    /// Executes the backward pass for this specific operation.
    /// Computes gradients w.r.t inputs and returns them along with the input contexts to visit next.
    pub fn execute_backward(&self) -> Result<(Vec<Tensor>, Vec<Option<Weak<Mutex<AutogradContext>>>>), TensorError> {
        match &self.accumulated_grad {
            Some(output_grad) => {
                // Calculate gradients for inputs using the specific backward op
                let input_grads = self.op.backward(&self.inputs, output_grad)?;

                // Ensure the number of gradients matches the number of inputs
                if input_grads.len() != self.inputs.len() {
                     return Err(TensorError::AutogradError(format!(
                        "Backward op {:?} produced {} gradients, but expected {}",
                        self.op, input_grads.len(), self.inputs.len()
                    )));
                }

                Ok((input_grads, self.input_contexts.clone()))
            }
            None => {
                // This should not happen if the topological sort is correct,
                // as a node is only processed after its output gradient is computed.
                // However, it could happen if backward is called on a non-scalar
                // without providing an initial gradient.
                 Err(TensorError::AutogradError(format!(
                    "Attempted to execute backward for op {:?} before output gradient was computed.", self.op
                )))
            }
        }
    }
}


// --- Main Backward Function ---

/// Performs the backward pass starting from a root tensor (usually the loss).
///
/// # Arguments
/// * `root_tensor`: The tensor to start backpropagation from (e.g., scalar loss).
/// * `initial_gradient`: The initial gradient to assign to the `root_tensor` (usually 1.0 for scalar loss).
pub fn backward(root_tensor: &Tensor, initial_gradient: Tensor) -> Result<(), TensorError> {
    if !root_tensor.requires_grad {
        // If the root doesn't require grad, no backward pass needed.
        return Ok(());
    }

    // Use VecDeque as a queue for nodes to visit (nodes = AutogradContexts)
    let mut queue: VecDeque<Arc<Mutex<AutogradContext>>> = VecDeque::new();
    // Use HashSet to keep track of visited nodes (contexts) to avoid redundant computation
    // Store the *pointer address* of the Arc<Mutex<...>> as a unique ID for the context.
    let mut visited: HashSet<usize> = HashSet::new();

    // --- Initialization ---
    // If the root tensor is the result of an operation, start from its context.
    if let Some(root_ctx_arc) = &root_tensor.grad_context {
        // Set the initial gradient for the root tensor's output node
        { // Scope for mutex lock
            let mut root_ctx = root_ctx_arc.lock().expect("AutogradContext Mutex poisoned");
            root_ctx.set_output_gradient(initial_gradient)?;
        } // Lock released here

        // Add the root context to the queue if not already visited
        let ctx_ptr = Arc::as_ptr(root_ctx_arc) as usize;
        if visited.insert(ctx_ptr) {
             queue.push_back(Arc::clone(root_ctx_arc));
        }
    }
    // If the root tensor is a leaf node, accumulate the initial gradient directly onto it.
    // This requires mutable access to the root tensor, which isn't available with `&Tensor`.
    // This indicates a design challenge. A common solution is:
    // 1. Make `backward` take `&mut self` (changes Tensor API significantly).
    // 2. Pass a mutable reference *only if* it's a leaf node.
    // 3. Have `execute_backward_pass` handle leaf node gradient accumulation separately.
    // Let's go with option 3: Accumulate gradients onto leaf nodes *after* they are computed
    // by the backward pass of the operation that uses them as input.
    // So, if the root is a leaf, there's no context to queue, the process ends unless
    // `initial_gradient` needs to be stored.
    else if root_tensor.is_leaf {
        // If the root tensor itself is a leaf and requires grad,
        // its gradient is the initial gradient. We need to store it.
        // This needs mutable access. Let's assume `accumulate_grad` can be called
        // externally or refactor how leaf grads are handled.
        // For now, we might need to *temporarily* allow mutable access here,
        // or redesign Tensor storage/access slightly.
        // A simpler approach for now: The caller should handle the leaf case?
        // Or, maybe the Tensor needs an internal `accumulate_grad_internal` method
        // that can be called via the Mutex/RwLock.

        // Let's try calling accumulate_grad directly, assuming we *can* get mut access somehow.
        // THIS IS A DESIGN FLAW IN THE CURRENT SIGNATURE - backward needs mut access
        // or the Tensor needs internal mutability methods exposed carefully.
        // For demonstration, let's assume we can:
        // root_tensor.accumulate_grad(initial_gradient)?; // THIS WON'T COMPILE with &Tensor

        // --- Revised approach: ---
        // The `execute_backward_pass` function will handle accumulating gradients
        // into the `grad` field of the *input tensors* (which might be leaves).

        // If the root is a leaf, AND requires grad, we still need to store the initial grad.
        // We will handle this case within `execute_backward_pass` when processing nodes
        // where one of the *inputs* is this root leaf node. This seems overly complex.

        // --- Simplest fix: ---
        // Let the `Tensor` struct provide a method like `init_grad` or modify `accumulate_grad`
        // to handle initialization. Let's assume `accumulate_grad` handles the Option internally.

        // *** We need `&mut Tensor` conceptually here to call `accumulate_grad`. ***
        // Let's PRETEND we have mutable access for the logic flow.
        // In reality, this needs a redesign (e.g., `backward` takes `&mut self`
        // or uses interior mutability pattern more aggressively).

        // --- Alternative Design: ---
        // Let Tensor's `backward` method take `&self`. Inside, it locks its own gradient
        // field for writing if it's a leaf node.

        // Let's stick to the queue approach. If root is leaf, queue is empty,
        // `execute_backward_pass` won't run. We need to manually set the root's grad.
        if root_tensor.is_leaf && root_tensor.requires_grad {
             // This is the tricky part without &mut self.
             // We rely on Tensor's internal locking to allow this modification.
             // Find a way to call accumulate_grad on root_tensor.
             // We might need a helper function on Tensor or modify accumulate_grad access.

             // Assuming Tensor has a method like `_accumulate_grad_internal` accessible here:
             // root_tensor._accumulate_grad_internal(initial_gradient)?;
             // Or maybe Tensor::accumulate_grad needs to be redesigned slightly.
             // Let's assume `root_tensor.grad` uses Arc<Mutex<Tensor>> and we can lock and update it.

             // Lock the gradient field of the root tensor
             let grad_arc = root_tensor.grad.get_or_insert_with(|| {
                 // Initialize gradient field if it doesn't exist
                 let zeros = ArrayD::zeros(IxDyn(root_tensor.shape()));
                 Arc::new(Mutex::new(Tensor::new(zeros, false)))
             }).clone(); // Clone the Arc to work with it

            let mut grad_tensor_mutex = grad_arc.lock().expect("Gradient Mutex poisoned");
            // Now grad_tensor_mutex is a MutexGuard<Tensor> for the gradient tensor

            // Check shapes before accumulating
             if grad_tensor_mutex.shape() != initial_gradient.shape() {
                 return Err(TensorError::ShapeMismatch {
                     expected: grad_tensor_mutex.shape().to_vec(),
                     got: initial_gradient.shape().to_vec(),
                 });
             }

             // Lock the data of the gradient tensor and the initial gradient
             let mut grad_data = grad_tensor_mutex.data_mut();
             let initial_gradient_data = initial_gradient.data();

             // Perform accumulation
             *grad_data += &*initial_gradient_data;

             // Root is a leaf, no further graph traversal needed from here.
             return Ok(());
        } else {
            // Root requires grad but is not a leaf and has no context? Invalid state.
             return Err(TensorError::AutogradError(
                "Root tensor requires grad but is not a leaf and has no grad_context.".to_string()
            ));
        }
    } else {
        // Root doesn't require grad, nothing to do.
        return Ok(());
    }


    // --- Topological Sort and Backward Execution ---
    execute_backward_pass(queue, visited)
}


/// Executes the backward pass using a queue-based topological sort.
pub(crate) fn execute_backward_pass(
    mut queue: VecDeque<Arc<Mutex<AutogradContext>>>,
    mut visited: HashSet<usize>,
) -> Result<(), TensorError> {

    while let Some(ctx_arc) = queue.pop_front() {
        let (input_grads, next_ctx_weak_refs) = {
             // Lock the current context to execute its backward op
             let ctx = ctx_arc.lock().expect("AutogradContext Mutex poisoned");
             ctx.execute_backward()? // This calls the op's specific backward method
        }; // Lock released here


        // Propagate gradients to the inputs of the current operation
        for (i, input_grad) in input_grads.into_iter().enumerate() {
            // Get the corresponding input tensor from the context
            // We stored clones in the context, so this is safe.
            let input_tensor = &ctx_arc.lock().expect("AutogradContext Mutex poisoned").inputs[i]; // Read lock ok

            if input_tensor.requires_grad {
                 // Check if this input tensor is a leaf node or the result of another op
                 if let Some(next_ctx_weak) = &next_ctx_weak_refs[i] {
                     // Input tensor is the result of another operation.
                     // Upgrade the Weak pointer to an Arc.
                     if let Some(next_ctx_arc) = next_ctx_weak.upgrade() {
                         // Accumulate the computed gradient into the *output gradient*
                         // of the *next* context (which corresponds to input_tensor).
                         { // Scope for mutex lock
                            let mut next_ctx = next_ctx_arc.lock().expect("AutogradContext Mutex poisoned");
                            next_ctx.set_output_gradient(input_grad)?;
                         } // Lock released here

                         // Add the next context to the queue if not visited
                         let next_ctx_ptr = Arc::as_ptr(&next_ctx_arc) as usize;
                         if visited.insert(next_ctx_ptr) {
                             queue.push_back(next_ctx_arc);
                         }
                     } else {
                         // Weak pointer expired. This might indicate the tensor it points
                         // to went out of scope, potentially an issue in graph handling logic.
                         // For now, we can ignore it, assuming it means that part of the
                         // graph is no longer needed. Or return an error.
                          eprintln!("Warning: Weak pointer upgrade failed during backward pass. Graph structure might be incomplete.");
                          // return Err(TensorError::AutogradError("Weak pointer upgrade failed".to_string()));
                     }
                 } else {
                     // Input tensor is a leaf node (or detached).
                     // Accumulate the gradient directly onto the input tensor's `grad` field.
                     // This requires mutable access to the input tensor's gradient field.
                     // We rely on the internal Mutex/RwLock within the Tensor's grad field.

                     // Lock the gradient field of the input tensor
                     let grad_arc = input_tensor.grad.get_or_insert_with(|| {
                         let zeros = ArrayD::zeros(IxDyn(input_tensor.shape()));
                         Arc::new(Mutex::new(Tensor::new(zeros, false)))
                     }).clone(); // Clone Arc

                     let mut grad_tensor_mutex = grad_arc.lock().expect("Gradient Mutex poisoned");

                      // Check shapes before accumulating
                     if grad_tensor_mutex.shape() != input_grad.shape() {
                         return Err(TensorError::ShapeMismatch {
                             expected: grad_tensor_mutex.shape().to_vec(),
                             got: input_grad.shape().to_vec(),
                         });
                     }

                     // Lock data and accumulate
                     let mut grad_data = grad_tensor_mutex.data_mut();
                     let incoming_grad_data = input_grad.data();
                     *grad_data += &*incoming_grad_data;
                 }
            }
            // If input_tensor.requires_grad is false, we simply drop the computed gradient.
        }
    }

    Ok(())
}

// --- Concrete BackwardOp Implementations ---
// Move these into a submodule `op_abstractions`? Yes, let's do that.

pub mod op_abstractions {
    use super::*;
    use ndarray::{ArrayD, Axis, IxDyn};
    use crate::tensor::ops; // Need access to forward ops potentially

    // ---- Add ----
    #[derive(Debug)]
    pub struct AddBackward;
    impl BackwardOp for AddBackward {
        fn backward(
            &self,
            inputs: &[Tensor],
            output_grad: &Tensor,
        ) -> Result<Vec<Tensor>, TensorError> {
            // Grad for input A = dOutput * 1
            // Grad for input B = dOutput * 1
            // Need to handle broadcasting backward: if an input was broadcasted,
            // the gradient needs to be summed along the broadcasted dimensions.

            let grad_a = output_grad.clone(); // Simple case: no broadcasting
            let grad_b = output_grad.clone(); // Simple case: no broadcasting

             // TODO: Handle broadcasting backward for Add
            let input_a_shape = inputs[0].shape();
            let input_b_shape = inputs[1].shape();
            let output_shape = output_grad.shape();

            let final_grad_a = unbroadcast(grad_a, input_a_shape, output_shape)?;
            let final_grad_b = unbroadcast(grad_b, input_b_shape, output_shape)?;


            Ok(vec![final_grad_a, final_grad_b])
        }
    }

     // ---- Mul ----
     #[derive(Debug)]
     pub struct MulBackward;
     impl BackwardOp for MulBackward {
         fn backward(
             &self,
             inputs: &[Tensor],
             output_grad: &Tensor,
         ) -> Result<Vec<Tensor>, TensorError> {
             // Grad for input A = dOutput * input B
             // Grad for input B = dOutput * input A
             let input_a = &inputs[0];
             let input_b = &inputs[1];

             let grad_a = ops::mul(output_grad, input_b)?;
             let grad_b = ops::mul(output_grad, input_a)?;

             // TODO: Handle broadcasting backward for Mul
             let final_grad_a = unbroadcast(grad_a, input_a.shape(), output_grad.shape())?;
             let final_grad_b = unbroadcast(grad_b, input_b.shape(), output_grad.shape())?;


             Ok(vec![final_grad_a, final_grad_b])
         }
     }

      // ---- Sum ----
      #[derive(Debug)]
      pub struct SumBackward {
        input_shape: Vec<usize>, // Need original shape to broadcast gradient back
      }
      impl SumBackward {
        pub fn new(input_shape: Vec<usize>) -> Self { Self { input_shape } }
      }
      impl BackwardOp for SumBackward {
          fn backward(
              &self,
              _inputs: &[Tensor], // Sum only has one input
              output_grad: &Tensor, // Should be scalar or reduced shape
          ) -> Result<Vec<Tensor>, TensorError> {
              // Grad for input = dOutput broadcasted to the original input shape.
              // If output_grad is scalar, broadcast it.
              if !output_grad.is_scalar() {
                 return Err(TensorError::AutogradError("Sum backward expects a scalar gradient.".to_string()));
                 // TODO: Handle sum along axis later - output_grad won't be scalar then.
              }
              let scalar_grad_val = output_grad.data().first().cloned().unwrap_or(0.0); // Get the scalar value
              let grad_data = ArrayD::from_elem(IxDyn(&self.input_shape), scalar_grad_val);
              let input_grad = Tensor::new(grad_data, false); // Gradient tensor doesn't need grad

              Ok(vec![input_grad])
          }
      }

       // ---- MatMul ----
      #[derive(Debug)]
      pub struct MatMulBackward;
      impl BackwardOp for MatMulBackward {
          fn backward(
              &self,
              inputs: &[Tensor],
              output_grad: &Tensor,
          ) -> Result<Vec<Tensor>, TensorError> {
              // A @ B = C
              // Grad A = dC @ B.T
              // Grad B = A.T @ dC
              // Need to handle vector/matrix cases correctly
              let input_a = &inputs[0];
              let input_b = &inputs[1];

              // Calculate Grad A = dC @ B.T
              // TODO: Need transpose operation
              let input_b_t = transpose(input_b)?; // Assuming transpose exists
              let grad_a = ops::matmul(output_grad, &input_b_t)?;

              // Calculate Grad B = A.T @ dC
               // TODO: Need transpose operation
              let input_a_t = transpose(input_a)?; // Assuming transpose exists
              let grad_b = ops::matmul(&input_a_t, output_grad)?;

              // TODO: Handle broadcasting/dimension mismatches carefully for matmul backward
              // e.g. if A=[m,k], B=[k], C=[m], then dC=[m].
              // grad_a = dC [m] @ B.T [1,k] ? -> needs outer product? [m, k]
              // grad_b = A.T [k,m] @ dC [m] -> [k]

              Ok(vec![grad_a, grad_b])
          }
      }

        // ---- ReLU ----
      #[derive(Debug)]
      pub struct ReluBackward;
      impl BackwardOp for ReluBackward {
          fn backward(
              &self,
              inputs: &[Tensor],
              output_grad: &Tensor,
          ) -> Result<Vec<Tensor>, TensorError> {
              let input_a = &inputs[0];
              // Grad Input = dOutput * (Input > 0 ? 1 : 0)
              let input_data = input_a.data();
              let positive_mask = input_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
              let positive_mask_tensor = Tensor::new(positive_mask, false);

              let input_grad = ops::mul(output_grad, &positive_mask_tensor)?;

              Ok(vec![input_grad])
          }
      }

      // ---- Reshape ----
      #[derive(Debug)]
      pub struct ReshapeBackward {
          original_shape: Vec<usize>,
      }
      impl ReshapeBackward {
        pub fn new(original_shape: Vec<usize>) -> Self { Self { original_shape } }
      }
      impl BackwardOp for ReshapeBackward {
          fn backward(
              &self,
              _inputs: &[Tensor],
              output_grad: &Tensor,
          ) -> Result<Vec<Tensor>, TensorError> {
              // Grad Input = dOutput reshaped back to original shape
              let input_grad = ops::reshape(output_grad, &self.original_shape)?;
              Ok(vec![input_grad])
          }
      }

      // --- Helper Functions ---

      /// Placeholder for transpose operation needed for MatMulBackward
      fn transpose(tensor: &Tensor) -> Result<Tensor, TensorError> {
           // TODO: Implement actual transpose logic using ndarray
          let data = tensor.data();
          if data.ndim() == 2 {
              let transposed_data = data.t().into_owned().into_dyn(); // Simple 2D transpose
              // Transpose should ideally be autograd-aware if inputs require grad,
              // but here we just need it for intermediate calculation.
              Ok(Tensor::new(transposed_data, false)) // Create detached tensor
          } else if data.ndim() == 1 {
               // Transposing a 1D vector [k] often means reshaping to [1, k] or [k, 1]
               // For A@B.T where B is [k], B.T might be [1, k]?
               // For A.T@dC where A is [k], A.T might be [k, 1]?
               // This depends on the specific matmul case.
               // Let's assume for matmul backward, we need to add a dimension.
               // This needs refinement based on exact matmul rules.
               Err(TensorError::Generic("Transpose for 1D vectors in backward not fully implemented".to_string()))
          } else {
              Err(TensorError::Generic("Transpose only implemented for 2D tensors currently".to_string()))
          }
      }

      /// Helper to reverse broadcasting. Sums gradient along broadcasted dimensions.
      fn unbroadcast(grad: Tensor, target_shape: &[usize], broadcasted_shape: &[usize]) -> Result<Tensor, TensorError> {
          if target_shape == broadcasted_shape {
              return Ok(grad); // No broadcasting occurred
          }

          // Example: target=[k], broadcasted=[m, k] -> sum along axis 0
          // Example: target=[] (scalar), broadcasted=[m, k] -> sum along all axes
          // Example: target=[m, 1], broadcasted=[m, k] -> sum along axis 1, keepdim=True

          // This is complex. Let's implement simple scalar -> tensor unbroadcasting first.
          if target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == 1) {
              // Target was scalar, sum the gradient over all dimensions
              return ops::sum(&grad);
          }

          // Implement summing along specific axes needed to reduce grad shape
          // back to target_shape. This requires careful axis alignment.
          // E.g., if target=[k] and broadcasted=[m,k], we need to sum axis 0.
          // If target=[m,1] and broadcasted=[m,k], we need to sum axis 1 and keepdim.

          // --- Simplified approach for now ---
          // Check if one is scalar version of the other
          if target_shape.iter().product::<usize>() == 1 && broadcasted_shape.iter().product::<usize>() > 1 {
              // Target was scalar, sum gradient
               return ops::sum(&grad);
          }
          if broadcasted_shape.iter().product::<usize>() == 1 && target_shape.iter().product::<usize>() > 1 {
                // Gradient is scalar, need to broadcast it back? No, this case shouldn't happen here.
                // The gradient should have the broadcasted shape.
                return Err(TensorError::AutogradError("Invalid unbroadcast scenario: scalar gradient for non-scalar target.".to_string()));
          }

          // Fallback: If shapes differ but not scalar broadcast, assume element-wise op
          // where shapes must match (handled by forward op error).
          // If we reach here with different shapes, it means broadcasting happened,
          // but we haven't implemented the logic to reverse it yet.
          // For now, return error or the unmodified grad (which might be wrong).
          // Returning error is safer.
           if target_shape != broadcasted_shape {
                 println!("Warning: Unbroadcast logic not fully implemented for shapes {:?} -> {:?}", broadcasted_shape, target_shape);
                 // For ops like Add/Mul, if target_shape=[k] and broadcast_shape=[m,k]
                 // grad has shape [m,k]. We need grad.sum(axis=0) -> [k].
                 // ndarray sum_axis can do this.
                 let ndims_diff = broadcasted_shape.len() as i32 - target_shape.len() as i32;
                 if ndims_diff > 0 {
                    // Summing over leading dimensions
                    let mut current_grad = grad.data_clone(); // Clone data to modify
                    for i in 0..ndims_diff as usize {
                         current_grad = current_grad.sum_axis(Axis(0)); // Sum first dimension repeatedly
                    }
                     // Now check trailing dimensions where size might be 1 in target
                     let mut axes_to_sum = Vec::new();
                    for i in 0..target_shape.len() {
                        if target_shape[i] == 1 && current_grad.shape()[i + ndims_diff as usize] > 1 {
                           axes_to_sum.push(i + ndims_diff as usize);
                        } else if target_shape[i] != current_grad.shape()[i + ndims_diff as usize] {
                             // Shapes mismatch unexpectedly after handling leading dims
                              return Err(TensorError::AutogradError(format!("Cannot unbroadcast shape {:?} to {:?}", broadcasted_shape, target_shape)));
                         }
                     }
                    // Perform summing on remaining axes
                     for axis_idx in axes_to_sum.into_iter().rev() { // Sum from highest axis index first
                        current_grad = current_grad.sum_axis(Axis(axis_idx));
                        // Need keep_dims equivalent. Insert axis back.
                        current_grad.insert_axis_inplace(Axis(axis_idx));
                    }

                    // Final shape check - might need reshape if keep_dims wasn't perfect
                    if current_grad.shape() != target_shape {
                         if current_grad.shape().iter().product::<usize>() == target_shape.iter().product() {
                              current_grad = current_grad.into_shape(IxDyn(target_shape)).map_err(TensorError::NdarrayError)?;
                         } else {
                              return Err(TensorError::AutogradError(format!("Unbroadcast result shape {:?} doesn't match target {:?}", current_grad.shape(), target_shape)));
                         }
                    }

                     return Ok(Tensor::new(current_grad, false)); // Return new tensor with summed grad


                 } else {
                     // Other broadcasting cases (e.g. [m, 1] to [m, k])
                    return Err(TensorError::AutogradError(format!("Unimplemented unbroadcast case: {:?} -> {:?}", broadcasted_shape, target_shape)));
                 }
           }


          Ok(grad) // Should be unreachable if shapes checked properly before
      }

} // end mod op_abstractions

// Re-export for use in ops.rs etc.
pub use op_abstractions::*;