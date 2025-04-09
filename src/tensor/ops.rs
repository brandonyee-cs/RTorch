//! # Tensor Operations
//!
//! Implements mathematical operations for Tensors, including autograd support.

use super::{Tensor, TensorData, TensorError};
use super::autograd::{AutogradContext, GradFn, op_abstractions::{BackwardOp, AddBackward, MulBackward, MatMulBackward, SumBackward, ReluBackward}};
use ndarray::{ArrayD, Axis, IxDyn,linalg::Dot}; // Import linalg::Dot for matmul
use std::sync::{Arc, Mutex};

// --- Helper Function for Autograd Setup ---

/// Creates a new tensor resulting from an operation, setting up autograd context if needed.
fn create_op_result(
    result_data: ArrayD<TensorData>,
    inputs: Vec<Tensor>, // The input tensors to the operation
    backward_op: Box<dyn BackwardOp>, // The specific backward logic
) -> Result<Tensor, TensorError> {
    // Determine if the output tensor should require gradient computation
    let requires_grad = inputs.iter().any(|t| t.requires_grad);

    if requires_grad {
        // Create autograd context only if needed
        let grad_context = Arc::new(Mutex::new(AutogradContext::new(inputs, backward_op)));
        Ok(Tensor::from_op(result_data, grad_context, true))
    } else {
        // If no input requires grad, the output doesn't either, and has no context
        Ok(Tensor::new(result_data, false))
    }
}

// --- Arithmetic Operations ---

/// Element-wise addition of two tensors.
/// Checks if tensor shapes can be broadcast together following NumPy broadcasting rules
pub fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
    // Iterate from the right (least significant dimensions)
    let mut i1 = shape1.len() as isize - 1;
    let mut i2 = shape2.len() as isize - 1;
    
    while i1 >= 0 && i2 >= 0 {
        let s1 = shape1[i1 as usize];
        let s2 = shape2[i2 as usize];
        
        // Dimensions must be equal or one of them must be 1
        if s1 != s2 && s1 != 1 && s2 != 1 {
            return false;
        }
        
        i1 -= 1;
        i2 -= 1;
    }
    
    true
}

/// Determines the output shape after broadcasting two tensors
pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, TensorError> {
    if !can_broadcast(shape1, shape2) {
        return Err(TensorError::IncompatibleShapes {
            op: "broadcast".to_string(),
            shape1: shape1.to_vec(),
            shape2: shape2.to_vec(),
        });
    }
    
    // Calculate the output shape
    let max_dims = std::cmp::max(shape1.len(), shape2.len());
    let mut output_shape = Vec::with_capacity(max_dims);
    
    // Pad the shorter shape with 1s on the left
    let padded_shape1 = pad_shape_left(shape1, max_dims);
    let padded_shape2 = pad_shape_left(shape2, max_dims);
    
    // For each dimension, take the maximum of the two shapes
    for i in 0..max_dims {
        output_shape.push(std::cmp::max(padded_shape1[i], padded_shape2[i]));
    }
    
    Ok(output_shape)
}

/// Pads a shape with 1s on the left to match the target length
fn pad_shape_left(shape: &[usize], target_len: usize) -> Vec<usize> {
    if shape.len() >= target_len {
        return shape.to_vec();
    }
    
    let mut padded = Vec::with_capacity(target_len);
    // Add padding 1s
    for _ in 0..(target_len - shape.len()) {
        padded.push(1);
    }
    // Add original shape
    padded.extend_from_slice(shape);
    
    padded
}

// Modified add function with broadcasting support
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    let b_data = b.data();
    
    if a.shape() == b.shape() {
        // Standard element-wise addition
        let result_data = &*a_data + &*b_data;
        create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(AddBackward))
    } else if a.is_scalar() {
        // Scalar + Tensor
        let scalar_val = a_data.first().ok_or_else(|| TensorError::Generic("Empty scalar tensor".into()))?;
        let result_data = &*b_data + *scalar_val;
        create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(AddBackward))
    } else if b.is_scalar() {
        // Tensor + Scalar
        let scalar_val = b_data.first().ok_or_else(|| TensorError::Generic("Empty scalar tensor".into()))?;
        let result_data = &*a_data + *scalar_val;
        create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(AddBackward))
    } else {
        // Try broadcasting
        let output_shape = broadcast_shapes(a.shape(), b.shape())?;
        
        // Broadcast tensors to output shape and add
        // This is a simplified implementation - a full one would use ndarray's broadcast capabilities
        let a_broadcast = broadcast_tensor(a, &output_shape)?;
        let b_broadcast = broadcast_tensor(b, &output_shape)?;
        
        // Now add the broadcasted tensors
        let a_broadcast_data = a_broadcast.data();
        let b_broadcast_data = b_broadcast.data();
        let result_data = &*a_broadcast_data + &*b_broadcast_data;
        
        create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(AddBackward))
    }
}

/// Broadcasts a tensor to the target shape
fn broadcast_tensor(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor, TensorError> {
    // Check if broadcasting is possible
    if !can_broadcast(tensor.shape(), target_shape) {
        return Err(TensorError::IncompatibleShapes {
            op: "broadcast_tensor".to_string(),
            shape1: tensor.shape().to_vec(),
            shape2: target_shape.to_vec(),
        });
    }
    
    // If shapes already match, return the tensor as is
    if tensor.shape() == target_shape {
        return Ok(tensor.clone());
    }
    
    // Get the data
    let tensor_data = tensor.data();
    
    // Simplified broadcast for common cases
    // For a complete implementation, we would use ndarray's broadcasting features
    
    // Create a new array with the target shape
    let mut result_data = ndarray::ArrayD::<TensorData>::zeros(ndarray::IxDyn(target_shape));
    
    // For simplicity, we'll implement a basic version here
    // This is inefficient but works for demonstration
    if tensor.is_scalar() {
        // Broadcasting a scalar is easy
        let value = tensor_data.first().unwrap();
        result_data.fill(*value);
    } else {
        // For more complex broadcasting, we should use ndarray features
        // This is a placeholder - a real implementation would use ndarray's broadcast mechanism
        return Err(TensorError::Generic("Complex broadcasting not fully implemented".to_string()));
    }
    
    Ok(Tensor::new(result_data, tensor.requires_grad))
}


/// Element-wise subtraction of two tensors (a - b).
/// Supports basic broadcasting similar to add.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
     let a_data = a.data();
     let b_data = b.data();

     // Basic Broadcasting check (simplified)
     let result_data = if a.shape() == b.shape() {
         &*a_data - &*b_data // ndarray handles element-wise sub
     } else if a.is_scalar() && !b.is_scalar() {
        let scalar_val = a_data.first().ok_or_else(|| TensorError::Generic("Empty scalar tensor".into()))?;
        // Need to be careful with order: scalar - tensor_element
        let scalar_array = ArrayD::from_elem(b.shape(), *scalar_val);
        scalar_array - &*b_data
     } else if !a.is_scalar() && b.is_scalar() {
         let scalar_val = b_data.first().ok_or_else(|| TensorError::Generic("Empty scalar tensor".into()))?;
         &*a_data - *scalar_val // ndarray broadcasts scalar
     }
      else {
         // TODO: Implement more general broadcasting rules
         return Err(TensorError::IncompatibleShapes {
             op: "sub".to_string(),
             shape1: a.shape().to_vec(),
             shape2: b.shape().to_vec(),
         });
     };

     // For autograd: Subtraction a - b is like a + (-1 * b)
     // The gradient w.r.t a is 1 * dOutput.
     // The gradient w.r.t b is -1 * dOutput.
     // We can reuse AddBackward logic if we negate b first, or create a specific SubBackward.
     // Let's create a SubBackward for clarity.
     // TODO: Implement SubBackward in autograd::op_abstractions
     // create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(SubBackward))
     // For now, let's fake it using Mul and Add (less efficient)
     let neg_b = mul_scalar(b, -1.0)?;
     add(a, &neg_b) // This will build the correct graph structure, though less direct
}


/// Element-wise multiplication of two tensors.
/// Supports basic broadcasting similar to add.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    let b_data = b.data();

    // Basic Broadcasting check (simplified)
    let result_data = if a.shape() == b.shape() {
        &*a_data * &*b_data // ndarray handles element-wise mul
    } else if a.is_scalar() && !b.is_scalar() {
        let scalar_val = a_data.first().ok_or_else(|| TensorError::Generic("Empty scalar tensor".into()))?;
        &*b_data * *scalar_val // ndarray broadcasts scalar
    } else if !a.is_scalar() && b.is_scalar() {
         let scalar_val = b_data.first().ok_or_else(|| TensorError::Generic("Empty scalar tensor".into()))?;
         &*a_data * *scalar_val // ndarray broadcasts scalar
    }
     else {
        // TODO: Implement more general broadcasting rules
        return Err(TensorError::IncompatibleShapes {
            op: "mul".to_string(),
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        });
    };

    create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(MulBackward))
}

/// Multiply a tensor by a scalar.
pub fn mul_scalar(a: &Tensor, scalar: TensorData) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    let result_data = &*a_data * scalar;

    // Treat scalar as a constant, so only `a` contributes to gradient requirement
    let requires_grad = a.requires_grad;

    if requires_grad {
        // Need a specialized backward op for scalar multiplication
        // TODO: Implement MulScalarBackward in autograd::op_abstractions
        // grad_a = dOutput * scalar
        // let backward_op = Box::new(MulScalarBackward::new(scalar));
        // For now, create a constant tensor for the scalar and use regular mul
        let scalar_tensor_data = ArrayD::from_elem(IxDyn(&[]), scalar); // 0-dim array
        let scalar_tensor = Tensor::new(scalar_tensor_data, false); // Scalar constant doesn't require grad
        mul(a, &scalar_tensor) // Use the existing mul operation
    } else {
        Ok(Tensor::new(result_data, false))
    }
}

// TODO: Implement Division (div, div_scalar) - need DivBackward
// pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> { ... }


// --- Matrix Operations ---

/// Matrix multiplication of two tensors.
/// Handles 1D (vector) and 2D (matrix) cases.
/// (Batch matmul NYI).
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    let b_data = b.data();
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    // --- Shape Validation ---
    // Case 1: Vector dot Vector ([n] @ [n] -> scalar)
    if a_ndim == 1 && b_ndim == 1 {
        if a.shape()[0] != b.shape()[0] {
            return Err(TensorError::IncompatibleShapes {
                op: "matmul (vec-vec)".to_string(),
                shape1: a.shape().to_vec(),
                shape2: b.shape().to_vec(),
            });
        }
        // Use ndarray's dot product for 1D arrays
        let result_scalar = a_data.dot(&*b_data);
        let result_data = ArrayD::from_elem(IxDyn(&[]), result_scalar); // 0-dim array for scalar result
         create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(MatMulBackward))

    // Case 2: Matrix dot Vector ([m, k] @ [k] -> [m])
    } else if a_ndim == 2 && b_ndim == 1 {
        if a.shape()[1] != b.shape()[0] {
            return Err(TensorError::IncompatibleShapes {
                op: "matmul (mat-vec)".to_string(),
                shape1: a.shape().to_vec(),
                shape2: b.shape().to_vec(),
            });
        }
        // Perform dot product. Result is 1D.
        let a_view = a_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_view = b_data.view().into_dimensionality::<ndarray::Ix1>().unwrap();
        let result = a_view.dot(&b_view); // Result shape [m]
        let result_data = result.into_dyn(); // Convert back to ArrayD
         create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(MatMulBackward))

    // Case 3: Vector dot Matrix ([k] @ [k, n] -> [n]) - Less common, transpose b first? Or handle directly?
    } else if a_ndim == 1 && b_ndim == 2 {
         if a.shape()[0] != b.shape()[0] {
             return Err(TensorError::IncompatibleShapes {
                 op: "matmul (vec-mat)".to_string(),
                 shape1: a.shape().to_vec(),
                 shape2: b.shape().to_vec(),
             });
         }
         // Perform dot product. Result is 1D.
         let a_view = a_data.view().into_dimensionality::<ndarray::Ix1>().unwrap();
         let b_view = b_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
         // ndarray doesn't directly support 1D @ 2D? Let's do (1, k) @ (k, n) -> (1, n) and reshape
         let a_reshaped = a_view.insert_axis(Axis(0)); // Shape [1, k]
         let result_2d = a_reshaped.dot(&b_view); // Result shape [1, n]
         let result = result_2d.remove_axis(Axis(0)); // Shape [n]
         let result_data = result.into_dyn();
         create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(MatMulBackward))

    // Case 4: Matrix dot Matrix ([m, k] @ [k, n] -> [m, n])
    } else if a_ndim == 2 && b_ndim == 2 {
        if a.shape()[1] != b.shape()[0] {
            return Err(TensorError::IncompatibleShapes {
                op: "matmul (mat-mat)".to_string(),
                shape1: a.shape().to_vec(),
                shape2: b.shape().to_vec(),
            });
        }
        // Perform dot product. Result is 2D.
        let a_view = a_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_view = b_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = a_view.dot(&b_view); // Result shape [m, n]
        let result_data = result.into_dyn(); // Convert back to ArrayD
         create_op_result(result_data, vec![a.clone(), b.clone()], Box::new(MatMulBackward))

    } else {
        // TODO: Handle batched matmul ([b, m, k] @ [b, k, n] -> [b, m, n])
        // TODO: Handle broadcasting for matmul (e.g., [m, k] @ [k] -> [m]) - Covered above
        return Err(TensorError::Generic(format!(
            "Matmul not implemented for shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
}


// --- Activation Functions ---

/// Rectified Linear Unit (ReLU) activation function.
/// max(0, x) applied element-wise.
pub fn relu(a: &Tensor) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    // Apply ReLU: x.mapv(|v| v.max(0.0))
    let result_data = a_data.mapv(|val| val.max(0.0 as TensorData));

    create_op_result(result_data, vec![a.clone()], Box::new(ReluBackward))
}

// Implementation for sigmoid function
pub fn sigmoid(a: &Tensor) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    // Apply sigmoid: 1 / (1 + exp(-x))
    let result_data = a_data.mapv(|val| 1.0 / (1.0 + (-val).exp()));
    
    // Use a proper SigmoidBackward implementation for autograd
    use super::autograd::op_abstractions::SigmoidBackward;
    create_op_result(result_data, vec![a.clone()], Box::new(SigmoidBackward))
}

// Implementation for tanh function
pub fn tanh(a: &Tensor) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    // Apply tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    let result_data = a_data.mapv(|val| val.tanh());
    
    // Use a proper TanhBackward implementation for autograd
    use super::autograd::op_abstractions::TanhBackward;
    create_op_result(result_data, vec![a.clone()], Box::new(TanhBackward))
}

// ---- Sigmoid ----
#[derive(Debug)]
pub struct SigmoidBackward;
impl BackwardOp for SigmoidBackward {
    fn backward(
        &self,
        inputs: &[Tensor],
        output_grad: &Tensor,
    ) -> Result<Vec<Tensor>, TensorError> {
        let input = &inputs[0];
        let sigmoid_output = ops::sigmoid(input)?;
        
        // Gradient: grad_input = grad_output * sigmoid(input) * (1 - sigmoid(input))
        let one_minus_sigmoid = ops::sub_scalar(&sigmoid_output, 1.0)?;
        let one_minus_sigmoid_neg = ops::mul_scalar(&one_minus_sigmoid, -1.0)?;
        let sigmoid_grad = ops::mul(&sigmoid_output, &one_minus_sigmoid_neg)?;
        let input_grad = ops::mul(output_grad, &sigmoid_grad)?;
        
        Ok(vec![input_grad])
    }
}

// ---- Tanh ----
#[derive(Debug)]
pub struct TanhBackward;
impl BackwardOp for TanhBackward {
    fn backward(
        &self,
        inputs: &[Tensor],
        output_grad: &Tensor,
    ) -> Result<Vec<Tensor>, TensorError> {
        let input = &inputs[0];
        let tanh_output = ops::tanh(input)?;
        
        // Gradient: grad_input = grad_output * (1 - tanh(input)^2)
        let tanh_squared = ops::mul(&tanh_output, &tanh_output)?;
        let one_minus_tanh_squared = ops::sub_scalar(&tanh_squared, 1.0)?;
        let one_minus_tanh_squared_neg = ops::mul_scalar(&one_minus_tanh_squared, -1.0)?;
        let input_grad = ops::mul(output_grad, &one_minus_tanh_squared_neg)?;
        
        Ok(vec![input_grad])
    }
}

// --- Reduction Operations ---

/// Sums all elements in the tensor. Returns a scalar tensor.
pub fn sum(a: &Tensor) -> Result<Tensor, TensorError> {
    let a_data = a.data();
    let result_scalar = a_data.sum();
    let result_data = ArrayD::from_elem(IxDyn(&[]), result_scalar); // 0-dim scalar output

    create_op_result(result_data, vec![a.clone()], Box::new(SumBackward::new(a.shape().to_vec())))
}

// TODO: Implement Mean: sum(a) / number_of_elements(a)
// pub fn mean(a: &Tensor) -> Result<Tensor, TensorError> { ... }

// TODO: Implement sum/mean along specific axes


// --- Shape Manipulation ---

/// Reshapes a tensor to the target shape.
/// The total number of elements must remain the same.
/// Note: This currently creates a *copy* for simplicity in autograd.
/// A true view-based reshape requires careful handling of strides in backward pass.
pub fn reshape(a: &Tensor, new_shape: &[usize]) -> Result<Tensor, TensorError> {
    let original_size: usize = a.shape().iter().product();
    let new_size: usize = new_shape.iter().product();

    if original_size != new_size {
        return Err(TensorError::ShapeMismatch{
            expected: vec![original_size], // Representing total elements
            got: vec![new_size]
        });
    }

    let a_data = a.data();
    // Attempt reshape. ndarray's reshape might return an error if shape is invalid.
    // Using `.to_shape` forces a copy if the layout changes, simplifying autograd for now.
    let result_data = a_data.to_shape(IxDyn(new_shape))
        .map_err(|e| TensorError::NdarrayError(e))?; // Convert ndarray error to TensorError

    // Autograd for reshape (when it's a copy): Gradient just needs to be reshaped back.
    // If it were a view, it's more complex (strides).
    // TODO: Implement ReshapeBackward
    // create_op_result(result_data, vec![a.clone()], Box::new(ReshapeBackward::new(a.shape().to_vec())))

    // For now, let's assume reshape doesn't participate directly in gradient flow
    // in the same way as math ops, or make it break the graph (detach).
    // A safe default for a simple implementation is to treat it like a copy that
    // might lose grad history if not handled carefully.
    // Let's return a non-gradient tracking tensor for now to avoid complexity.
    // If we want reshape to be differentiable, we need the ReshapeBackward op.
    // Ok(Tensor::new(result_data, false)) // Simplest, breaks graph

    // --- Let's try implementing a basic ReshapeBackward ---
    use super::autograd::op_abstractions::ReshapeBackward; // Add this import

    create_op_result(result_data, vec![a.clone()], Box::new(ReshapeBackward::new(a.shape().to_vec())))

}

// TODO: Implement Transpose
// pub fn transpose(a: &Tensor, dim1: usize, dim2: usize) -> Result<Tensor, TensorError> { ... }


// --- Other Operations ---

// TODO: Implement Power (pow)
// TODO: Implement Exponential (exp)
// TODO: Implement Logarithm (log)
// TODO: Implement Indexing/Slicing (tricky with autograd)


// --- Add methods to Tensor struct via PyO3 or directly ---
// This is often done in the main Tensor definition or in the Python bindings part.
// Example (if adding directly to Tensor impl):
// impl Tensor {
//     pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
//         add(self, other)
//     }
//     pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
//         matmul(self, other)
//     }
//     // ... etc
// }