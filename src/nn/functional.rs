//! # Neural Network Functional Interface (`nn::functional`)
//!
//! Provides stateless functions for common neural network operations,
//! mirroring `torch.nn.functional`. These functions operate directly on tensors.

use crate::tensor::{ops, Tensor, TensorData, TensorError};
// May need access to other crates like rand for dropout if implemented here
// use rand::prelude::*;
// use rand::distributions::Uniform;

// --- Activation Functions ---

/// Applies the Rectified Linear Unit (ReLU) function element-wise.
/// `relu(x) = max(0, x)`
pub fn relu(input: &Tensor) -> Result<Tensor, TensorError> {
    ops::relu(input)
}

/// Applies the Sigmoid function element-wise.
/// `sigmoid(x) = 1 / (1 + exp(-x))`
pub fn sigmoid(input: &Tensor) -> Result<Tensor, TensorError> {
    // Sigmoid = 1 / (1 + exp(-x))
    // Need ops::exp and ops::neg (or mul_scalar by -1) and ops::add_scalar, ops::div
    // Let's implement step-by-step using available ops (assuming they exist or are added)

    // 1. Negate input: -x
    // let neg_input = ops::mul_scalar(input, -1.0)?; // Assuming mul_scalar exists
    let neg_input = ops::mul_scalar(input, -1.0)?; // Use existing mul_scalar

    // 2. Exponentiate: exp(-x)
    let exp_neg_input = ops::exp(&neg_input)?; // Assuming ops::exp exists

    // 3. Add 1: 1 + exp(-x)
    // let one = Tensor::new(ndarray::arr0(1.0 as TensorData).into_dyn(), false); // Create scalar 1
    // let denominator = ops::add(&one, &exp_neg_input)?;
    let denominator = ops::add_scalar(&exp_neg_input, 1.0)?; // Assuming ops::add_scalar exists

    // 4. Reciprocal: 1 / (1 + exp(-x))
    // let one = Tensor::new(ndarray::arr0(1.0 as TensorData).into_dyn(), false); // Create scalar 1
    // let result = ops::div(&one, &denominator)?; // Assuming ops::div exists
    let result = ops::reciprocal(&denominator)?; // Assuming ops::reciprocal exists


    // TODO: Implement ops::exp, ops::add_scalar, ops::reciprocal (or ops::div with scalar)
    // For now, return unimplemented or a placeholder
    // Err(TensorError::Generic("Sigmoid function requires exp, add_scalar, reciprocal ops to be implemented".to_string()))

    // Placeholder implementation (if ops not ready):
    // Calculate using ndarray directly, but without autograd!
    // let input_data = input.data();
    // let result_data = input_data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    // Ok(Tensor::new(result_data, false)) // No autograd link!

    // Assuming the required ops exist and handle autograd:
     Ok(result) // Return the result calculated using tensor ops
}

/// Applies the Hyperbolic Tangent (Tanh) function element-wise.
/// `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
pub fn tanh(input: &Tensor) -> Result<Tensor, TensorError> {
    // Need ops::exp, ops::neg, ops::sub, ops::add, ops::div

    // 1. exp(x)
    let exp_pos = ops::exp(input)?; // Assume exists

    // 2. exp(-x)
    let neg_input = ops::mul_scalar(input, -1.0)?;
    let exp_neg = ops::exp(&neg_input)?;

    // 3. Numerator: exp(x) - exp(-x)
    let numerator = ops::sub(&exp_pos, &exp_neg)?; // Assume exists

    // 4. Denominator: exp(x) + exp(-x)
    let denominator = ops::add(&exp_pos, &exp_neg)?;

    // 5. Division
    let result = ops::div(&numerator, &denominator)?; // Assume exists

    // TODO: Implement ops::exp, ops::sub, ops::div if not already done.
    // For now, return unimplemented or placeholder
     // Err(TensorError::Generic("Tanh function requires exp, sub, div ops to be implemented".to_string()))

    // Assuming ops exist:
    Ok(result)
}

/// Applies the Softmax function to an n-dimensional input Tensor
/// rescaling them so that the elements of the n-dimensional output Tensor
/// lie in the range [0,1] and sum to 1.
/// Softmax is often computed along a specific dimension.
/// `softmax(x_i) = exp(x_i) / sum(exp(x_j))`
pub fn softmax(input: &Tensor, dim: i32) -> Result<Tensor, TensorError> {
    // Need ops::exp, ops::sum(axis), ops::div (with broadcasting)

    // 1. Find max for numerical stability: max(x) along dim
    let max_val = ops::max_axis(input, dim, true)?; // Assume exists, keep_dims=true

    // 2. Subtract max: x - max(x)
    let centered_input = ops::sub(input, &max_val)?; // Broadcasting subtraction

    // 3. Exponentiate: exp(x - max(x))
    let exp_input = ops::exp(¢ered_input)?; // Assume exists

    // 4. Sum exponents along dim: sum(exp(x - max(x)))
    let sum_exp = ops::sum_axis(&exp_input, dim, true)?; // Assume exists, keep_dims=true

    // 5. Divide: exp(x - max(x)) / sum(...)
    let result = ops::div(&exp_input, &sum_exp)?; // Broadcasting division

    // TODO: Implement ops::max_axis, ops::sum_axis, ops::exp, ops::sub, ops::div and broadcasting
    // Err(TensorError::Generic("Softmax function requires max_axis, sum_axis, exp, sub, div ops".to_string()))

    // Assuming ops exist:
    Ok(result)
}

/// Applies the LogSoftmax function.
/// `log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))` (numerically stable form)
pub fn log_softmax(input: &Tensor, dim: i32) -> Result<Tensor, TensorError> {
     // Need ops::max_axis, ops::sub, ops::exp, ops::sum_axis, ops::log, ops::sub

    // 1. Find max for numerical stability: max(x) along dim
    let max_val = ops::max_axis(input, dim, true)?; // keep_dims=true

    // 2. Subtract max: x - max(x)
    let centered_input = ops::sub(input, &max_val)?;

    // 3. Exponentiate: exp(x - max(x))
    let exp_centered = ops::exp(¢ered_input)?;

    // 4. Sum exponents along dim: sum(exp(x - max(x)))
    let sum_exp = ops::sum_axis(&exp_centered, dim, true)?;

    // 5. Log of sum: log(sum(exp(x - max(x))))
    let log_sum_exp = ops::log(&sum_exp)?; // Assume exists

    // 6. Final result: (x - max(x)) - log(sum(exp(x - max(x))))
    // Note: (x - max(x)) is `centered_input`
    //       log(sum(exp(x - max(x)))) is `log_sum_exp`
    // We need to compute `x - max(x) - log(...)`, which is `x - (max(x) + log(...))`
    // Stable form: `x - log_sum_exp(x)` where `log_sum_exp(x) = max(x) + log(sum(exp(x-max(x))))`
    let log_sum_exp_term = ops::add(&max_val, &log_sum_exp)?; // max(x) + log(...)
    let result = ops::sub(input, &log_sum_exp_term)?; // x - log_sum_exp(x)


    // TODO: Implement ops::max_axis, ops::sum_axis, ops::exp, ops::log, ops::sub, ops::add, broadcasting
    // Err(TensorError::Generic("LogSoftmax function requires various ops".to_string()))

     // Assuming ops exist:
    Ok(result)
}


// --- Loss Functions ---

/// Calculates the Mean Squared Error (MSE) loss between input and target.
/// `loss = mean((input - target)^2)`
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    // Need ops::sub, ops::pow_scalar (or mul), ops::mean
    if input.shape() != target.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: target.shape().to_vec(),
            got: input.shape().to_vec(),
        });
    }

    // 1. Difference: input - target
    let diff = ops::sub(input, target)?;

    // 2. Square: (input - target)^2
    // let squared_diff = ops::pow_scalar(&diff, 2.0)?; // Assume exists
    let squared_diff = ops::mul(&diff, &diff)?; // Element-wise multiplication

    // 3. Mean: mean(...)
    let loss = ops::mean(&squared_diff)?; // Assume exists

    // TODO: Implement ops::mean, ops::pow_scalar (optional)
     // Err(TensorError::Generic("MSE Loss function requires mean op".to_string()))

    // Assuming ops exist:
    Ok(loss)
}

/// Calculates the Negative Log Likelihood (NLL) loss.
/// Often used after a LogSoftmax layer for classification.
/// Expects input to be log-probabilities and target to be class indices.
/// `loss = -mean(input[i, target[i]])` (for batch i)
///
/// # Arguments
/// * `input`: Tensor of shape `(N, C)` where `N` is batch size, `C` is number of classes. Contains log-probabilities.
/// * `target`: Tensor of shape `(N)` containing class indices `(0 <= target[i] < C)`. Must be integer type (e.g., i64).
///
/// # Returns
/// * Scalar tensor representing the mean NLL loss.
pub fn nll_loss(log_probs: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    // Needs indexing/gathering capabilities, ops::neg, ops::mean

    // --- Input validation ---
    if log_probs.ndim() != 2 {
        return Err(TensorError::Generic("nll_loss: Expected input to be 2D (N, C)".to_string()));
    }
    if target.ndim() != 1 {
         return Err(TensorError::Generic("nll_loss: Expected target to be 1D (N)".to_string()));
    }
    let n = log_probs.shape()[0]; // Batch size
    let c = log_probs.shape()[1]; // Number of classes
    if n != target.shape()[0] {
        return Err(TensorError::IncompatibleShapes {
            op: "nll_loss".to_string(),
            shape1: log_probs.shape().to_vec(),
            shape2: target.shape().to_vec(),
        });
    }
    // TODO: Check target tensor data type is integer (e.g., i64). This requires Tensor generics or metadata.

    // --- Calculation ---
    // 1. Gather the log-probabilities corresponding to the target classes.
    //    Need `gather(input, dim=1, index=target)` functionality.
    //    Target needs shape (N, 1) for gather along dim 1.
    let target_expanded = ops::unsqueeze(target, 1)?; // Add dim: (N) -> (N, 1). Assume exists.
    // Gather requires target indices to be i64/usize typically. Need conversion or typed tensor.
    // Assume gather handles type or Tensor stores integer types.
    let gathered_log_probs = ops::gather(log_probs, 1, &target_expanded)?; // Assume exists. Shape (N, 1)

    // 2. Negate the gathered log-probabilities: -log P(correct_class)
    let neg_gathered = ops::neg(&gathered_log_probs)?; // Assume exists

    // 3. Compute the mean loss over the batch.
    //    Mean of a tensor with shape (N, 1) is the same as mean of shape (N).
    let loss = ops::mean(&neg_gathered)?; // Assume exists

    // TODO: Implement ops::gather, ops::unsqueeze, ops::neg, ops::mean
     // Err(TensorError::Generic("NLL Loss function requires gather, unsqueeze, neg, mean ops".to_string()))

     // Assuming ops exist:
     Ok(loss)
}


/// Calculates the Cross-Entropy loss.
/// This function combines `LogSoftmax` and `NLLLoss` in one step for better numerical stability.
/// `loss = -log(softmax(input))[target]`
///
/// # Arguments
/// * `input`: Tensor of shape `(N, C)` containing raw scores (logits).
/// * `target`: Tensor of shape `(N)` containing class indices `(0 <= target[i] < C)`. Must be integer type.
///
/// # Returns
/// * Scalar tensor representing the mean Cross-Entropy loss.
pub fn cross_entropy_loss(input: &Tensor, target: &Tensor) -> Result<Tensor, TensorError> {
    // 1. Apply LogSoftmax along the class dimension (dim=1 for N, C input)
    let log_probs = log_softmax(input, 1)?; // Use functional log_softmax

    // 2. Apply NLL loss to the log-probabilities and target indices
    let loss = nll_loss(&log_probs, target)?; // Use functional nll_loss

    // Note: This implementation relies on the component functions being correctly implemented.
    // A potentially more optimized version could implement the combined calculation directly.

    Ok(loss)
}


// --- Other Functional Operations (Placeholders) ---

// pub fn dropout(input: &Tensor, p: f64, training: bool) -> Result<Tensor, TensorError> {
//     if !training || p == 0.0 {
//         return Ok(input.clone()); // No dropout during eval or if p=0
//     }
//     if !(0.0..=1.0).contains(&p) {
//         return Err(TensorError::Generic("Dropout probability must be between 0 and 1".to_string()));
//     }
//     let scale = 1.0 / (1.0 - p);
//     let mask_data = ArrayD::random_using(input.shape(), Uniform::new(0.0, 1.0), &mut rand::thread_rng())
//         .mapv(|x| if x < p { 0.0 } else { scale as TensorData });
//     let mask = Tensor::new(mask_data, false); // Mask doesn't require grad
//     ops::mul(input, &mask) // Apply mask (autograd handles the multiplication)
// }

// pub fn max_pool2d(input: &Tensor, kernel_size: usize, stride: usize) -> Result<Tensor, TensorError> {
//     // TODO: Implement MaxPooling2D logic (complex)
//     Err(TensorError::Generic("max_pool2d not implemented".to_string()))
// }

// pub fn avg_pool2d(input: &Tensor, kernel_size: usize, stride: usize) -> Result<Tensor, TensorError> {
//     // TODO: Implement AvgPooling2D logic (complex)
//      Err(TensorError::Generic("avg_pool2d not implemented".to_string()))
// }