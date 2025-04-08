//! # Stochastic Gradient Descent (SGD) Optimizer

use super::Optimizer;
use crate::tensor::{ops, Tensor, TensorData, TensorError};
use std::collections::HashMap; // For momentum state

/// Implements the Stochastic Gradient Descent optimizer.
/// Supports momentum, dampening, weight decay, and Nesterov momentum.
pub struct SGD {
    params: Vec<Tensor>,
    lr: TensorData,
    momentum: TensorData,
    dampening: TensorData,
    weight_decay: TensorData,
    nesterov: bool,
    // State for momentum buffers (one per parameter)
    momentum_buffers: HashMap<usize, Tensor>, // Key: Parameter ID (e.g., memory address)
}

impl SGD {
    /// Creates a new SGD optimizer instance.
    ///
    /// # Arguments
    /// * `params`: An iterator over the parameters (Tensors that require grad) to optimize.
    /// * `lr`: Learning rate.
    /// * `momentum`: Momentum factor (default: 0).
    /// * `dampening`: Dampening for momentum (default: 0).
    /// * `weight_decay`: Weight decay (L2 penalty) (default: 0).
    /// * `nesterov`: Enables Nesterov momentum (default: false).
    pub fn new<I>(
        params: I,
        lr: TensorData,
        momentum: Option<TensorData>,
        dampening: Option<TensorData>,
        weight_decay: Option<TensorData>,
        nesterov: Option<TensorData>, // Nesterov requires momentum > 0
    ) -> Result<Self, TensorError>
    where
        I: IntoIterator<Item = Tensor>,
    {
        let params_vec: Vec<Tensor> = params.into_iter().collect();
        if lr < 0.0 {
            return Err(TensorError::Generic("Invalid learning rate: cannot be negative".to_string()));
        }
        let momentum_val = momentum.unwrap_or(0.0);
        let dampening_val = dampening.unwrap_or(0.0);
        let weight_decay_val = weight_decay.unwrap_or(0.0);
        let nesterov_val = nesterov.is_some(); // Check if Some (passed as Some(val))? Or just bool? Let's use bool.
                                               // pub fn new(..., nesterov: bool) -> ...
                                               // Let's assume nesterov is just a bool flag for simplicity
                                               // nesterov: bool,

        if momentum_val < 0.0 {
             return Err(TensorError::Generic("Invalid momentum value: cannot be negative".to_string()));
        }
        if weight_decay_val < 0.0 {
             return Err(TensorError::Generic("Invalid weight_decay value: cannot be negative".to_string()));
        }
        if nesterov_val && (momentum_val <= 0.0 || dampening_val != 0.0) {
             return Err(TensorError::Generic("Nesterov momentum requires momentum > 0 and dampening = 0".to_string()));
        }


        Ok(SGD {
            params: params_vec,
            lr,
            momentum: momentum_val,
            dampening: dampening_val,
            weight_decay: weight_decay_val,
            nesterov: nesterov_val,
            momentum_buffers: HashMap::new(),
        })
    }

    /// Simplified constructor with only lr.
     pub fn simple<I>(params: I, lr: TensorData) -> Result<Self, TensorError>
     where I: IntoIterator<Item = Tensor>
     {
        Self::new(params, lr, None, None, None, None) // Pass None for Nesterov flag? Let's refine constructor arg
        // Refined: pub fn new(..., nesterov: bool) -> ...
        // Self::new(params.into_iter().collect(), lr, 0.0, 0.0, 0.0, false)
     }
}


impl Optimizer for SGD {
    fn step(&mut self) -> Result<(), TensorError> {
        for param in &self.params {
            // Ensure parameter requires gradient and has a gradient computed
            if !param.requires_grad {
                continue; // Skip parameters that don't require gradients
            }
            let grad = match param.grad() {
                Some(g) => g,
                None => continue, // Skip parameters without computed gradients
            };

            // Ensure gradient is not detached and has data
            // let grad_data = grad.data(); // Read lock

            // --- Weight Decay ---
            // param.data = param.data - param.data * weight_decay * lr (Incorrect place)
            // grad = grad + param.data * weight_decay
            let mut current_grad = grad.clone(); // Clone grad tensor for modification
            if self.weight_decay != 0.0 {
                 // Need mutable access to current_grad's data
                 // let mut grad_data_mut = current_grad.data_mut();
                 // let param_data = param.data();
                 // *grad_data_mut += &*param_data * self.weight_decay;

                 // Or using ops:
                 let decay_term = ops::mul_scalar(param, self.weight_decay)?;
                 current_grad = ops::add(¤t_grad, &decay_term)?;
            }

             // Lock gradient data *after* potential modification by weight decay
            let final_grad_data = current_grad.data();


            // --- Momentum ---
            if self.momentum != 0.0 {
                let param_id = param.data.as_ref().as_ptr() as usize; // Use data pointer as unique ID
                let buf = self.momentum_buffers
                    .entry(param_id)
                    .or_insert_with(|| crate::tensor::zeros(param.shape(), false)); // Initialize buffer with zeros if not present

                // Update momentum buffer: buf = momentum * buf + (1 - dampening) * grad
                // Need mutable access to buf data
                 // let mut buf_data = buf.data_mut();
                 // *buf_data = &*buf_data * self.momentum + &*final_grad_data * (1.0 - self.dampening);

                // Or using ops:
                let buf_scaled = ops::mul_scalar(buf, self.momentum)?;
                let grad_scaled = ops::mul_scalar(¤t_grad, 1.0 - self.dampening)?; // Use current_grad (with weight decay)
                let updated_buf = ops::add(&buf_scaled, &grad_scaled)?;
                // Replace the old buffer in the map
                self.momentum_buffers.insert(param_id, updated_buf.clone()); // Store updated buffer

                // Update gradient based on Nesterov or standard momentum
                if self.nesterov {
                    // grad = grad + momentum * buf (using the *updated* buffer)
                     // *final_grad_data = &*final_grad_data + &*updated_buf.data() * self.momentum; // Can't modify final_grad_data directly
                     // Need to recalculate final_grad based on updated_buf
                     let nesterov_term = ops::mul_scalar(&updated_buf, self.momentum)?;
                     current_grad = ops::add(¤t_grad, &nesterov_term)?; // Update current_grad again

                } else {
                     // grad = buf (use the *updated* buffer as the effective gradient)
                     // *final_grad_data = updated_buf.data().clone(); // Cannot assign directly
                     current_grad = updated_buf; // Replace current_grad with the buffer value
                }

            } // End momentum block


            // --- Parameter Update ---
            // param.data = param.data - lr * grad
            // Need write access to parameter data
             let final_update_grad = current_grad; // The gradient to use for the update step
             let update_step = ops::mul_scalar(&final_update_grad, self.lr)?;

            { // Scope for write lock
                let mut param_data = param.data_mut();
                let update_data = update_step.data();
                *param_data -= &*update_data; // Perform the update in-place
            } // Lock released

        } // End loop over params
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            if param.requires_grad {
                param.zero_grad(); // Call Tensor's zero_grad method
            }
        }
        // Also zero momentum buffers? No, momentum buffers persist across steps.
        // PyTorch's optimizer.zero_grad() only zeros the .grad attribute of parameters.
    }
}