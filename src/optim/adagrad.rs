//! # Adagrad Optimizer

use super::Optimizer;
use crate::tensor::{ops, zeros, Tensor, TensorData, TensorError};
use std::collections::HashMap;

/// Implements the Adagrad algorithm.
/// Reference: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization - http://jmlr.org/papers/v12/duchi11a.html
pub struct Adagrad {
    params: Vec<Tensor>,
    lr: TensorData,
    lr_decay: TensorData,
    weight_decay: TensorData,
    initial_accumulator_value: TensorData,
    eps: TensorData,

    // State stored per parameter
    state: HashMap<usize, AdagradParamState>,
    // Time step (can influence learning rate decay)
    // Adagrad itself doesn't use 't' directly in its core update, but lr_decay might.
    // Let's keep track if lr_decay is used.
    t: usize, // Keep t for potential lr_decay implementation
}

#[derive(Clone, Debug)]
struct AdagradParamState {
    sum: Tensor, // Sum of squared gradients
}

impl Adagrad {
    /// Creates a new Adagrad optimizer instance.
    ///
    /// # Arguments
    /// * `params`: An iterator over the parameters to optimize.
    /// * `lr`: Learning rate (default: 1e-2).
    /// * `lr_decay`: Learning rate decay (default: 0).
    /// * `weight_decay`: Weight decay (L2 penalty) (default: 0).
    /// * `initial_accumulator_value`: Starting value for the accumulators (default: 0).
    /// * `eps`: Term added to the denominator to improve numerical stability (default: 1e-10).
    pub fn new<I>(
        params: I,
        lr: Option<TensorData>,
        lr_decay: Option<TensorData>,
        weight_decay: Option<TensorData>,
        initial_accumulator_value: Option<TensorData>,
        eps: Option<TensorData>,
    ) -> Result<Self, TensorError>
    where I: IntoIterator<Item = Tensor>
    {
        let params_vec: Vec<Tensor> = params.into_iter().collect();
        let lr_val = lr.unwrap_or(1e-2);
        let lr_decay_val = lr_decay.unwrap_or(0.0);
        let weight_decay_val = weight_decay.unwrap_or(0.0);
        let initial_acc_val = initial_accumulator_value.unwrap_or(0.0);
        let eps_val = eps.unwrap_or(1e-10);

        // --- Input Validation ---
        if !(0.0 <= lr_val) { return Err(TensorError::Generic("Invalid learning rate".into())); }
        if !(0.0 <= lr_decay_val) { return Err(TensorError::Generic("Invalid lr_decay value".into())); }
        if !(0.0 <= weight_decay_val) { return Err(TensorError::Generic("Invalid weight_decay value".into())); }
        if !(0.0 <= initial_acc_val) { return Err(TensorError::Generic("Invalid initial_accumulator_value".into())); }
        if !(0.0 <= eps_val) { return Err(TensorError::Generic("Invalid epsilon value".into())); }


         Ok(Adagrad{
            params: params_vec,
            lr: lr_val,
            lr_decay: lr_decay_val,
            weight_decay: weight_decay_val,
            initial_accumulator_value: initial_acc_val,
            eps: eps_val,
            state: HashMap::new(),
            t: 0,
         })
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self) -> Result<(), TensorError> {
        self.t += 1;

        // Apply learning rate decay if specified
        // clr = lr / (1 + (t - 1) * lr_decay) -> PyTorch formula
        let clr = if self.lr_decay != 0.0 {
             self.lr / (1.0 + (self.t -1) as TensorData * self.lr_decay)
        } else {
            self.lr
        };


        for param in &self.params {
            if !param.requires_grad { continue; }
             let grad = match param.grad() {
                 Some(g) => g,
                 None => continue,
             };

             // Apply weight decay to gradient: grad = grad + param * weight_decay
            let mut current_grad = grad.clone();
            if self.weight_decay != 0.0 {
                 let decay_term = ops::mul_scalar(param, self.weight_decay)?;
                 current_grad = ops::add(¤t_grad, &decay_term)?;
            }
            let final_grad = current_grad;


             // Get or initialize state
             let param_id = param.data.as_ref().as_ptr() as usize;
             let state = self.state
                .entry(param_id)
                .or_insert_with(|| {
                    let mut initial_sum_data = zeros(param.shape(), false).data_clone();
                    initial_sum_data.fill(self.initial_accumulator_value);
                    let initial_sum = Tensor::new(initial_sum_data, false);
                    AdagradParamState { sum: initial_sum }
                 });

             // --- Adagrad Update Rules ---
            // state_sum += grad * grad
            let grad_sq = ops::mul(&final_grad, &final_grad)?;
            // Need mutable access to state.sum data, or use ops::add
            // state.sum = ops::add(&state.sum, &grad_sq)?; // Reassigning tensor state

             // More efficient: update state.sum in place if possible
            {
                let mut sum_data = state.sum.data_mut();
                 let grad_sq_data = grad_sq.data();
                 *sum_data += &*grad_sq_data;
             } // Lock released, state.sum is now updated


            // std = sqrt(state_sum) + eps
            let std_dev = ops::add_scalar(&ops::sqrt(&state.sum)?, self.eps)?;

             // Update: param = param - clr * grad / std
            let update_step = ops::mul_scalar(
                &ops::div(&final_grad, &std_dev)?,
                clr
            )?;

             {
                 let mut param_data = param.data_mut();
                 let step_data = update_step.data();
                 *param_data -= &*step_data;
             }

             // TODO: Need ops: mul, add, sqrt, div, add_scalar, mul_scalar
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
             if param.requires_grad {
                 param.zero_grad();
             }
        }
    }
}