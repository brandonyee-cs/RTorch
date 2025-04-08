//! # Adam Optimizer

use super::Optimizer;
use crate::tensor::{ops, zeros, Tensor, TensorData, TensorError};
use std::collections::HashMap;

/// Implements the Adam algorithm.
/// Reference: Adam: A Method for Stochastic Optimization - https://arxiv.org/abs/1412.6980
pub struct Adam {
    params: Vec<Tensor>,
    lr: TensorData,
    betas: (TensorData, TensorData), // (beta1, beta2)
    eps: TensorData,
    weight_decay: TensorData,
    amsgrad: bool,

    // State stored per parameter
    state: HashMap<usize, AdamParamState>,
    // Time step (number of calls to step()) - use usize or TensorData? usize is fine.
    t: usize,
}

#[derive(Clone, Debug)]
struct AdamParamState {
    exp_avg: Tensor,       // 1st moment estimate (momentum) - m_t
    exp_avg_sq: Tensor,    // 2nd moment estimate (RMSprop like) - v_t
    max_exp_avg_sq: Option<Tensor>, // Max v_t, only used if amsgrad = true
}

impl Adam {
    /// Creates a new Adam optimizer instance.
    ///
    /// # Arguments
    /// * `params`: An iterator over the parameters to optimize.
    /// * `lr`: Learning rate (default: 1e-3).
    /// * `betas`: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
    /// * `eps`: Term added to the denominator to improve numerical stability (default: 1e-8).
    /// * `weight_decay`: Weight decay (L2 penalty) (default: 0).
    /// * `amsgrad`: Whether to use the AMSGrad variant of this algorithm (default: false).
     pub fn new<I>(
        params: I,
        lr: Option<TensorData>,
        betas: Option<(TensorData, TensorData)>,
        eps: Option<TensorData>,
        weight_decay: Option<TensorData>,
        amsgrad: bool,
    ) -> Result<Self, TensorError>
     where I: IntoIterator<Item = Tensor>
     {
        let params_vec: Vec<Tensor> = params.into_iter().collect();
        let lr_val = lr.unwrap_or(1e-3);
        let betas_val = betas.unwrap_or((0.9, 0.999));
        let eps_val = eps.unwrap_or(1e-8);
        let weight_decay_val = weight_decay.unwrap_or(0.0);

        // --- Input Validation ---
        if !(0.0 <= lr_val) { return Err(TensorError::Generic("Invalid learning rate: must be >= 0".into())); }
        if !(0.0 <= eps_val) { return Err(TensorError::Generic("Invalid epsilon value: must be >= 0".into())); }
        if !(0.0 <= betas_val.0 && betas_val.0 < 1.0) { return Err(TensorError::Generic("Invalid beta parameter at index 0".into())); }
        if !(0.0 <= betas_val.1 && betas_val.1 < 1.0) { return Err(TensorError::Generic("Invalid beta parameter at index 1".into())); }
        if !(0.0 <= weight_decay_val) { return Err(TensorError::Generic("Invalid weight_decay value: must be >= 0".into())); }


         Ok(Adam {
            params: params_vec,
            lr: lr_val,
            betas: betas_val,
            eps: eps_val,
            weight_decay: weight_decay_val,
            amsgrad,
            state: HashMap::new(),
            t: 0,
         })
     }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<(), TensorError> {
        self.t += 1; // Increment time step
        let beta1 = self.betas.0;
        let beta2 = self.betas.1;

        // Bias correction terms
        // Use ops::pow_scalar or std::powf
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32); // beta1^t
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32); // beta2^t
        // Correction factor for step size calculation
        let step_size_correction = bias_correction2.sqrt() / bias_correction1;


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
            let final_grad = current_grad; // Renamed for clarity

             // Get or initialize state for this parameter
             let param_id = param.data.as_ref().as_ptr() as usize;
             let state = self.state
                 .entry(param_id)
                 .or_insert_with(|| {
                    let shape = param.shape();
                    AdamParamState {
                        exp_avg: zeros(shape, false), // m_0 = 0
                        exp_avg_sq: zeros(shape, false), // v_0 = 0
                        max_exp_avg_sq: if self.amsgrad { Some(zeros(shape, false)) } else { None },
                    }
                 });

             // --- Adam Update Rules ---
            // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            let m_prev = &state.exp_avg;
            let m_t = ops::add(
                &ops::mul_scalar(m_prev, beta1)?,
                &ops::mul_scalar(&final_grad, 1.0 - beta1)?
            )?;

             // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            let v_prev = &state.exp_avg_sq;
            let grad_sq = ops::mul(&final_grad, &final_grad)?; // g_t^2
            let v_t = ops::add(
                &ops::mul_scalar(v_prev, beta2)?,
                &ops::mul_scalar(&grad_sq, 1.0 - beta2)?
            )?;

            // --- Denominator calculation (sqrt(v_hat_t) + eps) ---
            let denom = if self.amsgrad {
                 // v_hat_max = max(v_hat_max_{t-1}, v_hat_t)
                 // Use ops::maximum
                 let max_v_prev = state.max_exp_avg_sq.as_ref().unwrap(); // Should exist if amsgrad
                 let v_t_clone = v_t.clone(); // Clone v_t as it's needed later if amsgrad updates state
                 let max_v_t = ops::maximum(max_v_prev, &v_t_clone)?; // Assume ops::maximum exists (element-wise max)
                 // Update state
                 state.max_exp_avg_sq = Some(max_v_t.clone());
                 // denom = sqrt(v_hat_max) / sqrt(bias_corr2) + eps -- Error in formula, use v_hat_max directly
                 // denom = sqrt(max_v_t / bias_correction2) + eps -> Needs sqrt op
                 // denom = sqrt(max_v_t) / sqrt(bias_correction2) + eps ?? Let's stick closer to PyTorch:
                 // denom = sqrt(max_v_t) + eps
                 ops::add_scalar(&ops::sqrt(&max_v_t)?, self.eps)? // Assume ops::sqrt exists

            } else {
                 // denom = sqrt(v_t / bias_correction2) + eps -> Needs sqrt op
                 // denom = sqrt(v_t) / sqrt(bias_correction2) + eps
                 // denom = sqrt(v_t) + eps (PyTorch applies bias correction to step size later)
                 ops::add_scalar(&ops::sqrt(&v_t)?, self.eps)?
            };


            // --- Parameter Update ---
            // step = lr * m_hat_t / denom
            // m_hat_t = m_t / bias_correction1
            // step = lr * (m_t / bias_correction1) / denom
            // step = (lr / bias_correction1) * m_t / denom -- Potential numerical issue if bias_correction1 is tiny
            // step = lr * (m_t / denom) / bias_correction1 ??
            // PyTorch: step = (lr * step_size_correction) * m_t / denom
             let step = ops::mul_scalar(
                &ops::div(&m_t, &denom)?, // m_t / denom
                self.lr * step_size_correction // effective step size
            )?;

            // Update parameter: param = param - step
            {
                let mut param_data = param.data_mut();
                let step_data = step.data();
                *param_data -= &*step_data;
            }

             // Update state persistence
             state.exp_avg = m_t;
             state.exp_avg_sq = v_t;
             // max_exp_avg_sq already updated inside amsgrad block


             // TODO: Need ops: mul, add, sqrt, div, maximum, add_scalar, mul_scalar
             // Ensure these ops handle autograd correctly (though results aren't tracked here)
             // and potential broadcasting.
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