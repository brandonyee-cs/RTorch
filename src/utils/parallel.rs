//! # Parallelism Utilities (CPU Threading)
//!
//! Provides basic data parallelism helpers using threads (e.g., `rayon`).
//! (Experimental / Placeholder)

use crate::nn::Module;
use crate::tensor::{Tensor, TensorError, ops}; // Need ops for gradient aggregation
use rayon::prelude::*; // Import rayon prelude for parallel iterators
use std::sync::{Arc, Mutex}; // For sharing models and collecting gradients

// --- DataParallel Wrapper (Conceptual) ---
// This attempts to mimic torch.nn.DataParallel for CPU.
// It replicates the module on multiple threads and processes input chunks in parallel.

#[derive(Debug)]
pub struct DataParallel<M: Module + Clone + Send + Sync> {
    // The original module (might not be strictly needed if replicas hold all state)
    // module: Arc<M>, // Use Arc for shared ownership if needed
    // Replicas of the module, one per thread (or fewer)
    replicas: Vec<Arc<Mutex<M>>>, // Mutex needed if module state changes during forward (e.g. BatchNorm stats update)
                                // If module forward is truly &self and stateless internally, Mutex might not be needed.
                                // But BatchNorm breaks this assumption. Let's use Mutex for safety.
                                // Note: Cloning a complex module might be expensive.
    // Device IDs (here just indices for CPU threads)
    // device_ids: Vec<usize>, // Represents thread indices
}

impl<M: Module + Clone + Send + Sync + Debug + 'static> DataParallel<M> {
    /// Creates a DataParallel wrapper for a module.
    ///
    /// # Arguments
    /// * `module`: The module to parallelize.
    /// * `num_threads`: The number of threads (and replicas) to use. Defaults to number of logical CPUs.
    pub fn new(module: M, num_threads: Option<usize>) -> Self {
        let threads = num_threads.unwrap_or_else(num_cpus::get);
        if threads == 0 { panic!("Cannot run DataParallel with 0 threads."); }

        // Create replicas by cloning the original module
        let mut replicas = Vec::with_capacity(threads);
        for _ in 0..threads {
            // Clone the module for each replica. This requires M: Clone.
            // Wrap in Arc<Mutex<>> for thread safety during potential state updates (e.g., BatchNorm)
            // and shared access across threads if needed later (though scatter/gather avoids direct sharing during forward).
            replicas.push(Arc::new(Mutex::new(module.clone())));
        }

        DataParallel { replicas }
    }

    /// Gets the number of replicas (threads).
    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }

    /// Scatter input data across replicas (simple chunking along batch dim).
    /// Assumes input is a batch tensor where the first dimension is the batch size.
    fn scatter(&self, input: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let batch_size = input.shape().first().ok_or_else(|| TensorError::Generic("Input tensor has no dimensions".into()))?;
        if *batch_size < self.num_replicas() {
            // Not enough data for all threads, maybe just run on one? Or distribute as is?
            // Let's distribute unevenly for now.
             eprintln!("Warning: Batch size ({}) is smaller than number of replicas ({}).", batch_size, self.num_replicas());
        }

        // Calculate chunk sizes
        let chunk_size = (*batch_size as f64 / self.num_replicas() as f64).ceil() as usize;
        let mut chunks = Vec::with_capacity(self.num_replicas());

        // Use ndarray's slicing/chunking along Axis(0)
        // TODO: Tensor needs a slicing/chunking operation
        // For now, let's assume a placeholder `tensor.chunk(num_chunks, dim)` exists.
        let input_chunks = ops::chunk(input, self.num_replicas(), 0)?; // Assume ops::chunk exists

        for chunk_data in input_chunks {
            // chunks.push(Tensor::new(chunk_data, input.requires_grad)); // Re-wrap data as Tensor
             chunks.push(chunk_data); // Assuming ops::chunk returns Vec<Tensor>
        }

        // Handle case where batch_size isn't divisible - ops::chunk should handle this.

        Ok(chunks)
    }


    /// Gather results from replicas back onto the main device/thread.
    /// Simply concatenates tensors along the batch dimension.
    fn gather(&self, results: Vec<Result<Tensor, TensorError>>) -> Result<Tensor, TensorError> {
        // Collect results, handling potential errors from threads
        let collected_tensors: Result<Vec<Tensor>, TensorError> = results.into_iter().collect();
        let tensors_to_cat = collected_tensors?;

        if tensors_to_cat.is_empty() {
            return Err(TensorError::Generic("Gather received no tensors.".into()));
        }

        // Concatenate along the batch dimension (Axis(0))
        // TODO: Tensor needs a concatenation operation
        ops::cat(&tensors_to_cat, 0) // Assume ops::cat exists
    }

    /// Aggregate gradients from replicas onto the primary module's parameters.
    /// This assumes the primary module (or replica 0) holds the parameters whose
    /// gradients need to be updated by the optimizer.
    fn aggregate_gradients(&self /*, primary_module: &M */) -> Result<(), TensorError> {
         if self.replicas.is_empty() {
            return Ok(());
        }

        // Get parameters of the first replica (considered the primary one)
        let primary_replica_lock = self.replicas[0].lock().map_err(|_| TensorError::Generic("Mutex poisoned".into()))?;
        let primary_params = primary_replica_lock.parameters();
        let num_replicas_f = self.num_replicas() as TensorData;


        // Iterate through parameters of the primary replica
        for (name, primary_param) in primary_params {
             if !primary_param.requires_grad { continue; }

             // Get the gradient computed on the primary replica
             let mut aggregated_grad = primary_param.grad().ok_or_else(|| {
                  // This might happen if backward wasn't called or if the primary replica didn't process data.
                  eprintln!("Warning: Gradient missing for '{}' on primary replica during aggregation.", name);
                  TensorError::NoGradient
             })?.clone(); // Clone the primary grad tensor to modify it

            // Accumulate gradients from other replicas
            for i in 1..self.replicas.len() {
                 let replica_lock = self.replicas[i].lock().map_err(|_| TensorError::Generic("Mutex poisoned".into()))?;
                 // Find the corresponding parameter in the replica by name
                 // This assumes parameter names are consistent across replicas (which they should be if cloned)
                 let replica_params = replica_lock.parameters(); // Inefficient? Maybe store param refs directly?
                 if let Some(replica_param) = replica_params.get(&name) {
                     if let Some(replica_grad) = replica_param.grad() {
                          // Add replica_grad to aggregated_grad
                          // aggregated_grad = ops::add(&aggregated_grad, &replica_grad)?; // This creates new tensors repeatedly
                          // More efficient: Add in-place to aggregated_grad's data
                          {
                              let mut agg_grad_data = aggregated_grad.data_mut();
                              let repl_grad_data = replica_grad.data();
                               if agg_grad_data.shape() == repl_grad_data.shape() {
                                  *agg_grad_data += &*repl_grad_data;
                              } else {
                                   return Err(TensorError::ShapeMismatch { key: name.clone(), expected: agg_grad_data.shape().to_vec(), got: repl_grad_data.shape().to_vec() });
                               }
                          }
                     } else {
                         // Gradient missing on a replica, might indicate uneven data distribution or error
                         eprintln!("Warning: Gradient missing for '{}' on replica {} during aggregation.", name, i);
                     }
                 } else {
                     // Parameter missing on a replica - indicates a serious inconsistency
                     return Err(TensorError::Generic(format!("Parameter '{}' missing on replica {} during gradient aggregation.", name, i)));
                 }
            } // End loop over other replicas

            // Average the aggregated gradient
            let averaged_grad = ops::div_scalar(&aggregated_grad, num_replicas_f)?;

            // Set the averaged gradient back onto the primary parameter
             {
                 // Lock the primary parameter's gradient field
                 let grad_arc = primary_param.grad.as_ref().ok_or(TensorError::NoGradient)?.clone(); // Clone Arc
                 let mut grad_tensor_guard = grad_arc.lock().map_err(|_| TensorError::Generic("Mutex poisoned".into()))?;
                 // Lock the data of the gradient tensor and copy the averaged gradient data into it
                 let mut grad_data_mut = grad_tensor_guard.data_mut();
                 let avg_grad_data = averaged_grad.data();
                 if grad_data_mut.shape() != avg_grad_data.shape() {
                     return Err(TensorError::ShapeMismatch { key: name.clone(), expected: grad_data_mut.shape().to_vec(), got: avg_grad_data.shape().to_vec() });
                 }
                 grad_data_mut.assign(&*avg_grad_data);
             }

        } // End loop over parameters

        Ok(())
    }


    // Note: The standard Module trait's forward might not be the best fit here.
    // We might need a custom `parallel_forward` or integrate this into the training loop logic.
    // Let's try to implement the Module trait, but acknowledge its limitations.

    // Issue: Module::forward returns ONE tensor. DataParallel needs to handle scatter/gather.
    // This wrapper might be better used *outside* the standard Module structure,
    // e.g., in the training loop:
    // let inputs = dp_wrapper.scatter(batch_input)?;
    // let results = dp_wrapper.parallel_apply(inputs)?;
    // let output = dp_wrapper.gather(results)?;
    // output.backward()? // Standard backward call
    // dp_wrapper.aggregate_gradients()?; // Special step before optimizer.step()
    // optimizer.step()?;

    // If we MUST implement Module:
    // The forward pass will involve scatter, parallel apply, and gather.
    // Backward pass needs careful consideration - does calling .backward() on the gathered output
    // correctly propagate gradients back through the scatter/gather ops (if they were autograd-aware)
    // and then to the individual replicas? If scatter/gather aren't autograd ops, then no.
    // PyTorch handles this internally.

    // Let's implement a simplified `parallel_apply` method instead of forcing into Module::forward
    pub fn parallel_apply(&self, inputs: Vec<Tensor>) -> Vec<Result<Tensor, TensorError>> {
        if inputs.len() != self.replicas.len() {
            // Or handle mismatch differently
            panic!("Number of input chunks ({}) does not match number of replicas ({})", inputs.len(), self.replicas.len());
        }

        // Use rayon to process chunks in parallel
        inputs.into_par_iter()
            .zip(self.replicas.par_iter()) // Parallel zip
            .map(|(input_chunk, replica_arc)| {
                // Lock the replica's mutex to call forward
                let replica = replica_arc.lock().map_err(|_| TensorError::Generic("Mutex poisoned".into()))?;
                // Call the replica's forward method
                replica.forward(&input_chunk)
            })
            .collect() // Collect results (Vec<Result<Tensor, TensorError>>)
    }
}

// --- Placeholder ops needed by DataParallel ---
mod ops_local_parallel {
     use super::*;
     use ndarray::{Axis, stack, ArrayViewD};

    /// Placeholder for chunking a tensor along a dimension.
    pub fn chunk(tensor: &Tensor, chunks: usize, dim: usize) -> Result<Vec<Tensor>, TensorError> {
         if chunks == 0 { return Err(TensorError::Generic("Cannot chunk into 0 pieces.".into())); }
         let dim_size = tensor.shape().get(dim).ok_or_else(|| TensorError::Generic("Invalid dimension for chunking.".into()))?;
         if *dim_size < chunks {
             eprintln!("Warning: Chunk dimension size ({}) is smaller than number of chunks ({}).", dim_size, chunks);
             // Fallback: return fewer chunks than requested? Or error? Let's try returning fewer.
             // Or maybe ndarray handles this? view().axis_chunks(Axis(dim), chunks) might fail.
             // Let's assume we return exactly `chunks` tensors, some might be empty if needed? No, ndarray won't do that.
             // For simplicity, let's just split into available chunks.
         }

        let num_chunks_to_make = chunks.min(*dim_size); // Don't make more chunks than dim size
        let chunk_size = (*dim_size as f64 / num_chunks_to_make as f64).ceil() as isize; // isize for stride

        let data = tensor.data();
        let views = data.axis_chunks(Axis(dim), chunk_size); // This returns an iterator

        let mut result_tensors = Vec::new();
         for view in views {
            // Create new tensors from views (requires copy for ownership)
            // For autograd, this needs to be an op!
             result_tensors.push(Tensor::new(view.to_owned(), tensor.requires_grad)); // Copy data
         }

        // If fewer chunks were created than requested due to dim_size < chunks,
        // what should we do? Pad with empty tensors? Error?
        // For now, just return the chunks created. DataParallel needs adjustment.
        // A better chunk would distribute remainders.

         // TODO: Implement ChunkBackward op for autograd.
         Ok(result_tensors)
    }

    /// Placeholder for concatenating tensors along a dimension.
     pub fn cat(tensors: &[Tensor], dim: usize) -> Result<Tensor, TensorError> {
         if tensors.is_empty() {
             return Err(TensorError::Generic("Cannot concatenate empty list of tensors.".into()));
         }
         // Check shapes are compatible for concatenation (all dims match except `dim`)
         // ... (validation needed) ...

         // Get views of the data
         let views: Vec<ArrayViewD<'_, TensorData>> = tensors.iter().map(|t| t.data().view()).collect();

         // Use ndarray::stack or ndarray::concatenate
         let result_data = ndarray::concatenate(Axis(dim), &views)
             .map_err(|e| TensorError::NdarrayError(e))?; // Convert ndarray error

         // TODO: Implement ConcatenateBackward op for autograd.
         // Determine requires_grad based on inputs
         let requires_grad = tensors.iter().any(|t| t.requires_grad);
         Ok(Tensor::new(result_data, requires_grad))
    }
}
// Make placeholders visible
use ops_local_parallel::{chunk, cat};


// --- DataParallel does not easily fit Module trait ---
// Let's comment out the Module implementation attempt.
// Users should interact with DataParallel via scatter/parallel_apply/gather/aggregate_gradients.

// impl<M: Module + Clone + Send + Sync + Debug + 'static> Module for DataParallel<M> {
//     fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
//         // 1. Scatter input
//         let chunks = self.scatter(input)?;
//         // 2. Apply replicas in parallel
//         let results = self.parallel_apply(chunks);
//         // 3. Gather results
//         self.gather(results)
//         // PROBLEM: How does backward() on the gathered output work?
//         // It needs to propagate through gather -> parallel_apply -> scatter.
//         // This requires these functions to be autograd-aware operations.
//     }

//     fn parameters(&self) -> BTreeMap<String, Tensor> {
//         // Return parameters from the primary replica (replica 0)
//         if let Some(primary_replica_arc) = self.replicas.first() {
//              if let Ok(primary_replica) = primary_replica_arc.lock() {
//                 return primary_replica.parameters();
//              }
//         }
//         BTreeMap::new() // Return empty if replicas is empty or lock fails
//     }

//      fn tensors(&self) -> BTreeMap<String, Tensor> {
//          // Return tensors from the primary replica
//          if let Some(primary_replica_arc) = self.replicas.first() {
//              if let Ok(primary_replica) = primary_replica_arc.lock() {
//                 return primary_replica.tensors();
//              }
//         }
//         BTreeMap::new()
//     }

//     // zero_grad should apply to the primary replica (whose params are optimized)
//      fn zero_grad(&self) {
//          if let Some(primary_replica_arc) = self.replicas.first() {
//              if let Ok(primary_replica) = primary_replica_arc.lock() {
//                  // Need a way to call zero_grad on the locked module.
//                  // If Module::zero_grad takes &self (with internal mutability), this works.
//                  primary_replica.zero_grad();
//              } else {
//                  eprintln!("Warning: Could not acquire lock on primary replica to zero gradients.");
//              }
//         }
//      }

//      // train/eval should apply to *all* replicas
//       fn train(&mut self) { // Requires &mut self? Or &self if using interior mutability
//           for replica_arc in &self.replicas {
//                if let Ok(mut replica) = replica_arc.lock() { // Need mutable lock guard if train takes &mut self
//                     // replica.train(); // Call train on the replica
//                } else {
//                   eprintln!("Warning: Could not acquire lock on replica during train().");
//               }
//           }
//            // If train takes &self:
//            // for replica_arc in &self.replicas {
//            //      if let Ok(replica) = replica_arc.lock() {
//            //           replica.train(); // Assumes Module::train(&self)
//            //      }
//            // }
//       }

//        fn eval(&mut self) { // Similar issues as train()
//            for replica_arc in &self.replicas {
//                 if let Ok(mut replica) = replica_arc.lock() {
//                     // replica.eval();
//                 } else {
//                   eprintln!("Warning: Could not acquire lock on replica during eval().");
//               }
//            }
//            // If eval takes &self:
//            // for replica_arc in &self.replicas {
//            //      if let Ok(replica) = replica_arc.lock() {
//            //           replica.eval(); // Assumes Module::eval(&self)
//            //      }
//            // }
//        }
// }