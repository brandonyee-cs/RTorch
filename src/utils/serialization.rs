//! # Model Serialization Utilities
//!
//! Functions for saving and loading model parameters (and potentially optimizer state).
//! Uses `serde` for serialization and `bincode` as the binary format.

use crate::tensor::{Tensor, TensorData}; // Need Tensor definition
use crate::nn::Module; // Need Module trait to get parameters
// If saving optimizer state:
// use crate::optim::{Optimizer, Adam, SGD, Adagrad}; // Need optimizer types and potentially their state

use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use bincode; // Choose a serialization format

// --- Error Type ---
#[derive(thiserror::Error, Debug)]
pub enum SerializationError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization Error (Bincode): {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Tensor shape mismatch during loading: key '{key}', expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
     #[error("Missing key in state dict during loading: '{0}'")]
    MissingKey(String),
     #[error("Unexpected key in state dict during loading: '{0}'")]
    UnexpectedKey(String), // If strict loading is enabled
    #[error("Could not access tensor data: {0}")]
    TensorDataAccessError(String),
     #[error("Could not acquire lock for tensor data")]
    TensorLockError,
     // Add errors for optimizer state loading if implemented
}


// --- Serializable Tensor Wrapper ---
// We need a way to serialize/deserialize the tensor data (ndarray)
// because ArrayD itself might not derive Serialize/Deserialize directly,
// or we want a specific format (e.g., just data and shape).

#[derive(Serialize, Deserialize, Debug)]
struct SerializableTensor {
    shape: Vec<usize>,
    // Store data as a flat Vec<TensorData> for simple serialization
    data: Vec<TensorData>,
}

impl SerializableTensor {
    /// Creates a serializable wrapper from a Tensor.
    /// Requires read access to the tensor's data.
    fn from_tensor(tensor: &Tensor) -> Result<Self, SerializationError> {
        let data_guard = tensor.data.read().map_err(|_| SerializationError::TensorLockError)?;
        // Clone data into a flat Vec. Use `iter()` for portability across layouts.
        let flat_data: Vec<TensorData> = data_guard.iter().cloned().collect();
        Ok(SerializableTensor {
            shape: tensor.shape().to_vec(),
            data: flat_data,
        })
    }

    /// Converts the serializable wrapper back into an ndarray::ArrayD.
    fn to_ndarray(&self) -> Result<ndarray::ArrayD<TensorData>, SerializationError> {
         ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.shape), self.data.clone())
            .map_err(|e| SerializationError::ShapeMismatch{
                 key: "unknown".to_string(), // Key is unknown at this stage
                 expected: vec![], // Don't know expected shape here
                 got: self.shape.clone() , // We know the shape we tried to use
            }) // Propagate potential shape errors from ndarray
           // Refine error: Maybe just a generic serialization error?
           // map_err(|e| SerializationError::Bincode(format!("Failed to create ndarray from shape/vec: {}", e).into())) // bincode::Error doesn't directly fit
             .map_err(|e| SerializationError::Bincode(Box::new(bincode::ErrorKind::Custom(format!("Failed to create ndarray from shape/vec: {}", e)))))

    }
}


// --- State Dictionary Type ---
// Use BTreeMap for consistent ordering (helpful for diffs/debugging).
// Key is the parameter/buffer name (String), Value is the serializable tensor data.
type StateDict = BTreeMap<String, SerializableTensor>;


// --- Save Function ---

/// Saves the state dictionary of a module to a file.
///
/// # Arguments
/// * `module`: The module whose parameters (and optionally buffers) should be saved.
/// * `path`: The file path where the state dictionary will be saved.
/// * `include_buffers`: Whether to include buffers (like running_mean/var in BatchNorm) in the saved state.
pub fn save<P: AsRef<Path>>(
    module: &dyn Module,
    path: P,
    include_buffers: bool,
) -> Result<(), SerializationError> {
    let tensors_to_save = if include_buffers {
        module.tensors() // Get params and buffers
    } else {
        module.parameters() // Get only params
    };

    let mut state_dict: StateDict = BTreeMap::new();
    for (key, tensor) in tensors_to_save {
        // Convert each tensor to its serializable form
        let serializable_tensor = SerializableTensor::from_tensor(&tensor)?;
        state_dict.insert(key, serializable_tensor);
    }

    // Serialize the state_dict using bincode
    let file = File::create(path.as_ref())?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &state_dict)?;

    Ok(())
}


// --- Load Function ---

/// Loads a state dictionary from a file and updates the module's parameters/buffers.
///
/// # Arguments
/// * `module`: The module whose parameters/buffers will be updated.
/// * `path`: The file path from which to load the state dictionary.
/// * `strict`: If `true`, requires that the keys in the loaded state dict exactly match the
///             keys returned by the module's `tensors()` or `parameters()` method (depending on `include_buffers`).
///             Extra or missing keys will cause an error. If `false`, ignores extra keys and
///             skips missing keys.
/// * `include_buffers`: Whether to load buffers in addition to parameters. Should generally match
///                      the setting used during saving.
pub fn load<P: AsRef<Path>>(
    module: &dyn Module, // Need access to module's tensors for updating
    path: P,
    strict: bool,
    include_buffers: bool,
) -> Result<(), SerializationError> {
    // Deserialize the state_dict from the file
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let loaded_state_dict: StateDict = bincode::deserialize_from(reader)?;

    // Get the current tensors from the module
     let module_tensors = if include_buffers {
         module.tensors()
     } else {
         module.parameters()
     };

    // --- Key Matching and Loading ---
    let mut loaded_keys: std::collections::HashSet<&String> = loaded_state_dict.keys().collect();
    let module_keys: std::collections::HashSet<&String> = module_tensors.keys().collect();

    for (key, target_tensor) in module_tensors {
         if let Some(loaded_serializable) = loaded_state_dict.get(&key) {
             // Key found in both module and loaded dict: Load the data
             let loaded_ndarray = loaded_serializable.to_ndarray()?;

             // --- Shape Check ---
             if target_tensor.shape() != loaded_ndarray.shape() {
                  return Err(SerializationError::ShapeMismatch {
                      key: key.clone(),
                      expected: target_tensor.shape().to_vec(),
                      got: loaded_ndarray.shape().to_vec(),
                  });
             }

             // --- Data Update ---
             // Get write access to the target tensor's data
             { // Scope for write lock
                 let mut target_data_guard = target_tensor.data.write()
                     .map_err(|_| SerializationError::TensorLockError)?;
                 // Copy data from loaded_ndarray into the target tensor's storage
                 // Use assign() for efficient copy if layouts match, or element-wise copy otherwise.
                 target_data_guard.assign(&loaded_ndarray);
             } // Write lock released

             // Mark key as used
             loaded_keys.remove(&key);

         } else {
             // Key exists in module but not in loaded dict
             if strict {
                 return Err(SerializationError::MissingKey(key.clone()));
             } else {
                 // Warning or log: Skipping parameter/buffer not found in state_dict
                 eprintln!("Warning: Key '{}' not found in loaded state dict. Skipping.", key);
             }
         }
    }

    // Check for keys that were in the loaded dict but not in the module
    if strict && !loaded_keys.is_empty() {
         // Report the first unexpected key found
         if let Some(unexpected_key) = loaded_keys.iter().next() {
              return Err(SerializationError::UnexpectedKey((*unexpected_key).clone()));
         }
    } else if !strict && !loaded_keys.is_empty() {
         for key in loaded_keys {
             eprintln!("Warning: Key '{}' found in loaded state dict but not used by the module.", key);
         }
    }

    Ok(())
}

// TODO: Implement saving/loading for Optimizer state if needed.
// This would involve:
// 1. Defining serializable structs for optimizer state (e.g., AdamParamState).
// 2. Adding methods to Optimizers to return/load their state.
// 3. Extending save/load functions or creating new ones for optimizer state.