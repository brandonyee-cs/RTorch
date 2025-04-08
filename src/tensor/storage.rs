//! # Tensor Storage
//!
//! Defines the underlying memory storage for Tensors.
//! Currently, this uses `ndarray::ArrayD` for CPU storage.

use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, RwLock}; // Import necessary synchronization primitives

use super::TensorData; // Use the globally defined TensorData type

// --- Storage Trait (Optional Future Extension) ---
// pub trait StorageTrait {
//     type Elem; // Element type (e.g., f32)
//     // Methods for allocation, data access, device management, etc.
//     fn get_data(&self) -> &[Self::Elem];
//     fn get_data_mut(&mut self) -> &mut [Self::Elem];
//     fn shape(&self) -> &[usize];
//     fn device(&self) -> Device; // Example for device placement
// }

// --- Concrete CPU Storage ---

/// Represents the CPU memory backing a Tensor.
///
/// It wraps the actual `ndarray` array within synchronization primitives
/// (`Arc<RwLock<...>>`) to allow safe sharing and modification across
/// different parts of the computation graph (including gradients).
#[derive(Debug, Clone)]
pub struct CpuStorage {
    /// The core data store using ndarray's dynamic dimension array,
    /// protected for thread-safe access.
    /// `pub(crate)` allows access within the `tensor` module but not outside.
    pub(crate) data: Arc<RwLock<ArrayD<TensorData>>>,
    // We could store shape/device here, but currently Tensor struct holds shape,
    // and device is implicitly CPU.
}

impl CpuStorage {
    /// Creates a new `CpuStorage` instance from an existing `ndarray::ArrayD`.
    pub fn new(array_data: ArrayD<TensorData>) -> Self {
        CpuStorage {
            data: Arc::new(RwLock::new(array_data)),
        }
    }

    /// Creates a new `CpuStorage` instance initialized with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let array_data = ArrayD::zeros(IxDyn(shape));
        Self::new(array_data)
    }

    /// Creates a new `CpuStorage` instance initialized with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let array_data = ArrayD::ones(IxDyn(shape));
        Self::new(array_data)
    }

    // Add other creation methods as needed (e.g., from_slice, random, etc.)

    /// Provides read access to the underlying `ndarray`.
    /// Locks the `RwLock` for reading. Panics if the lock is poisoned.
    pub fn read_lock(&self) -> std::sync::RwLockReadGuard<'_, ArrayD<TensorData>> {
        self.data.read().expect("CPU Storage RwLock poisoned (read)")
    }

    /// Provides write access to the underlying `ndarray`.
    /// Locks the `RwLock` for writing. Panics if the lock is poisoned.
    /// Use with caution regarding side effects, especially with autograd.
    pub fn write_lock(&self) -> std::sync::RwLockWriteGuard<'_, ArrayD<TensorData>> {
        self.data.write().expect("CPU Storage RwLock poisoned (write)")
    }

    /// Returns the shape of the stored data.
    /// Acquires a read lock temporarily.
    pub fn shape(&self) -> Vec<usize> {
        self.read_lock().shape().to_vec()
    }

    /// Returns the number of dimensions (rank) of the stored data.
    /// Acquires a read lock temporarily.
    pub fn ndim(&self) -> usize {
        self.read_lock().ndim()
    }

     /// Returns the total number of elements in the storage.
    /// Acquires a read lock temporarily.
     pub fn size(&self) -> usize {
        self.read_lock().len() // or .size() depending on ndarray version API
    }

    // --- Methods below might be useful if implementing a StorageTrait ---
    // fn get_data(&self) -> &[Self::Elem] {
    //     // This is tricky with RwLockGuard. We might need unsafe code
    //     // or a different API that returns the guard or uses closures.
    //     unimplemented!("Direct slice access needs careful design with RwLock")
    // }
    // fn get_data_mut(&mut self) -> &mut [Self::Elem] {
    //     unimplemented!("Direct mutable slice access needs careful design with RwLock")
    // }
    // fn device(&self) -> Device {
    //     Device::Cpu
    // }
}

// --- Device Enum ---
/// Represents the device where the tensor data resides.
/// Currently only CPU is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    // Gpu(u32), // Placeholder for future GPU device ID
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

// Display trait for better printing
impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            // Device::Gpu(id) => write!(f, "gpu:{}", id),
        }
    }
}