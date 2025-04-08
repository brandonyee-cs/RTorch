//! # RTorch Core Library
//!
//! This crate provides the core Rust implementation for RTorch, including
//! tensors, neural network modules, optimizers, and utilities.
//! It's designed to be used both directly in Rust and via Python bindings.

// Re-export key components for easier use, especially from Python bindings
pub mod tensor;
pub mod nn;
pub mod optim;
pub mod utils;
pub mod bindings; // Module specifically for PyO3 bindings setup

// Optional: Define common error types or prelude
// pub mod error;
// pub mod prelude {
//     pub use crate::tensor::Tensor;
//     pub use crate::nn::Module;
//     // ... other commonly used items
// }

// When the "python" feature is enabled (or by default if bindings are core),
// expose the Python module. This logic might live in src/bindings/python.rs instead.
// #[cfg(feature = "python")]
// pub use bindings::python::rtorch_module;