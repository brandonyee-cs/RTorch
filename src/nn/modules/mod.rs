//! # Neural Network Layer Modules
//!
//! Contains implementations of common neural network layers (modules).

// --- Re-export Layer Implementations ---
pub mod linear;
pub use linear::Linear;

pub mod activation;
pub use activation::ReLU; // Example activation module
// pub use activation::Sigmoid; // If implemented
// pub use activation::Tanh;    // If implemented

pub mod dropout;
pub use dropout::Dropout;

// --- Placeholders for more complex layers ---
pub mod conv;
// pub use conv::Conv2d; // If implemented

pub mod rnn;
// pub use rnn::LSTM; // If implemented
// pub use rnn::GRU;  // If implemented

pub mod normalization;
// pub use normalization::BatchNorm2d; // If implemented
// pub use normalization::LayerNorm;   // If implemented

// TODO: Add other modules as needed (Pooling, Embedding, etc.)