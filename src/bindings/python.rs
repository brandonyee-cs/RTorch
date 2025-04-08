//! # Python Bindings for RTorch (`rtorch_lib`)
//!
//! This module uses PyO3 to expose the Rust RTorch library functionality to Python.
//! It defines the `rtorch` Python module.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::{PyValueError, PyTypeError, PyIndexError, PyRuntimeError};
use pyo3::types::{PyTuple, PyList, PySequence};
use numpy::{PyArrayDyn, IntoPyArray, PyReadonlyArrayDyn}; // For numpy interoperability
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex}; // Ensure Arc/Mutex usage is compatible if exposing internals

// Import necessary items from the core RTorch library
use crate::{
    tensor::{self, Tensor, TensorData, TensorError},
    nn::{self, Module, Linear, ReLU, Sequential, MSELoss, CrossEntropyLoss}, // Import specific modules/losses
    optim::{self, Optimizer, SGD, Adam, Adagrad}, // Import specific optimizers
    utils, // Import utils module, e.g., for serialization
};

// --- Helper to Convert Rust Errors to Python Exceptions ---
// Define a custom Python exception base class (optional)
// pyo3::create_exception!(rtorch_lib, RTorchError, pyo3::exceptions::PyException);

impl std::convert::From<TensorError> for PyErr {
    fn from(err: TensorError) -> PyErr {
        // Map specific TensorErrors to appropriate Python exceptions
        match err {
            TensorError::ShapeMismatch { expected, got } => {
                PyValueError::new_err(format!("Shape mismatch: expected {:?}, got {:?}", expected, got))
            }
            TensorError::IncompatibleShapes { op, shape1, shape2 } => {
                 PyValueError::new_err(format!("Incompatible shapes for operation {}: {:?} and {:?}", op, shape1, shape2))
            }
            TensorError::IndexOutOfBounds { index, shape } => {
                 PyIndexError::new_err(format!("Index out of bounds: index {:?} for shape {:?}", index, shape))
            }
             TensorError::RequiresGradNotSet => {
                 PyRuntimeError::new_err("Operation requires gradient but tensor does not have it")
             }
             TensorError::NoGradient => {
                  PyRuntimeError::new_err("Tensor does not have a gradient computed or gradient is not applicable")
             }
             TensorError::DetachedTensor => {
                  PyRuntimeError::new_err("Cannot perform operation on detached tensor")
             }
            TensorError::NdarrayError(e) => {
                 PyValueError::new_err(format!("Internal ndarray error: {}", e))
             }
             TensorError::AutogradError(msg) => {
                  PyRuntimeError::new_err(format!("Autograd error: {}", msg))
             }
            TensorError::Generic(msg) => PyValueError::new_err(msg), // General errors map to ValueError
        }
    }
}

// --- Tensor Python Wrapper (`rtorch.Tensor`) ---

// Expose the Tensor struct to Python
// Note: Tensor struct already has #[pyclass] in tensor/mod.rs
#[pymethods]
impl Tensor {
    /// Creates a new Tensor from a Python list/tuple or NumPy array.
    #[new]
    #[pyo3(signature = (data, requires_grad=false))] // Default requires_grad to false
    fn py_new(data: &PyAny, requires_grad: bool) -> PyResult<Self> {
        // Try converting from NumPy array first
        if let Ok(py_array) = data.extract::<PyReadonlyArrayDyn<TensorData>>() {
            // Convert NumPy array (read-only view) to owned ndarray ArrayD
            // This involves copying the data.
            let array_view = py_array.as_array();
            let nd_array = array_view.to_owned();
            Ok(Tensor::new(nd_array, requires_grad))
        }
        // Try converting from Python sequences (list/tuple)
        else if let Ok(seq) = data.downcast::<PySequence>() {
            // This is more complex: need to recursively parse sequence to get shape and data
            // For simplicity, let's handle only flat lists for now or delegate to numpy conversion
             // return Err(PyTypeError::new_err("Cannot create tensor directly from nested Python lists/tuples yet. Use NumPy array instead."));

             // Attempt numpy conversion:
             let np = PyModule::import(data.py(), "numpy")?;
             let np_array_obj = np.call_method1("array", (data,))?;
             let py_array = np_array_obj.extract::<PyReadonlyArrayDyn<TensorData>>()?;
             let array_view = py_array.as_array();
             let nd_array = array_view.to_owned();
             Ok(Tensor::new(nd_array, requires_grad))

        } else {
            Err(PyTypeError::new_err(format!("Unsupported data type for Tensor creation: {}", data.get_type().name()?)))
        }
    }

    /// Returns the shape of the tensor as a Python tuple.
    #[getter]
    fn shape(&self) -> PyResult<Py<PyTuple>> {
         let gil = Python::acquire_gil();
         let py = gil.python();
         Ok(PyTuple::new(py, &self.shape).into())
    }

    /// Returns the number of dimensions (rank) of the tensor.
    fn ndim(&self) -> usize {
        self.ndim()
    }

    /// Returns the total number of elements in the tensor.
    fn size(&self) -> usize {
        self.size()
    }

    /// Returns the data type (currently always f32).
    fn dtype(&self) -> String {
         // TODO: Make this dynamic if TensorData becomes generic
        "float32".to_string()
    }

    /// Returns the gradient tensor if it exists.
    #[getter]
    fn grad(&self) -> Option<Tensor> {
         // Call the inherent `grad` method, which returns Option<Tensor>
         self.grad()
         // PyO3 handles Option<PyClass> automatically, converting None to Python None
    }

    /// Returns a NumPy array view/copy of the tensor's data.
    /// Use `.numpy()` convention.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<TensorData>> {
        // We need to provide access to the data.
        // Returning a direct view is complex due to the RwLock.
        // Easiest approach: clone the data and return a new PyArray.
        let data_guard = self.data.read().map_err(|_| PyRuntimeError::new_err("Tensor data lock poisoned"))?;
        Ok(data_guard.clone().into_pyarray(py))
        // Note: This clones the data! A view would be more efficient but harder.
    }

    /// Initiates the backward pass to compute gradients.
    fn backward(&self) -> PyResult<()> {
        // Create the initial gradient (scalar 1.0)
         if !self.is_scalar() {
            return Err(PyValueError::new_err("backward() can only be called on scalar tensors currently."));
         }
        let initial_grad_data = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), 1.0 as TensorData);
        let initial_grad = Tensor::new(initial_grad_data, false);

        // Call the inherent backward method, converting error to PyErr
        tensor::backward(self, initial_grad).map_err(PyErr::from)
    }

     /// Detaches the tensor from the computation graph.
     fn detach(&self) -> PyResult<Tensor> {
         // Call the inherent detach method
         Ok(self.detach())
     }

    /// Returns a string representation of the tensor.
    fn __repr__(&self) -> PyResult<String> {
         // Include data preview, shape, and requires_grad status
         let data_str = format!("{:?}", self.data()); // Use Debug format for data preview
         // Truncate data string if too long?
         let grad_str = if self.requires_grad { ", requires_grad=True" } else { "" };
         Ok(format!("Tensor({}{})", data_str, grad_str))
    }

     fn __str__(&self) -> PyResult<String> {
        self.__repr__()
     }

     // --- Operator Overloading ---
     fn __add__(&self, other: &PyAny) -> PyResult<Tensor> {
        if let Ok(other_tensor) = other.extract::<Tensor>() {
            tensor::ops::add(self, &other_tensor).map_err(PyErr::from)
        } else if let Ok(scalar) = other.extract::<TensorData>() {
            // TODO: Implement scalar addition tensor + scalar
             tensor::ops::add_scalar(self, scalar).map_err(PyErr::from)
            // Err(PyTypeError::new_err("Tensor + scalar addition not implemented yet."))
        }
         else {
             Err(PyTypeError::new_err(format!("Unsupported operand type(s) for +: 'Tensor' and '{}'", other.get_type().name()?)))
         }
     }

      fn __radd__(&self, other: &PyAny) -> PyResult<Tensor> {
          // Handle scalar + tensor
         if let Ok(scalar) = other.extract::<TensorData>() {
             // TODO: Implement scalar addition scalar + tensor
             tensor::ops::add_scalar(self, scalar).map_err(PyErr::from) // Add scalar is commutative for data
            // Err(PyTypeError::new_err("Scalar + tensor addition not implemented yet."))
         } else {
              Err(PyTypeError::new_err(format!("Unsupported operand type(s) for +: '{}' and 'Tensor'", other.get_type().name()?)))
         }
      }


     fn __sub__(&self, other: &PyAny) -> PyResult<Tensor> {
         if let Ok(other_tensor) = other.extract::<Tensor>() {
            tensor::ops::sub(self, &other_tensor).map_err(PyErr::from)
         } else if let Ok(scalar) = other.extract::<TensorData>() {
             // TODO: Implement tensor - scalar
              tensor::ops::sub_scalar(self, scalar).map_err(PyErr::from)
             // Err(PyTypeError::new_err("Tensor - scalar subtraction not implemented yet."))
         } else {
             Err(PyTypeError::new_err(format!("Unsupported operand type(s) for -: 'Tensor' and '{}'", other.get_type().name()?)))
         }
     }

       fn __rsub__(&self, other: &PyAny) -> PyResult<Tensor> {
          // Handle scalar - tensor
         if let Ok(scalar) = other.extract::<TensorData>() {
             // TODO: Implement scalar - tensor
             tensor::ops::rsub_scalar(self, scalar).map_err(PyErr::from)
             // Err(PyTypeError::new_err("Scalar - tensor subtraction not implemented yet."))
         } else {
             Err(PyTypeError::new_err(format!("Unsupported operand type(s) for -: '{}' and 'Tensor'", other.get_type().name()?)))
         }
       }

     fn __mul__(&self, other: &PyAny) -> PyResult<Tensor> {
        if let Ok(other_tensor) = other.extract::<Tensor>() {
            tensor::ops::mul(self, &other_tensor).map_err(PyErr::from)
        } else if let Ok(scalar) = other.extract::<TensorData>() {
            tensor::ops::mul_scalar(self, scalar).map_err(PyErr::from)
        } else {
            Err(PyTypeError::new_err(format!("Unsupported operand type(s) for *: 'Tensor' and '{}'", other.get_type().name()?)))
        }
    }

     fn __rmul__(&self, other: &PyAny) -> PyResult<Tensor> {
         // Handle scalar * tensor
         if let Ok(scalar) = other.extract::<TensorData>() {
             tensor::ops::mul_scalar(self, scalar).map_err(PyErr::from) // Multiplication is commutative
         } else {
             Err(PyTypeError::new_err(format!("Unsupported operand type(s) for *: '{}' and 'Tensor'", other.get_type().name()?)))
         }
     }

    // TODO: Implement __truediv__, __pow__, etc.

    // --- Expose Ops as Methods ---
     #[pyo3(text_signature = "($self, other)")]
    fn add(&self, other: &PyAny) -> PyResult<Tensor> { self.__add__(other) }
     #[pyo3(text_signature = "($self, other)")]
    fn sub(&self, other: &PyAny) -> PyResult<Tensor> { self.__sub__(other) }
     #[pyo3(text_signature = "($self, other)")]
    fn mul(&self, other: &PyAny) -> PyResult<Tensor> { self.__mul__(other) }
    // Alias for mul
     #[pyo3(text_signature = "($self, other)")]
    fn multiply(&self, other: &PyAny) -> PyResult<Tensor> { self.__mul__(other) }


    #[pyo3(text_signature = "($self, other)")]
    fn matmul(&self, other: Tensor) -> PyResult<Tensor> {
        tensor::ops::matmul(self, &other).map_err(PyErr::from)
    }

    #[pyo3(text_signature = "($self)")]
    fn sum(&self) -> PyResult<Tensor> {
        tensor::ops::sum(self).map_err(PyErr::from)
    }

     #[pyo3(text_signature = "($self)")]
    fn mean(&self) -> PyResult<Tensor> {
        tensor::ops::mean(self).map_err(PyErr::from) // Assuming mean op exists
    }

     #[pyo3(text_signature = "($self, shape)")]
     fn reshape(&self, shape: Vec<usize>) -> PyResult<Tensor> {
          tensor::ops::reshape(self, &shape).map_err(PyErr::from)
     }

     // --- In-place methods (use with caution regarding autograd) ---
     // Example: fill_
     // fn fill_(&mut self, value: TensorData) -> PyResult<()> {
     //     // Needs mutable access - requires careful design with PyO3 classes
     //     // Possibly return Self? Or modify internal data via lock.
     //     self.data_mut().fill(value);
     //     Ok(())
     // }

     // --- Indexing ---
     // TODO: Implement __getitem__, __setitem__ (Complex due to ndarray indexing and autograd)


}

// --- Tensor Creation Functions (Exposed at module level `rtorch.*`) ---

#[pyfunction]
#[pyo3(signature = (args, requires_grad=false))]
fn tensor(args: &PyAny, requires_grad: bool) -> PyResult<Tensor> {
    Tensor::py_new(args, requires_grad)
}

#[pyfunction]
#[pyo3(signature = (*py_shape, requires_grad=false))] // Use *args to capture shape tuple
fn zeros(py_shape: &PyTuple, requires_grad: bool) -> PyResult<Tensor> {
    let shape: Vec<usize> = py_shape.extract()?;
    Ok(tensor::zeros(&shape, requires_grad))
}

#[pyfunction]
#[pyo3(signature = (*py_shape, requires_grad=false))]
fn ones(py_shape: &PyTuple, requires_grad: bool) -> PyResult<Tensor> {
    let shape: Vec<usize> = py_shape.extract()?;
    Ok(tensor::ones(&shape, requires_grad))
}

#[pyfunction]
#[pyo3(signature = (*py_shape, requires_grad=false))]
fn rand(py_shape: &PyTuple, requires_grad: bool) -> PyResult<Tensor> {
    let shape: Vec<usize> = py_shape.extract()?;
    // Check if rand feature is enabled in Cargo.toml and compiled
    #[cfg(feature = "rand")]
    {
         tensor::rand(&shape, requires_grad).map_err(PyErr::from)
    }
    #[cfg(not(feature = "rand"))]
    {
         Err(PyRuntimeError::new_err("RTorch was not compiled with the 'rand' feature enabled."))
    }
}

#[pyfunction]
#[pyo3(signature = (*py_shape, requires_grad=false))]
fn randn(py_shape: &PyTuple, requires_grad: bool) -> PyResult<Tensor> {
     let shape: Vec<usize> = py_shape.extract()?;
     #[cfg(feature = "rand")]
     {
         tensor::randn(&shape, requires_grad).map_err(PyErr::from)
     }
     #[cfg(not(feature = "rand"))]
     {
          Err(PyRuntimeError::new_err("RTorch was not compiled with the 'rand' feature enabled."))
     }
}

// --- Neural Network Modules (`rtorch.nn.*`) ---

// Helper macro to implement common Module methods for PyO3
macro_rules! py_module_impl {
    ($struct_name:ty) => {
        #[pymethods]
        impl $struct_name {
             /// Performs the forward pass of the module.
            fn forward(&self, input: Tensor) -> PyResult<Tensor> {
                 // Assuming Module trait's forward takes &self and &Tensor
                Module::forward(self, &input).map_err(PyErr::from)
             }

             /// Returns a dictionary of the module's parameters.
             fn parameters(&self) -> PyResult<BTreeMap<String, Tensor>> {
                 // Clone the tensors returned by the inherent `parameters` method
                Ok(Module::parameters(self).into_iter().collect())
             }

             /// Set the module to training mode.
             fn train(&self) -> PyResult<()> {
                 // Assumes Module::train takes &self (using interior mutability)
                 Module::train(self);
                 Ok(())
             }

             /// Set the module to evaluation mode.
             fn eval(&self) -> PyResult<()> {
                 // Assumes Module::eval takes &self (using interior mutability)
                 Module::eval(self);
                 Ok(())
             }

             /// Zeroes the gradients of all parameters in the module.
             fn zero_grad(&self) -> PyResult<()> {
                  // Assumes Module::zero_grad takes &self (using interior mutability)
                 Module::zero_grad(self);
                 Ok(())
             }

             /// Generic __call__ to behave like PyTorch modules.
             fn __call__(&self, input: Tensor) -> PyResult<Tensor> {
                 self.forward(input)
             }

              fn __repr__(&self) -> PyResult<String> {
                  // Use the Debug representation from Rust
                  Ok(format!("{:?}", self))
              }
        }
    };
}

// Expose Linear layer
#[pyclass(extends=PyAny, subclass)]
#[derive(Debug, Clone)] // Clone might be needed for DataParallel or other wrappers
struct PyLinear {
    module: Linear, // Store the Rust module internally
}
#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true))]
    fn new(in_features: usize, out_features: usize, bias: bool) -> PyResult<Self> {
        let module = Linear::new_with_init(in_features, out_features, bias); // Use constructor with Rust-side init
        Ok(PyLinear { module })
    }
    // Forward, parameters, etc. are needed
    fn forward(&self, input: Tensor) -> PyResult<Tensor> {
        Module::forward(&self.module, &input).map_err(PyErr::from)
    }
    fn parameters(&self) -> PyResult<BTreeMap<String, Tensor>> {
        Ok(Module::parameters(&self.module))
    }
     fn train(&self) -> PyResult<()> { Ok(()) } // Linear has no training state
     fn eval(&self) -> PyResult<()> { Ok(()) }
     fn zero_grad(&self) -> PyResult<()> { Module::zero_grad(&self.module); Ok(()) }
     fn __call__(&self, input: Tensor) -> PyResult<Tensor> { self.forward(input) }
     fn __repr__(&self) -> PyResult<String> { Ok(format!("{:?}", self.module)) }

     // Expose weight and bias as properties
     #[getter]
     fn weight(&self) -> PyResult<Tensor> {
        Ok(self.module.weight.clone())
     }
     #[getter]
     fn bias(&self) -> PyResult<Option<Tensor>> {
        Ok(self.module.bias.clone())
     }

}
// py_module_impl!(PyLinear); // Macro won't work easily with internal module field access

// Expose ReLU layer
#[pyclass(extends=PyAny, subclass)]
#[derive(Debug, Clone, Copy)]
struct PyReLU {
     module: ReLU,
}
#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> Self { PyReLU { module: ReLU::new() } }
     // Forward, parameters, etc.
     fn forward(&self, input: Tensor) -> PyResult<Tensor> { Module::forward(&self.module, &input).map_err(PyErr::from) }
     fn parameters(&self) -> PyResult<BTreeMap<String, Tensor>> { Ok(Module::parameters(&self.module)) }
     fn train(&self) -> PyResult<()> { Ok(()) } // ReLU has no training state
     fn eval(&self) -> PyResult<()> { Ok(()) }
     fn zero_grad(&self) -> PyResult<()> { Ok(()) } // No params
     fn __call__(&self, input: Tensor) -> PyResult<Tensor> { self.forward(input) }
     fn __repr__(&self) -> PyResult<String> { Ok(format!("{:?}", self.module)) }
}
// py_module_impl!(PyReLU);

// Expose Sequential container
// This is tricky because it holds dyn Module. How to expose this?
// Option 1: Create a PySequential holding Vec<PyObject> representing Python modules.
// Option 2: Create a PySequential holding Vec<Arc<Mutex<dyn Module>>> if possible? Less pythonic.
// Let's try Option 1.
#[pyclass(extends=PyAny, subclass)]
struct PySequential {
     // Store Python objects directly
     modules: Vec<PyObject>,
}
#[pymethods]
impl PySequential {
     #[new]
     fn new(modules: &PySequence) -> PyResult<Self> {
         let mut py_modules = Vec::new();
         for item in modules.iter()? {
             let module_obj = item?;
             // TODO: Check if module_obj is actually a valid module (e.g., has 'forward')?
             py_modules.push(module_obj.to_object(modules.py()));
         }
         Ok(PySequential { modules: py_modules })
     }

     fn forward(&self, input: Tensor) -> PyResult<Tensor> {
         let gil = Python::acquire_gil();
         let py = gil.python();
         let mut current_tensor = input.into_py(py); // Convert initial Tensor to PyObject

         for module_obj in &self.modules {
             // Call the forward method of the Python module object
             current_tensor = module_obj.call_method1(py, "forward", (current_tensor.clone_ref(py),))?;
         }
         // Extract the final Tensor from the PyObject
         current_tensor.extract(py)
     }

     fn parameters(&self, py: Python) -> PyResult<BTreeMap<String, Tensor>> {
          // Need to iterate through python modules, call their parameters(), and aggregate.
          // This requires Python-side logic or careful FFI.
          // For simplicity, we might skip exposing parameters directly here,
          // assuming users access parameters of contained modules directly.
          Err(PyRuntimeError::new_err("parameters() not implemented for PySequential yet."))
     }

      fn __call__(&self, input: Tensor) -> PyResult<Tensor> {
          self.forward(input)
      }

      // TODO: Implement train, eval, zero_grad by iterating through Python module objects.
      // TODO: Implement __repr__
}


// --- Functional Interface (`rtorch.nn.functional.*`) ---
#[pyfunction]
fn relu(input: Tensor) -> PyResult<Tensor> {
    nn::functional::relu(&input).map_err(PyErr::from)
}

#[pyfunction]
fn sigmoid(input: Tensor) -> PyResult<Tensor> {
    nn::functional::sigmoid(&input).map_err(PyErr::from)
}

#[pyfunction]
fn tanh(input: Tensor) -> PyResult<Tensor> {
    nn::functional::tanh(&input).map_err(PyErr::from)
}

#[pyfunction]
fn mse_loss(input: Tensor, target: Tensor) -> PyResult<Tensor> {
    nn::functional::mse_loss(&input, &target).map_err(PyErr::from)
}

#[pyfunction]
fn cross_entropy_loss(input: Tensor, target: Tensor) -> PyResult<Tensor> {
     // Note: Target tensor needs to be integer type. Current Tensor only supports f32.
     // This highlights the need for generic Tensor types or type checking/conversion.
     // For now, assume target data can be handled correctly by the backend op (e.g., gather).
    nn::functional::cross_entropy_loss(&input, &target).map_err(PyErr::from)
}

// --- Optimizers (`rtorch.optim.*`) ---

// Helper macro? Or implement manually.
#[pyclass(extends=PyAny, subclass)]
struct PySGD {
    optimizer: SGD,
}
#[pymethods]
impl PySGD {
     #[new]
     #[pyo3(signature = (params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=false))]
     fn new(params: &PySequence, lr: TensorData, momentum: TensorData, dampening: TensorData, weight_decay: TensorData, nesterov: bool) -> PyResult<Self> {
        let mut rust_params = Vec::new();
        for item in params.iter()? {
            rust_params.push(item?.extract::<Tensor>()?);
        }
         // Refined SGD::new signature assumed here:
         let optimizer = SGD::new(rust_params, lr, Some(momentum), Some(dampening), Some(weight_decay), Some(nesterov))?; // Pass Nesterov flag correctly
         Ok(PySGD { optimizer })
     }

     fn step(&mut self) -> PyResult<()> {
         self.optimizer.step().map_err(PyErr::from)
     }

     fn zero_grad(&mut self) -> PyResult<()> {
          self.optimizer.zero_grad();
          Ok(())
     }
      fn __repr__(&self) -> PyResult<String> { Ok(format!("{:?}", self.optimizer)) } // Need Debug on SGD
}


#[pyclass(extends=PyAny, subclass)]
struct PyAdam {
     optimizer: Adam,
}
#[pymethods]
impl PyAdam {
     #[new]
     #[pyo3(signature = (params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=false))]
     fn new(params: &PySequence, lr: TensorData, betas: (TensorData, TensorData), eps: TensorData, weight_decay: TensorData, amsgrad: bool) -> PyResult<Self> {
         let rust_params: Vec<Tensor> = params.iter()?.map(|p| p?.extract::<Tensor>()).collect::<PyResult<_>>()?;
         let optimizer = Adam::new(rust_params, Some(lr), Some(betas), Some(eps), Some(weight_decay), amsgrad)?;
         Ok(PyAdam { optimizer })
     }
      fn step(&mut self) -> PyResult<()> { self.optimizer.step().map_err(PyErr::from) }
      fn zero_grad(&mut self) -> PyResult<()> { self.optimizer.zero_grad(); Ok(()) }
      fn __repr__(&self) -> PyResult<String> { Ok(format!("{:?}", self.optimizer)) } // Need Debug on Adam
}


// --- Utilities (`rtorch.utils.*`) ---
#[pyfunction]
#[pyo3(signature = (module, path, include_buffers = true))]
fn save(module: &PyAny, path: String, include_buffers: bool) -> PyResult<()> {
     // How to get a `&dyn Module` from a PyObject? This is hard.
     // We might need the Python module wrappers (PyLinear etc.) to implement a trait
     // that allows access to the underlying Rust module.
     // Or, we require the user to pass the Rust object if available (less pythonic).

     // Alternative: Implement saving logic on the Python side using parameters()
     // and tensor.numpy() ? Less efficient, loses Rust type safety.

     // Simplification: Assume the PyObject *is* one of our wrapped types (e.g. PyLinear)
     // and try to extract the Rust module. This is not robust.
      return Err(PyRuntimeError::new_err("Saving directly from Python objects not fully implemented yet. Consider saving parameters manually using .parameters() and numpy.savez".to_string()));

     // --- Conceptual (if we could get &dyn Module) ---
     // let rust_module: &dyn Module = ... extract from PyAny ...;
     // utils::serialization::save(rust_module, path, include_buffers).map_err(PyErr::from)?;
     // Ok(())
}

#[pyfunction]
#[pyo3(signature = (module, path, strict = true, include_buffers = true))]
fn load(module: &PyAny, path: String, strict: bool, include_buffers: bool) -> PyResult<()> {
      // Similar challenge as save() - how to get a mutable reference to the underlying Rust module?
     return Err(PyRuntimeError::new_err("Loading directly into Python objects not fully implemented yet. Consider loading manually using numpy.load and assigning to parameters.".to_string()));

      // --- Conceptual ---
      // let rust_module: &mut dyn Module = ... extract mut ref ...;
      // utils::serialization::load(rust_module, path, strict, include_buffers).map_err(PyErr::from)?;
      // Ok(())
}


// --- Main Python Module Definition (`rtorch`) ---
#[pymodule]
fn rtorch(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add Tensor class
    m.add_class::<Tensor>()?;

    // Add Tensor creation functions
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;

    // --- nn submodule ---
    let nn_module = PyModule::new(_py, "nn")?;
    // Add nn Modules
    nn_module.add_class::<PyLinear>()?;
    nn_module.add_class::<PyReLU>()?;
    nn_module.add_class::<PySequential>()?;
    // Add Loss classes (if exposing them as classes)
    // nn_module.add_class::<MSELoss>()?; // Or maybe not, favour functional
    // nn_module.add_class::<CrossEntropyLoss>()?;

    // Add nn.functional submodule
    let functional_module = PyModule::new(_py, "functional")?;
    functional_module.add_function(wrap_pyfunction!(relu, functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(sigmoid, functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(tanh, functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(mse_loss, functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(cross_entropy_loss, functional_module)?)?;
    nn_module.add_submodule(functional_module)?;

    m.add_submodule(nn_module)?;


    // --- optim submodule ---
    let optim_module = PyModule::new(_py, "optim")?;
    optim_module.add_class::<PySGD>()?;
    optim_module.add_class::<PyAdam>()?;
    // optim_module.add_class::<PyAdagrad>()?; // Add if wrapper created
    m.add_submodule(optim_module)?;

     // --- utils submodule ---
     let utils_module = PyModule::new(_py, "utils")?;
     utils_module.add_function(wrap_pyfunction!(save, utils_module)?)?;
     utils_module.add_function(wrap_pyfunction!(load, utils_module)?)?;
     m.add_submodule(utils_module)?;

    // Add custom error (optional)
    // m.add("RTorchError", _py.get_type::<RTorchError>())?;

    Ok(())
}