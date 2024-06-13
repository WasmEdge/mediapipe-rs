/// result containers
mod containers;
pub use containers::*;

/// stateless operators for tensor
mod ops;
pub(crate) use ops::{Activation, QuantizationParameters};

/// stateful objects, convert tensor to results
mod processing;
pub(crate) use processing::*;

/// Utils to make use of task results, such as drawing utils.
pub mod utils;
