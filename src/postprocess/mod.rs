/// result containers
mod containers;
pub use containers::*;

/// stateless operators for tensor
mod ops;
#[cfg(feature = "vision")]
pub(crate) use ops::Activation;
pub(crate) use ops::QuantizationParameters;

/// stateful objects, convert tensor to results
mod processing;
pub(crate) use processing::*;

/// Utils to make use of task results, such as drawing utils.
pub mod utils;
