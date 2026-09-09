mod dequantize;
#[cfg(feature = "vision")]
mod sigmoid;
#[cfg(feature = "vision")]
mod softmax;

pub(super) use dequantize::Dequantize;
#[cfg(feature = "vision")]
pub(super) use sigmoid::Sigmoid;
#[cfg(feature = "vision")]
pub(super) use softmax::Softmax;

pub use dequantize::QuantizationParameters;

#[cfg(feature = "vision")]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Default)]
pub enum Activation {
    #[default]
    None,
    Sigmoid,
    Softmax,
}
