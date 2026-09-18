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
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum Activation {
    None,
    SIGMOID,
    SOFTMAX,
}

#[cfg(feature = "vision")]
impl Default for Activation {
    #[inline(always)]
    fn default() -> Self {
        Self::None
    }
}
