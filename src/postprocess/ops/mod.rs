#![allow(unused)]

mod dequantize;
mod sigmoid;
mod softmax;

pub(super) use dequantize::Dequantize;
pub(super) use sigmoid::Sigmoid;
pub(super) use softmax::Softmax;

pub use dequantize::QuantizationParameters;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum Activation {
    None,
    SIGMOID,
    SOFTMAX,
}

impl Default for Activation {
    #[inline(always)]
    fn default() -> Self {
        Self::None
    }
}
