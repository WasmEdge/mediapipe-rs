#![allow(unused)]

#[macro_use]
#[cfg(any(feature = "audio", feature = "vision"))]
mod results_iter_impl;

mod category;
mod classification_result;
mod embedding_result;

pub use category::*;
pub use classification_result::*;
pub use embedding_result::*;
