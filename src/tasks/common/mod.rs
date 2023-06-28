#[macro_use]
mod options;

#[macro_use]
#[cfg(feature = "vision")]
mod detection_common_impl;

pub(crate) use options::*;
