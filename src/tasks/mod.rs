#[macro_use]
pub(crate) mod common;

#[cfg(feature = "audio")]
pub mod audio;

#[cfg(feature = "text")]
pub mod text;

#[cfg(feature = "vision")]
pub mod vision;
