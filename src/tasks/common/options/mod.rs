#![allow(unused)]

#[macro_use]
mod base_task_options;

#[macro_use]
mod classification_options;

#[macro_use]
mod embedding_options;

#[macro_use]
#[cfg(feature = "vision")]
mod hand_landmark_options;

pub(crate) use base_task_options::BaseTaskOptions;
pub(crate) use classification_options::ClassificationOptions;
pub(crate) use embedding_options::EmbeddingOptions;
#[cfg(feature = "vision")]
pub(crate) use hand_landmark_options::HandLandmarkOptions;
