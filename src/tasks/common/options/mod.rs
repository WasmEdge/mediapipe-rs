#[macro_use]
mod base_task_options;

#[macro_use]
mod classification_options;

#[macro_use]
#[cfg(any(feature = "vision", feature = "text"))]
mod embedding_options;

#[macro_use]
#[cfg(feature = "vision")]
mod hand_landmark_options;
#[macro_use]
#[cfg(feature = "vision")]
mod face_landmark_options;

pub(crate) use base_task_options::BaseTaskOptions;
pub(crate) use classification_options::ClassificationOptions;
#[cfg(any(feature = "vision", feature = "text"))]
pub(crate) use embedding_options::EmbeddingOptions;
#[cfg(feature = "vision")]
pub(crate) use face_landmark_options::FaceLandmarkOptions;
#[cfg(feature = "vision")]
pub(crate) use hand_landmark_options::HandLandmarkOptions;
