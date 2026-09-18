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

/// Check that a confidence option lies in `[0.0, 1.0]`.
#[cfg(feature = "vision")]
pub(crate) fn check_confidence(name: &str, value: f32) -> Result<(), crate::Error> {
    if !(0.0..=1.0).contains(&value) {
        return Err(crate::Error::ArgumentError(format!(
            "The {} must in range [0.0, 1.0], but got `{}`",
            name, value
        )));
    }
    Ok(())
}
