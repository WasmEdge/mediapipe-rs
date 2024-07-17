mod face_detection;
mod face_landmark;
mod gesture_recognition;
mod hand_detection;
mod hand_landmark;
mod image_classification;
mod image_embedding;
mod image_segmentation;
mod object_detection;

pub use face_detection::{FaceDetector, FaceDetectorBuilder, FaceDetectorSession};
pub use face_landmark::{
    FaceLandmarkBlendshapes, FaceLandmarkConnections, FaceLandmarker, FaceLandmarkerBuilder,
    FaceLandmarkerSession,
};
pub use gesture_recognition::{
    GestureRecognizer, GestureRecognizerBuilder, GestureRecognizerSession,
};
pub use hand_detection::{HandDetector, HandDetectorBuilder, HandDetectorSession};
pub use hand_landmark::{
    HandLandmark, HandLandmarker, HandLandmarkerBuilder, HandLandmarkerSession,
};
pub use image_classification::{ImageClassifier, ImageClassifierBuilder, ImageClassifierSession};
pub use image_embedding::{ImageEmbedder, ImageEmbedderBuilder, ImageEmbedderSession};
pub use image_segmentation::{ImageSegmenter, ImageSegmenterBuilder, ImageSegmenterSession};
pub use object_detection::{ObjectDetector, ObjectDetectorBuilder, ObjectDetectorSession};

/// Re-export some task results types
pub mod results {
    pub use super::face_landmark::{FaceLandmarkResult, FaceLandmarkResults};
    pub use super::gesture_recognition::{GestureRecognizerResult, GestureRecognizerResults};
    pub use super::hand_landmark::{HandLandmarkResult, HandLandmarkResults};
    pub use super::image_segmentation::ImageSegmentationResult;
}

/// Task session trait to process the video stream data
pub trait TaskSession {
    type Result: 'static;

    /// Process the next tensors from input stream
    fn process_next(
        &mut self,
        process_options: &ImageProcessingOptions,
        video_data: &mut impl crate::preprocess::vision::VideoData,
    ) -> Result<Option<Self::Result>, crate::Error>;
}

/// Options for image processing. Set **Region of Interest** for image before processing task.
///
/// If both region-or-interest and rotation are specified, the crop around the
/// region-of-interest is extracted first, then the specified rotation is applied to the crop.
#[derive(Clone, Debug)]
pub struct ImageProcessingOptions {
    pub(crate) region_of_interest: Option<crate::postprocess::CropRect>,
    /// clockwise, in radian
    pub(crate) rotation: f32,
}

impl Default for ImageProcessingOptions {
    #[inline(always)]
    fn default() -> Self {
        Self {
            region_of_interest: None,
            rotation: 0.,
        }
    }
}

impl ImageProcessingOptions {
    /// Create a new instance with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the rotation to apply to the image (or cropped region-of-interest), in degrees clockwise.
    ///
    /// The rotation must be a multiple (positive or negative) of 90°.
    /// default is 0.
    #[inline(always)]
    pub fn rotation_degrees(mut self, mut rotation_degrees: i32) -> Result<Self, crate::Error> {
        if rotation_degrees % 90 != 0 {
            return Err(crate::Error::ArgumentError(format!(
                "The rotation must be a multiple (positive or negative) of 90°, but got `{}`",
                rotation_degrees
            )));
        }
        rotation_degrees = rotation_degrees % 360;
        if rotation_degrees < 0 {
            rotation_degrees += 360;
        }
        self.rotation = rotation_degrees as f32 * std::f32::consts::PI / 180.0;
        Ok(self)
    }

    /// Set the optional region-of-interest to crop from the image.
    /// If not specified, the full image is used.
    ///
    /// Coordinates must be in \[0,1\] with 'left' < 'right' and 'top' < bottom.
    #[inline(always)]
    pub fn region_of_interest(
        mut self,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    ) -> Result<Self, crate::Error> {
        self.region_of_interest =
            Some(crate::postprocess::CropRect::new(left, top, right, bottom)?);
        Ok(self)
    }

    #[inline]
    pub(crate) fn from_normalized_rect(rect: &crate::postprocess::NormalizedRect) -> Self {
        Self {
            region_of_interest: Some(crate::postprocess::CropRect::from(rect)),
            rotation: -rect.rotation.unwrap_or(0.),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_image_process_options_check() {
        let default: ImageProcessingOptions = Default::default();
        assert_eq!(default.rotation, 0.);
        assert!(default.region_of_interest.is_none());

        assert!(ImageProcessingOptions::new().rotation_degrees(10).is_err());
        assert!(ImageProcessingOptions::new().rotation_degrees(-10).is_err());
        assert!(ImageProcessingOptions::new().rotation_degrees(-180).is_ok());
        assert!(ImageProcessingOptions::new().rotation_degrees(270).is_ok());
    }
}
