use super::{HandDetectorBuilder, HandLandmarker};

use crate::model::ZipFiles;
use crate::tasks::common::{BaseTaskOptions, HandLandmarkOptions};

/// Configure the build options of a new **Hand Landmark** task instance.
///
/// Methods can be chained on it in order to configure it.
#[derive(Default)]
pub struct HandLandmarkerBuilder {
    pub(in super::super) base_task_options: BaseTaskOptions,
    pub(in super::super) hand_landmark_options: HandLandmarkOptions,
}

impl HandLandmarkerBuilder {
    /// Create a new builder with default options.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    base_task_options_impl!(HandLandmarker);

    hand_landmark_options_impl!();

    pub const HAND_DETECTOR_CANDIDATE_NAMES: &'static [&'static str] = &["hand_detector.tflite"];
    pub const HAND_LANDMARKS_CANDIDATE_NAMES: &'static [&'static str] =
        &["hand_landmarks_detector.tflite"];

    /// Use the current build options and use the buffer as model data to create a new task instance.
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<HandLandmarker, crate::Error> {
        self.hand_landmark_options.check()?;
        let buf = buffer.as_ref();

        let zip_file = ZipFiles::new(buf)?;
        let landmark_file = crate::model::search_file_in_zip(
            &zip_file,
            Self::HAND_LANDMARKS_CANDIDATE_NAMES,
            "HandLandmark",
        )?;
        let hand_detection_file = crate::model::search_file_in_zip(
            &zip_file,
            Self::HAND_DETECTOR_CANDIDATE_NAMES,
            "HandDetection",
        )?;

        let subtask = HandDetectorBuilder::new()
            .device(self.base_task_options.device)
            .num_hands(self.hand_landmark_options.num_hands)
            .min_detection_confidence(self.hand_landmark_options.min_hand_detection_confidence)
            .build_from_buffer(hand_detection_file)?;

        // parse model and get model resources.
        let model_resource = crate::model::parse_model(landmark_file)?;

        // check model
        model_resource.check_tensor_counts(Some(1), 4)?;
        model_resource.expect_to_tensor_info(0)?.try_to_image()?;
        let input_tensor_type = model_resource.expect_input_tensor_type(0)?;

        // todo: get these from metadata
        let handedness_buf_index = 2;
        let score_buf_index = 1;
        let landmarks_buf_index = 0;
        let world_landmarks_buf_index = 3;
        // now only fp32 model
        model_resource.check_scalar_f32_output(handedness_buf_index)?;
        model_resource.check_scalar_f32_output(score_buf_index)?;

        let graph = crate::tasks::common::build_graph(
            model_resource.as_ref(),
            self.base_task_options.device,
            landmark_file,
        )?;

        Ok(HandLandmarker {
            build_options: self,
            model_resource,
            graph,
            hand_detector: subtask,
            handedness_buf_index,
            score_buf_index,
            landmarks_buf_index,
            world_landmarks_buf_index,
            input_tensor_type,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::tasks::vision::HandLandmarkerBuilder;
    use crate::Error;

    #[test]
    fn test_nan_confidence_is_rejected() {
        for builder in [
            HandLandmarkerBuilder::new().min_hand_detection_confidence(f32::NAN),
            HandLandmarkerBuilder::new().min_hand_presence_confidence(f32::NAN),
        ] {
            assert!(matches!(
                builder.build_from_buffer(Vec::<u8>::new()),
                Err(Error::ArgumentError(_))
            ));
        }
    }
}
