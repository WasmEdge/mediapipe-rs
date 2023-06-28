use super::{HandDetectorBuilder, HandLandmarker, TensorType};

use crate::model::ZipFiles;
use crate::tasks::common::{BaseTaskOptions, HandLandmarkOptions};

/// Configure the build options of a new **Hand Landmark** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct HandLandmarkerBuilder {
    pub(in super::super) base_task_options: BaseTaskOptions,
    pub(in super::super) hand_landmark_options: HandLandmarkOptions,
}

impl Default for HandLandmarkerBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            hand_landmark_options: Default::default(),
        }
    }
}

impl HandLandmarkerBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            hand_landmark_options: Default::default(),
        }
    }

    base_task_options_impl!(HandLandmarker);

    hand_landmark_options_impl!();

    pub const HAND_DETECTOR_CANDIDATE_NAMES: &'static [&'static str] = &["hand_detector.tflite"];
    pub const HAND_LANDMARKS_CANDIDATE_NAMES: &'static [&'static str] =
        &["hand_landmarks_detector.tflite"];

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<HandLandmarker, crate::Error> {
        hand_landmark_options_check!(self);
        let buf = buffer.as_ref();

        let zip_file = ZipFiles::new(buf)?;
        let landmark_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::HAND_LANDMARKS_CANDIDATE_NAMES,
            "HandLandmark"
        );
        let hand_detection_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::HAND_DETECTOR_CANDIDATE_NAMES,
            "HandDetection"
        );

        let subtask = HandDetectorBuilder::new()
            .device(self.base_task_options.device)
            .num_hands(self.hand_landmark_options.num_hands)
            .min_detection_confidence(self.hand_landmark_options.min_hand_detection_confidence)
            .build_from_buffer(hand_detection_file)?;

        // parse model and get model resources.
        let model_resource = crate::model::parse_model(landmark_file.as_ref())?;

        // check model
        model_base_check_impl!(model_resource, 1, 4);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        // todo: get these from metadata
        let handedness_buf_index = 2;
        let score_buf_index = 1;
        let landmarks_buf_index = 0;
        let world_landmarks_buf_index = 3;
        // now only fp32 model
        check_tensor_type!(
            model_resource,
            handedness_buf_index,
            output_tensor_type,
            TensorType::F32
        );
        check_tensor_type!(
            model_resource,
            score_buf_index,
            output_tensor_type,
            TensorType::F32
        );

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.device,
        )
        .build_from_bytes([landmark_file])?;

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
