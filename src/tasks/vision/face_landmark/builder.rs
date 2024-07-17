use super::{FaceDetectorBuilder, FaceLandmarker, TensorType};

use crate::model::ZipFiles;
use crate::tasks::common::{BaseTaskOptions, FaceLandmarkOptions};

/// Configure the build options of a new **Face Landmark** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct FaceLandmarkerBuilder {
    pub(in super::super) base_task_options: BaseTaskOptions,
    pub(in super::super) face_landmark_options: FaceLandmarkOptions,
}

impl Default for FaceLandmarkerBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            face_landmark_options: Default::default(),
        }
    }
}

impl FaceLandmarkerBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            face_landmark_options: Default::default(),
        }
    }

    base_task_options_impl!(FaceLandmarker);

    face_landmark_options_impl!();

    pub const FACE_DETECTOR_CANDIDATE_NAMES: &'static [&'static str] = &["face_detector.tflite"];
    pub const FACE_LANDMARKS_CANDIDATE_NAMES: &'static [&'static str] =
        &["face_landmarks_detector.tflite"];

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<FaceLandmarker, crate::Error> {
        face_landmark_options_check!(self);
        let buf = buffer.as_ref();

        let zip_file = ZipFiles::new(buf)?;
        let face_detection_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::FACE_DETECTOR_CANDIDATE_NAMES,
            "FaceDetection"
        );
        let landmark_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::FACE_LANDMARKS_CANDIDATE_NAMES,
            "FaceLandmark"
        );

        let subtask_face_detector = FaceDetectorBuilder::new()
            .device(self.base_task_options.device)
            .num_faces(self.face_landmark_options.num_faces)
            .min_detection_confidence(self.face_landmark_options.min_face_detection_confidence)
            .build_from_buffer(face_detection_file)?;

        // parse model and get model resources.
        let model_resource = crate::model::parse_model(landmark_file.as_ref())?;

        // check model
        model_base_check_impl!(model_resource, 1, 3);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        // todo: get these from metadata
        let score_buf_index = 1;
        let landmarks_buf_index = 0;
        // now only fp32 model
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

        Ok(FaceLandmarker {
            build_options: self,
            model_resource,
            graph,
            face_detector: subtask_face_detector,
            score_buf_index,
            landmarks_buf_index,
            input_tensor_type,
        })
    }
}
