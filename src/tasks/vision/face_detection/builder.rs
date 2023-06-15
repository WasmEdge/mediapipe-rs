use super::FaceDetector;
use crate::postprocess::SsdAnchorsBuilder;
use crate::tasks::common::BaseTaskOptions;

/// Configure the build options of a new **Face Detection** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct FaceDetectorBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    /// The maximum number of faces output by the detector.
    pub(super) num_faces: i32,
    /// The minimum confidence score for the face detection to be considered successful.
    pub(super) min_detection_confidence: f32,
    /// The minimum non-maximum-suppression threshold for face detection to be considered overlapped.
    pub(super) min_suppression_threshold: f32,
}

impl Default for FaceDetectorBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl FaceDetectorBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            num_faces: -1,
            min_detection_confidence: 0.5,
            min_suppression_threshold: 0.3,
        }
    }

    base_task_options_impl!(FaceDetector);

    /// Set the maximum number of faces can be detected by the HandDetector.
    /// Default is -1, (no limits)
    #[inline(always)]
    pub fn num_faces(mut self, num_faces: i32) -> Self {
        self.num_faces = num_faces;
        self
    }

    /// Set the minimum confidence score for the face detection to be considered successful.
    /// Default is 0.5
    #[inline(always)]
    pub fn min_detection_confidence(mut self, min_detection_confidence: f32) -> Self {
        self.min_detection_confidence = min_detection_confidence;
        self
    }

    /// Set the minimum non-maximum-suppression threshold for face detection to be considered overlapped.
    /// Default is 0.3
    #[inline(always)]
    pub fn min_suppression_threshold(mut self, min_suppression_threshold: f32) -> Self {
        self.min_suppression_threshold = min_suppression_threshold;
        self
    }

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(self, buffer: impl AsRef<[u8]>) -> Result<FaceDetector, crate::Error> {
        if self.num_faces == 0 {
            return Err(crate::Error::ArgumentError(
                "The number of max faces cannot be zero".into(),
            ));
        }

        let buf = buffer.as_ref();
        // parse model and get model resources.
        let model_resource = crate::model::parse_model(buf)?;

        // check model
        model_base_check_impl!(model_resource, 1, 2);
        let img_info =
            model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;

        // generate anchors
        // todo: read info from metadata
        let num_box = 896;
        let width = img_info.width();
        let height = img_info.height();
        let anchors = SsdAnchorsBuilder::new(width, height, 0.1484375, 0.75, 4)
            .interpolated_scale_aspect_ratio(1.0)
            .anchor_offset_x(0.5)
            .anchor_offset_y(0.5)
            .strides(vec![8, 16, 16, 16])
            .aspect_ratios(vec![1.0])
            .fixed_anchor_size(true)
            .generate();

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.device,
        )
        .build_from_bytes([buf])?;

        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        return Ok(FaceDetector {
            build_options: self,
            model_resource,
            graph,
            anchors,
            location_buf_index: 0,
            score_buf_index: 1,
            num_box,
            input_tensor_type,
        });
    }
}
