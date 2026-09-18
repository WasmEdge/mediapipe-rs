use super::HandDetector;
use crate::postprocess::SsdAnchorsBuilder;
use crate::tasks::common::BaseTaskOptions;

/// Configure the build options of a new **Hand Detection** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct HandDetectorBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    /// The maximum number of hands output by the detector.
    pub(super) num_hands: Option<usize>,
    /// Minimum confidence value ([0.0, 1.0]) for confidence score to be considered
    /// successfully detecting a hand in the image.
    pub(super) min_detection_confidence: f32,
}

impl Default for HandDetectorBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl HandDetectorBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            num_hands: None,
            min_detection_confidence: 0.5,
        }
    }

    base_task_options_impl!(HandDetector);

    /// Set the maximum number of hands can be detected by the HandDetector.
    /// By default there is no limit.
    #[inline(always)]
    pub fn num_hands(mut self, num_hands: usize) -> Self {
        self.num_hands = Some(num_hands);
        self
    }

    /// Set the minimum confidence score for the hand detection to be considered successful.
    /// Default is 0.5
    #[inline(always)]
    pub fn min_detection_confidence(mut self, min_detection_confidence: f32) -> Self {
        self.min_detection_confidence = min_detection_confidence;
        self
    }

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(self, buffer: impl AsRef<[u8]>) -> Result<HandDetector, crate::Error> {
        if self.num_hands == Some(0) {
            return Err(crate::Error::ArgumentError(
                "The number of max hands cannot be zero".into(),
            ));
        }

        let buf = buffer.as_ref();
        // parse model and get model resources.
        let model_resource = crate::model::parse_model(buf)?;

        // check model
        model_resource.check_tensor_counts(Some(1), 2)?;
        let img_info = model_resource.expect_to_tensor_info(0)?.try_to_image()?;

        // generate anchors
        // todo: read info from metadata
        let num_box = 2016;
        let width = img_info.width();
        let height = img_info.height();
        let anchors = SsdAnchorsBuilder::new(width, height, 0.1484375, 0.75, 4)
            .anchor_offset_x(0.5)
            .anchor_offset_y(0.5)
            .strides(vec![8, 16, 16, 16])
            .aspect_ratios(vec![1.0])
            .fixed_anchor_size(true)
            .generate();
        if anchors.len() != num_box {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect `{}` anchors for the hand detector, but the `{}x{}` model input generates `{}`",
                num_box,
                width,
                height,
                anchors.len()
            )));
        }

        let graph = crate::tasks::common::build_graph(
            model_resource.as_ref(),
            self.base_task_options.device,
            buf,
        )?;

        let input_tensor_type = model_resource.expect_input_tensor_type(0)?;

        Ok(HandDetector {
            build_options: self,
            model_resource,
            graph,
            anchors,
            location_buf_index: 0,
            score_buf_index: 1,
            num_box,
            input_tensor_type,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::tasks::vision::HandDetectorBuilder;
    use crate::Error;

    const FACE_DETECTOR: &str = "assets/models/face_detection/face_detection_short_range.tflite";

    #[test]
    fn test_anchor_count_mismatch() {
        // the face detector takes a 128x128 input, which generates 896 anchors instead of 2016
        let face_detector = std::fs::read(FACE_DETECTOR).unwrap();
        match HandDetectorBuilder::new().build_from_buffer(face_detector) {
            Err(Error::ModelInconsistentError(msg)) => assert!(msg.contains("anchors"), "{}", msg),
            Err(e) => panic!("expected an anchor count error, got {:?}", e),
            Ok(_) => panic!("expected an anchor count error"),
        }
    }
}
