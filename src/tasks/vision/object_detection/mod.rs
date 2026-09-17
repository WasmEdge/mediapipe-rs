mod builder;
pub use builder::ObjectDetectorBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{fetch_output, CategoriesFilter, DetectionResult, TensorsToDetection};
use crate::preprocess::vision::ImageToTensorInfo;
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs object detection on images and video frames.
pub struct ObjectDetector {
    build_options: ObjectDetectorBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    bound_box_properties: [usize; 4],
    location_buf_index: usize,
    categories_buf_index: usize,
    score_buf_index: usize,
    num_box_buf_index: usize,
    // only one input and one output
    input_tensor_type: TensorType,
}

impl ObjectDetector {
    /// Upper bound for the detection count when the model declares dynamic output shapes.
    const MAX_DYNAMIC_DETECTIONS: usize = 1024;

    classification_options_get_impl!();

    detector_impl!(ObjectDetectorSession, DetectionResult);

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<ObjectDetectorSession, Error> {
        let image_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);
        let labels = self.model_resource.output_tensor_labels_locale(
            self.categories_buf_index,
            self.build_options
                .classification_options
                .display_names_locale
                .as_ref(),
        )?;

        let categories_filter = CategoriesFilter::new(
            &self.build_options.classification_options,
            labels.0,
            labels.1,
        );
        let mut tensors_to_detection = TensorsToDetection::new(
            categories_filter,
            self.build_options.classification_options.max_results,
            get_type_and_quantization!(self.model_resource, self.location_buf_index),
            get_type_and_quantization!(self.model_resource, self.categories_buf_index),
            get_type_and_quantization!(self.model_resource, self.score_buf_index),
        )?;
        tensors_to_detection.set_box_indices(&self.bound_box_properties)?;
        let location_shape = model_resource_check_and_get_impl!(
            self.model_resource,
            output_tensor_shape,
            self.location_buf_index
        );
        // TFLite_Detection_PostProcess declares dynamic (empty) output shapes.
        let max_num_box = if location_shape.is_empty() {
            Self::MAX_DYNAMIC_DETECTIONS
        } else {
            tensors_to_detection.max_boxes(location_shape.iter().product())
        };

        let execution_ctx = self.graph.init_execution_context()?;
        Ok(ObjectDetectorSession {
            detector: self,
            execution_ctx,
            tensors_to_detection,
            num_box_buf: [0f32],
            max_num_box,
            image_to_tensor_info,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }
}

/// Session to run inference.
/// If process multiple images or videos, reuse it can get better performance.
///
/// ```rust
/// use mediapipe_rs::tasks::vision::ObjectDetector;
///
/// let object_detector: ObjectDetector;
/// let mut session = object_detector.new_session()?;
/// for image in images {
///     session.detect(image)?;
/// }
/// ```
pub struct ObjectDetectorSession<'model> {
    detector: &'model ObjectDetector,
    execution_ctx: GraphExecutionContext<'model>,
    tensors_to_detection: TensorsToDetection<'model>,

    image_to_tensor_info: &'model ImageToTensorInfo,
    num_box_buf: [f32; 1],
    max_num_box: usize,
    input_tensor_shape: &'model [usize],
    input_buffer: Vec<u8>,
}

impl<'model> ObjectDetectorSession<'model> {
    // todo: usage the timestamp
    #[allow(unused)]
    #[inline(always)]
    fn compute(&mut self, timestamp_ms: Option<u64>) -> Result<DetectionResult, Error> {
        self.execution_ctx.set_input(
            0,
            self.detector.input_tensor_type,
            self.input_tensor_shape,
            self.input_buffer.as_slice(),
        )?;
        self.execution_ctx.compute()?;

        // get num box
        fetch_output(
            &self.execution_ctx,
            self.detector.num_box_buf_index,
            &mut self.num_box_buf,
        )?;
        let num_box = TensorsToDetection::detection_count(self.num_box_buf[0], self.max_num_box)?;

        // realloc
        self.tensors_to_detection.realloc(num_box);

        // get other buffers
        self.tensors_to_detection
            .location_buf()
            .fetch(&self.execution_ctx, self.detector.location_buf_index)?;
        self.tensors_to_detection
            .categories_buf()
            .ok_or_else(|| {
                Error::ModelInconsistentError("Missing categories output buffer".into())
            })?
            .fetch(&self.execution_ctx, self.detector.categories_buf_index)?;
        self.tensors_to_detection
            .score_buf()
            .fetch(&self.execution_ctx, self.detector.score_buf_index)?;

        // generate result
        Ok(self.tensors_to_detection.result(num_box))
    }

    detector_session_impl!(DetectionResult);
}

detection_task_session_impl!(ObjectDetectorSession, DetectionResult);
