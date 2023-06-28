mod builder;
pub use builder::ImageClassifierBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{
    CategoriesFilter, ClassificationResult, TensorsToClassification, VideoResultsIter,
};
use crate::preprocess::vision::{ImageToTensor, ImageToTensorInfo, VideoData};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on images and video frames.
pub struct ImageClassifier {
    build_options: ImageClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    input_tensor_type: TensorType,
}

impl ImageClassifier {
    base_task_options_get_impl!();

    classification_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<ImageClassifierSession, Error> {
        let input_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);
        let output_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_shape, 0);

        let labels = self.model_resource.output_tensor_labels_locale(
            0,
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
        let mut tensors_to_classification = TensorsToClassification::new();
        tensors_to_classification.add_classification_options(
            categories_filter,
            self.build_options.classification_options.max_results,
            get_type_and_quantization!(self.model_resource, 0),
            output_tensor_shape,
        );

        let execution_ctx = self.graph.init_execution_context()?;
        Ok(ImageClassifierSession {
            execution_ctx,
            tensors_to_classification,
            input_to_tensor_info,
            input_tensor_shape,
            input_tensor_buf: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
            input_tensor_type: self.input_tensor_type,
        })
    }

    /// Classify one image using a new session.
    #[inline(always)]
    pub fn classify(&self, input: &impl ImageToTensor) -> Result<ClassificationResult, Error> {
        self.new_session()?.classify(input)
    }

    /// Classify one image using a new session with options to specify the region of interest.
    #[inline(always)]
    pub fn classify_with_options(
        &self,
        input: &impl ImageToTensor,
        process_options: &super::ImageProcessingOptions,
    ) -> Result<ClassificationResult, Error> {
        self.new_session()?
            .classify_with_options(input, process_options)
    }

    /// Classify video stream using a new task session, and collect all results to [`Vec`].
    #[inline(always)]
    pub fn classify_for_video(
        &self,
        video_data: impl VideoData,
    ) -> Result<Vec<ClassificationResult>, Error> {
        self.new_session()?.classify_for_video(video_data)?.to_vec()
    }
}

/// Session to run inference.
/// If process multiple images or videos, reuse it can get better performance.
///
/// ```rust
/// use mediapipe_rs::tasks::vision::ImageClassifier;
///
/// let image_classifier: ImageClassifier;
/// let mut session = image_classifier.new_session()?;
/// for image in images {
///     session.classify(image)?;
/// }
/// ```
pub struct ImageClassifierSession<'model> {
    execution_ctx: GraphExecutionContext<'model>,
    tensors_to_classification: TensorsToClassification<'model>,

    // only one input and one output
    input_to_tensor_info: &'model ImageToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_tensor_buf: Vec<u8>,
    input_tensor_type: TensorType,
}

impl<'model> ImageClassifierSession<'model> {
    #[inline(always)]
    fn compute(&mut self, timestamp_ms: Option<u64>) -> Result<ClassificationResult, Error> {
        self.execution_ctx.set_input(
            0,
            self.input_tensor_type,
            self.input_tensor_shape,
            self.input_tensor_buf.as_slice(),
        )?;

        self.execution_ctx.compute()?;

        let output_buffer = self.tensors_to_classification.output_buffer(0);
        let output_size = self.execution_ctx.get_output(0, output_buffer)?;
        if output_size != output_buffer.len() {
            return Err(Error::ModelInconsistentError(format!(
                "Model output bytes size is `{}`, but got `{}`",
                output_buffer.len(),
                output_size
            )));
        }

        Ok(self.tensors_to_classification.result(timestamp_ms))
    }

    /// Classify one image, reuse this session data to speedup.
    #[inline(always)]
    pub fn classify(&mut self, input: &impl ImageToTensor) -> Result<ClassificationResult, Error> {
        input.to_tensor(
            self.input_to_tensor_info,
            &Default::default(),
            &mut self.input_tensor_buf,
        )?;
        self.compute(input.timestamp_ms())
    }

    /// Classify one image with region-of-interest options, reuse this session data to speedup.
    #[inline(always)]
    pub fn classify_with_options(
        &mut self,
        input: &impl ImageToTensor,
        process_options: &super::ImageProcessingOptions,
    ) -> Result<ClassificationResult, Error> {
        input.to_tensor(
            self.input_to_tensor_info,
            process_options,
            &mut self.input_tensor_buf,
        )?;
        self.compute(input.timestamp_ms())
    }

    /// Classify input video stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn classify_for_video<InputVideoData: VideoData>(
        &mut self,
        video_data: InputVideoData,
    ) -> Result<VideoResultsIter<Self, InputVideoData>, Error> {
        Ok(VideoResultsIter::new(self, video_data))
    }
}

impl<'model> super::TaskSession for ImageClassifierSession<'model> {
    type Result = ClassificationResult;

    #[inline]
    fn process_next(
        &mut self,
        process_options: &super::ImageProcessingOptions,
        video_data: &mut impl VideoData,
    ) -> Result<Option<Self::Result>, Error> {
        if let Some(frame) = video_data.next_frame()? {
            frame.to_tensor(
                self.input_to_tensor_info,
                process_options,
                &mut self.input_tensor_buf,
            )?;
            return Ok(Some(self.compute(frame.timestamp_ms())?));
        }
        Ok(None)
    }
}
