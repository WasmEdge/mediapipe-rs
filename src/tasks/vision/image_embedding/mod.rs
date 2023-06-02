mod builder;
pub use builder::ImageEmbedderBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{EmbeddingResult, TensorsToEmbedding, VideoResultsIter};
use crate::preprocess::vision::{ImageToTensor, ImageToTensorInfo, VideoData};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs embedding on images and video frames.
pub struct ImageEmbedder {
    build_options: ImageEmbedderBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    input_tensor_type: TensorType,
}

impl ImageEmbedder {
    base_task_options_get_impl!();

    embedding_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<ImageEmbedderSession, Error> {
        let input_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);
        let output_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_shape, 0);
        let mut tensor_to_embedding = TensorsToEmbedding::new(
            self.build_options.embedding_options.quantize,
            self.build_options.embedding_options.l2_normalize,
        );
        tensor_to_embedding.add_output_cfg(
            get_type_and_quantization!(self.model_resource, 0),
            output_tensor_shape,
            None,
        );

        let execution_ctx = self.graph.init_execution_context()?;
        Ok(ImageEmbedderSession {
            execution_ctx,
            tensor_to_embedding,
            input_to_tensor_info,
            input_tensor_shape,
            input_tensor_buf: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
            input_tensor_type: self.input_tensor_type,
        })
    }

    /// Embed one image using a new session.
    #[inline(always)]
    pub fn embed(&self, input: &impl ImageToTensor) -> Result<EmbeddingResult, Error> {
        self.new_session()?.embed(input)
    }

    /// Embed one image using a new session with options to specify the region of interest.
    #[inline(always)]
    pub fn embed_with_options(
        &self,
        input: &impl ImageToTensor,
        process_options: &super::ImageProcessingOptions,
    ) -> Result<EmbeddingResult, Error> {
        self.new_session()?
            .embed_with_options(input, process_options)
    }

    /// Embed audio stream using a new task session, and collect all results to [`Vec`].
    #[inline(always)]
    pub fn embed_for_video(
        &self,
        video_data: impl VideoData,
    ) -> Result<Vec<EmbeddingResult>, Error> {
        self.new_session()?.embed_for_video(video_data)?.to_vec()
    }
}

/// Session to run inference.
/// If process multiple images or videos, reuse it can get better performance.
pub struct ImageEmbedderSession<'model> {
    execution_ctx: GraphExecutionContext<'model>,
    tensor_to_embedding: TensorsToEmbedding,

    input_to_tensor_info: &'model ImageToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_tensor_buf: Vec<u8>,
    input_tensor_type: TensorType,
}

impl<'model> ImageEmbedderSession<'model> {
    #[inline(always)]
    fn compute(&mut self, timestamp_ms: Option<u64>) -> Result<EmbeddingResult, Error> {
        self.execution_ctx.set_input(
            0,
            self.input_tensor_type,
            self.input_tensor_shape,
            self.input_tensor_buf.as_slice(),
        )?;
        self.execution_ctx.compute()?;

        let output_buffer = self.tensor_to_embedding.output_buffer(0);
        self.execution_ctx.get_output(0, output_buffer)?;

        Ok(self.tensor_to_embedding.result(timestamp_ms))
    }

    /// Embed one image, reuse this session data to speedup.
    #[inline(always)]
    pub fn embed(&mut self, input: &impl ImageToTensor) -> Result<EmbeddingResult, Error> {
        input.to_tensor(
            self.input_to_tensor_info,
            &Default::default(),
            &mut self.input_tensor_buf,
        )?;
        self.compute(input.timestamp_ms())
    }

    /// Embed one image, reuse this session data to speedup.
    #[inline(always)]
    pub fn embed_with_options(
        &mut self,
        input: &impl ImageToTensor,
        process_options: &super::ImageProcessingOptions,
    ) -> Result<EmbeddingResult, Error> {
        input.to_tensor(
            self.input_to_tensor_info,
            process_options,
            &mut self.input_tensor_buf,
        )?;
        self.compute(input.timestamp_ms())
    }

    /// Embed input video stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn embed_for_video<InputVideoData: VideoData>(
        &mut self,
        video_data: InputVideoData,
    ) -> Result<VideoResultsIter<Self, InputVideoData>, Error> {
        Ok(VideoResultsIter::new(self, video_data))
    }
}

impl<'model> super::TaskSession for ImageEmbedderSession<'model> {
    type Result = EmbeddingResult;

    #[inline]
    fn process_next(
        &mut self,
        process_options: &super::ImageProcessingOptions,
        video_data: &mut impl VideoData,
    ) -> Result<Option<Self::Result>, Error> {
        if let Some(frame) = video_data.next_frame()? {
            return self
                .embed_with_options(&frame, process_options)
                .map(|r| Some(r));
        }
        Ok(None)
    }
}
