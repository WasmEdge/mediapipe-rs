mod builder;
pub use builder::TextEmbedderBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{EmbeddingResult, TensorsToEmbedding};
use crate::preprocess::text::{TextToTensorInfo, TextToTensors};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs embedding on texts.
pub struct TextEmbedder {
    build_options: TextEmbedderBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    input_count: usize,
}

impl TextEmbedder {
    base_task_options_get_impl!();

    embedding_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<TextEmbedderSession, Error> {
        let input_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_text()?;
        let mut input_tensor_shapes = Vec::with_capacity(self.input_count);
        let mut input_tensor_bufs = Vec::with_capacity(self.input_count);
        for i in 0..self.input_count {
            let input_tensor_shape =
                model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, i);
            let bytes = input_tensor_shape.iter().fold(4, |sum, b| sum * *b);
            input_tensor_shapes.push(input_tensor_shape);
            input_tensor_bufs.push(vec![0; bytes]);
        }
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
        Ok(TextEmbedderSession {
            execution_ctx,
            tensor_to_embedding,
            input_to_tensor_info,
            input_tensor_shapes,
            input_tensor_bufs,
        })
    }

    /// Embed one text using a new session.
    #[inline(always)]
    pub fn embed(&self, input: &impl TextToTensors) -> Result<EmbeddingResult, Error> {
        self.new_session()?.embed(input)
    }
}

/// Session to run inference.
/// If process multiple texts, reuse it can get better performance.
pub struct TextEmbedderSession<'a> {
    execution_ctx: GraphExecutionContext<'a>,
    tensor_to_embedding: TensorsToEmbedding,

    input_to_tensor_info: &'a TextToTensorInfo,
    input_tensor_shapes: Vec<&'a [usize]>,
    input_tensor_bufs: Vec<Vec<u8>>,
}

impl<'a> TextEmbedderSession<'a> {
    /// Embed one text use this session.
    #[inline(always)]
    pub fn embed(&mut self, input: &impl TextToTensors) -> Result<EmbeddingResult, Error> {
        input.to_tensors(self.input_to_tensor_info, &mut self.input_tensor_bufs)?;

        let tensor_type = TensorType::I32;
        for index in 0..self.input_tensor_bufs.len() {
            self.execution_ctx.set_input(
                index,
                tensor_type,
                self.input_tensor_shapes[index],
                self.input_tensor_bufs[index].as_slice(),
            )?;
        }
        self.execution_ctx.compute()?;

        let output_buffer = self.tensor_to_embedding.output_buffer(0);
        self.execution_ctx.get_output(0, output_buffer)?;

        Ok(self.tensor_to_embedding.result(None))
    }
}
