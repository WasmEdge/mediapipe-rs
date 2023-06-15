mod builder;
pub use builder::TextClassifierBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{CategoriesFilter, ClassificationResult, TensorsToClassification};
use crate::preprocess::text::{TextToTensorInfo, TextToTensors};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on text.
pub struct TextClassifier {
    build_options: TextClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
}

impl TextClassifier {
    base_task_options_get_impl!();

    classification_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<TextClassifierSession, Error> {
        let input_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_text()?;
        let input_count = self.model_resource.input_tensor_count();
        let mut input_tensor_shapes = Vec::with_capacity(input_count);
        let mut input_tensor_bufs = Vec::with_capacity(input_count);
        for i in 0..input_count {
            let input_tensor_shape =
                model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, i);
            let bytes = input_tensor_shape.iter().fold(4, |sum, b| sum * *b);
            input_tensor_shapes.push(input_tensor_shape);
            input_tensor_bufs.push(vec![0; bytes]);
        }

        let output_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_shape, 0);

        let execution_ctx = self.graph.init_execution_context()?;
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

        Ok(TextClassifierSession {
            execution_ctx,
            tensors_to_classification,
            input_to_tensor_info,
            input_tensor_shapes,
            input_tensor_bufs,
        })
    }

    /// Classify the input using a new session.
    #[inline(always)]
    pub fn classify(&self, input: &impl TextToTensors) -> Result<ClassificationResult, Error> {
        self.new_session()?.classify(input)
    }
}

/// Session to run inference.
/// If process multiple text, reuse it can get better performance.
///
/// ```rust
/// use mediapipe_rs::tasks::text::TextClassifier;
///
/// let text_classifier: TextClassifier;
/// let mut session = text_classifier.new_session()?;
/// for text in texts {
///     session.classify(text)?;
/// }
/// ```
pub struct TextClassifierSession<'a> {
    execution_ctx: GraphExecutionContext<'a>,
    tensors_to_classification: TensorsToClassification<'a>,

    input_to_tensor_info: &'a TextToTensorInfo,
    input_tensor_shapes: Vec<&'a [usize]>,
    input_tensor_bufs: Vec<Vec<u8>>,
}

impl<'a> TextClassifierSession<'a> {
    /// Classify the input using this session.
    pub fn classify(&mut self, input: &impl TextToTensors) -> Result<ClassificationResult, Error> {
        input.to_tensors(self.input_to_tensor_info, &mut self.input_tensor_bufs)?;

        for index in 0..self.input_tensor_bufs.len() {
            self.execution_ctx.set_input(
                index,
                TensorType::I32,
                self.input_tensor_shapes[index],
                self.input_tensor_bufs[index].as_slice(),
            )?;
        }
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

        Ok(self.tensors_to_classification.result(None))
    }
}
