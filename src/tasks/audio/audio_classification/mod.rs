mod builder;
pub use builder::AudioClassifierBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{
    AudioResultsIter, CategoriesFilter, ClassificationResult, TensorsToClassification,
};
use crate::preprocess::audio::{AudioData, AudioDataToTensorIter, AudioToTensorInfo};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on audio.
pub struct AudioClassifier {
    build_options: AudioClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    input_tensor_type: TensorType,
}

impl AudioClassifier {
    base_task_options_get_impl!();

    classification_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    pub fn new_session(&self) -> Result<AudioClassifierSession<'_>, Error> {
        let input_to_tensor_info = self
            .model_resource
            .expect_to_tensor_info(0)?
            .try_to_audio()?;
        let input_tensor_shape = self.model_resource.expect_input_tensor_shape(0)?;
        let output_tensor_shape = self.model_resource.expect_output_tensor_shape(0)?;

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
            self.model_resource.output_type_and_quantization(0)?,
            output_tensor_shape,
        )?;
        Ok(AudioClassifierSession {
            classifier: self,
            execution_ctx,
            tensors_to_classification,
            input_to_tensor_info,
            input_tensor_shape,
            input_buffer: vec![
                0;
                crate::model::tensor_bytes(
                    self.input_tensor_type,
                    input_tensor_shape
                )
            ],
        })
    }

    /// Classify audio stream using a new session, and collect all results to [`Vec`]
    #[inline]
    pub fn classify(
        &self,
        input_stream: impl AudioData,
    ) -> Result<Vec<ClassificationResult>, Error> {
        self.new_session()?.classify(input_stream)?.to_vec()
    }
}

/// Session to run inference.
/// If process multiple audio input, reuse it can get better performance.
pub struct AudioClassifierSession<'model> {
    classifier: &'model AudioClassifier,
    execution_ctx: GraphExecutionContext<'model>,
    tensors_to_classification: TensorsToClassification<'model>,

    // only one input and one output
    input_to_tensor_info: &'model AudioToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_buffer: Vec<u8>,
}

impl<'model> AudioClassifierSession<'model> {
    /// Classify audio stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline]
    pub fn classify<T>(
        &mut self,
        input_stream: T,
    ) -> Result<AudioResultsIter<'_, '_, Self, T>, Error>
    where
        T: AudioData,
    {
        let audio_data = AudioDataToTensorIter::new(self.input_to_tensor_info, input_stream)?;
        Ok(AudioResultsIter::new(self, audio_data))
    }
}

impl<'model> super::TaskSession for AudioClassifierSession<'model> {
    type Result = ClassificationResult;

    fn process_next<Source: AudioData>(
        &mut self,
        input_stream: &mut AudioDataToTensorIter<Source>,
    ) -> Result<Option<Self::Result>, Error> {
        if let Some(timestamp_ms) = input_stream.poll_next_tensors(&mut [&mut self.input_buffer])? {
            self.execution_ctx.set_input(
                0,
                self.classifier.input_tensor_type,
                self.input_tensor_shape,
                self.input_buffer.as_slice(),
            )?;
            self.execution_ctx.compute()?;

            self.tensors_to_classification
                .output_buffer(0)
                .fetch(&self.execution_ctx, 0)?;

            return Ok(Some(
                self.tensors_to_classification.result(Some(timestamp_ms)),
            ));
        }
        Ok(None)
    }
}
