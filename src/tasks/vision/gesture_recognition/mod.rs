mod builder;
mod landmarks_to_tensor;
mod result;

pub use builder::GestureRecognizerBuilder;
use landmarks_to_tensor::*;
pub use result::{GestureRecognizerResult, GestureRecognizerResults};

use super::{HandLandmarker, HandLandmarkerBuilder, HandLandmarkerSession};
use crate::model::ModelResourceTrait;
use crate::postprocess::{
    fetch_output, CategoriesFilter, Category, Landmarks, TensorsToClassification, VideoResultsIter,
};
use crate::preprocess::vision::{ImageToTensor, VideoData};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs gesture recognition on images and video frames.
pub struct GestureRecognizer {
    build_options: GestureRecognizerBuilder,

    gesture_embed_model_resources: Box<dyn ModelResourceTrait>,
    gesture_embed_graph: Graph,

    canned_classifier: GestureClassifier,
    custom_classifier: Option<GestureClassifier>,

    hand_landmarker: HandLandmarker,

    gesture_embed_out_size: usize,
}

/// A gesture classifier model that consumes the gesture embedding.
pub(super) struct GestureClassifier {
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
}

impl GestureClassifier {
    pub(super) fn build(
        file: &[u8],
        device: crate::Device,
        embed_out_size: usize,
    ) -> Result<Self, Error> {
        let model_resource = crate::model::parse_model(file)?;
        model_resource.check_tensor_counts(Some(1), 1)?;
        model_resource.check_input_tensor_type(0, TensorType::F32)?;
        let input_size = model_resource
            .expect_input_tensor_shape(0)?
            .iter()
            .product::<usize>();
        if input_size != embed_out_size {
            return Err(Error::ModelInconsistentError(format!(
                "Expect gesture classifier input elements is `{}`, but got `{}`",
                embed_out_size, input_size
            )));
        }
        let graph = crate::tasks::common::build_graph(model_resource.as_ref(), device, file)?;
        Ok(Self {
            model_resource,
            graph,
        })
    }

    fn add_to<'a>(
        &'a self,
        tensors_to_classification: &mut TensorsToClassification<'a>,
        options: &crate::tasks::common::ClassificationOptions,
    ) -> Result<(), Error> {
        let output_tensor_shape = self.model_resource.expect_output_tensor_shape(0)?;
        let labels = self
            .model_resource
            .output_tensor_labels_locale(0, options.display_names_locale.as_ref())?;
        let categories_filter = CategoriesFilter::new(options, labels.0, labels.1);
        tensors_to_classification.add_classification_options(
            categories_filter,
            options.max_results,
            self.model_resource.output_type_and_quantization(0)?,
            output_tensor_shape,
        )
    }

    fn new_session(&self) -> Result<GestureClassifierSession<'_>, Error> {
        Ok(GestureClassifierSession {
            execution_ctx: self.graph.init_execution_context()?,
            input_shape: self.model_resource.expect_input_tensor_shape(0)?,
        })
    }
}

struct GestureClassifierSession<'model> {
    execution_ctx: GraphExecutionContext<'model>,
    input_shape: &'model [usize],
}

impl GestureClassifierSession<'_> {
    fn classify(
        &mut self,
        embedding: &[f32],
        output: &mut crate::postprocess::OutputBuffer,
    ) -> Result<(), Error> {
        self.execution_ctx
            .set_input(0, TensorType::F32, self.input_shape, embedding)?;
        self.execution_ctx.compute()?;
        output.fetch(&self.execution_ctx, 0)
    }
}

impl GestureRecognizer {
    const HAND_LANDMARKS_INPUT_INDEX: usize = 0;
    const HANDEDNESS_INPUT_INDEX: usize = 1;
    const HAND_WORLD_LANDMARKS_INPUT_INDEX: usize = 2;

    base_task_options_get_impl!();

    classification_options_get_impl!();

    hand_landmark_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    pub fn new_session(&self) -> Result<GestureRecognizerSession<'_>, Error> {
        let embed = &self.gesture_embed_model_resources;
        let gesture_embed_hand_landmarks_input_shape =
            embed.expect_input_tensor_shape(Self::HAND_LANDMARKS_INPUT_INDEX)?;
        let gesture_embed_handedness_input_shape =
            embed.expect_input_tensor_shape(Self::HANDEDNESS_INPUT_INDEX)?;
        let gesture_embed_hand_world_landmarks_input_shape =
            embed.expect_input_tensor_shape(Self::HAND_WORLD_LANDMARKS_INPUT_INDEX)?;

        let mut tensors_to_classification = TensorsToClassification::new();
        self.canned_classifier.add_to(
            &mut tensors_to_classification,
            &self.build_options.classification_options,
        )?;
        let custom_classifier_session = match &self.custom_classifier {
            Some(custom) => {
                custom.add_to(
                    &mut tensors_to_classification,
                    &self.build_options.custom_classification_options,
                )?;
                Some(custom.new_session()?)
            }
            None => None,
        };

        Ok(GestureRecognizerSession {
            gesture_embed_execution_ctx: self.gesture_embed_graph.init_execution_context()?,
            canned_classifier_session: self.canned_classifier.new_session()?,
            custom_classifier_session,
            hand_landmarker_session: self.hand_landmarker.new_session()?,
            gesture_embed_hand_landmarks_input_shape,
            gesture_embed_handedness_input_shape,
            gesture_embed_hand_world_landmarks_input_shape,
            gesture_embed_hand_landmarks_input_buffer: vec![
                0.;
                gesture_embed_hand_landmarks_input_shape
                    .iter()
                    .product()
            ],
            gesture_embed_hand_world_landmarks_input_buffer: vec![
                0.;
                gesture_embed_hand_world_landmarks_input_shape
                    .iter()
                    .product()
            ],
            gesture_embed_handedness_input_buffer: [0.],
            gesture_embed_out_buffer: vec![0.; self.gesture_embed_out_size],
            tensors_to_classification,
        })
    }

    /// Recognize one image using a new task session.
    #[inline]
    pub fn recognize(&self, input: &impl ImageToTensor) -> Result<GestureRecognizerResults, Error> {
        self.new_session()?.recognize(input)
    }

    /// Recognize video stream using a new task session, and collect all results to [`Vec`].
    #[inline]
    pub fn recognize_for_video(
        &self,
        video_data: impl VideoData,
    ) -> Result<Vec<GestureRecognizerResults>, Error> {
        self.new_session()?
            .recognize_for_video(video_data)?
            .to_vec()
    }
}

/// Session to run inference.
/// If process multiple images or videos, reuse it can get better performance.
pub struct GestureRecognizerSession<'model> {
    gesture_embed_execution_ctx: GraphExecutionContext<'model>,
    canned_classifier_session: GestureClassifierSession<'model>,
    custom_classifier_session: Option<GestureClassifierSession<'model>>,

    hand_landmarker_session: HandLandmarkerSession<'model>,

    gesture_embed_hand_landmarks_input_shape: &'model [usize],
    gesture_embed_handedness_input_shape: &'model [usize],
    gesture_embed_hand_world_landmarks_input_shape: &'model [usize],
    gesture_embed_hand_landmarks_input_buffer: Vec<f32>,
    gesture_embed_hand_world_landmarks_input_buffer: Vec<f32>,
    gesture_embed_handedness_input_buffer: [f32; 1],
    gesture_embed_out_buffer: Vec<f32>,

    tensors_to_classification: TensorsToClassification<'model>,
}

impl<'model> GestureRecognizerSession<'model> {
    /// Recognize one image using this session.
    pub fn recognize(
        &mut self,
        input: &impl ImageToTensor,
    ) -> Result<GestureRecognizerResults, Error> {
        let img_size = input.image_size();
        let timestamp_ms = input.timestamp_ms();

        let hand_landmark_results = self.hand_landmarker_session.detect(input)?;
        let mut gesture_recognizer_results = Vec::with_capacity(hand_landmark_results.len());

        for hand_landmark in hand_landmark_results {
            self.gesture_embed_handedness_input_buffer[0] =
                handedness_to_tensor(&hand_landmark.handedness);
            landmarks_to_tensor(
                &hand_landmark.hand_landmarks,
                &mut self.gesture_embed_hand_landmarks_input_buffer,
                img_size,
                0,
            );
            world_landmarks_to_tensor(
                &hand_landmark.hand_world_landmarks,
                &mut self.gesture_embed_hand_world_landmarks_input_buffer,
            );
            self.gesture_embed_execution_ctx.set_input(
                GestureRecognizer::HANDEDNESS_INPUT_INDEX,
                TensorType::F32,
                self.gesture_embed_handedness_input_shape,
                self.gesture_embed_handedness_input_buffer,
            )?;
            self.gesture_embed_execution_ctx.set_input(
                GestureRecognizer::HAND_LANDMARKS_INPUT_INDEX,
                TensorType::F32,
                self.gesture_embed_hand_landmarks_input_shape,
                self.gesture_embed_hand_landmarks_input_buffer.as_slice(),
            )?;
            self.gesture_embed_execution_ctx.set_input(
                GestureRecognizer::HAND_WORLD_LANDMARKS_INPUT_INDEX,
                TensorType::F32,
                self.gesture_embed_hand_world_landmarks_input_shape,
                self.gesture_embed_hand_world_landmarks_input_buffer
                    .as_slice(),
            )?;

            self.gesture_embed_execution_ctx.compute()?;
            fetch_output(
                &self.gesture_embed_execution_ctx,
                0,
                &mut self.gesture_embed_out_buffer,
            )?;

            self.canned_classifier_session.classify(
                &self.gesture_embed_out_buffer,
                self.tensors_to_classification.output_buffer(0),
            )?;
            if let Some(custom) = &mut self.custom_classifier_session {
                custom.classify(
                    &self.gesture_embed_out_buffer,
                    self.tensors_to_classification.output_buffer(1),
                )?;
            }

            let result = GestureRecognizerResult {
                gestures: self.tensors_to_classification.result(timestamp_ms),
                hand_landmark,
            };
            gesture_recognizer_results.push(result)
        }
        Ok(gesture_recognizer_results.into())
    }

    /// Recognize input video stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline]
    pub fn recognize_for_video<InputVideoData: VideoData>(
        &mut self,
        video_data: InputVideoData,
    ) -> Result<VideoResultsIter<'_, Self, InputVideoData>, Error> {
        Ok(VideoResultsIter::new(self, video_data))
    }
}

impl<'model> super::TaskSession for GestureRecognizerSession<'model> {
    type Result = GestureRecognizerResults;

    fn process_next(
        &mut self,
        _process_options: &super::ImageProcessingOptions,
        video_data: &mut impl VideoData,
    ) -> Result<Option<Self::Result>, Error> {
        // todo: video track optimize
        if let Some(frame) = video_data.next_frame()? {
            return self.recognize(&frame).map(Some);
        }
        Ok(None)
    }
}

fn handedness_to_tensor(category: &Category) -> f32 {
    if category.index == 0 {
        category.score
    } else {
        1.0 - category.score
    }
}
