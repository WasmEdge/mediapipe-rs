mod builder;
mod landmarks_to_tensor;
mod result;

pub use builder::GestureRecognizerBuilder;
use landmarks_to_tensor::*;
pub use result::{GestureRecognizerResult, GestureRecognizerResults};

use super::{HandLandmarker, HandLandmarkerBuilder, HandLandmarkerSession};
use crate::model::ModelResourceTrait;
use crate::postprocess::{
    CategoriesFilter, Category, Landmarks, TensorsToClassification, VideoResultsIter,
};
use crate::preprocess::vision::{ImageToTensor, VideoData};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs gesture recognition on images and video frames.
pub struct GestureRecognizer {
    build_options: GestureRecognizerBuilder,

    gesture_embed_model_resources: Box<dyn ModelResourceTrait>,
    gesture_embed_graph: Graph,

    canned_classify_model_resources: Box<dyn ModelResourceTrait>,
    canned_classify_graph: Graph,

    custom_classify_resources: Option<Box<dyn ModelResourceTrait>>,
    custom_classify_graph: Option<Graph>,

    hand_landmarker: HandLandmarker,

    gesture_embed_hand_landmarks_input_index: usize,
    gesture_embed_handedness_input_index: usize,
    gesture_embed_hand_world_landmarks_input_index: usize,
    gesture_embed_out_size: usize,
}

macro_rules! add_tensors_to_classifications {
    ( $tensors_to_classification:ident, $self:ident, $resource:expr, $classify_option_field:ident, ) => {{
        let output_tensor_shape =
            model_resource_check_and_get_impl!($resource, output_tensor_shape, 0);
        let labels = $resource.output_tensor_labels_locale(
            0,
            $self
                .build_options
                .$classify_option_field
                .display_names_locale
                .as_ref(),
        )?;
        let categories_filter = CategoriesFilter::new(
            &$self.build_options.$classify_option_field,
            labels.0,
            labels.1,
        );
        $tensors_to_classification.add_classification_options(
            categories_filter,
            $self.build_options.$classify_option_field.max_results,
            get_type_and_quantization!($resource, 0),
            output_tensor_shape,
        );
    };};
}

impl GestureRecognizer {
    base_task_options_get_impl!();

    classification_options_get_impl!();

    hand_landmark_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<GestureRecognizerSession, Error> {
        // get input shapes
        let gesture_embed_hand_landmarks_input_shape = model_resource_check_and_get_impl!(
            self.gesture_embed_model_resources,
            input_tensor_shape,
            self.gesture_embed_hand_landmarks_input_index
        );
        let gesture_embed_handedness_input_shape = model_resource_check_and_get_impl!(
            self.gesture_embed_model_resources,
            input_tensor_shape,
            self.gesture_embed_handedness_input_index
        );
        let gesture_embed_hand_world_landmarks_input_shape = model_resource_check_and_get_impl!(
            self.gesture_embed_model_resources,
            input_tensor_shape,
            self.gesture_embed_hand_world_landmarks_input_index
        );
        let canned_classify_input_shape = model_resource_check_and_get_impl!(
            self.canned_classify_model_resources,
            input_tensor_shape,
            0
        );
        let custom_classify_input_shape = if let Some(ref r) = self.custom_classify_resources {
            Some(model_resource_check_and_get_impl!(r, input_tensor_shape, 0))
        } else {
            None
        };
        let gesture_embed_hand_landmarks_input_size = gesture_embed_hand_landmarks_input_shape
            .iter()
            .fold(1, |a, b| a * b);
        let gesture_embed_hand_world_landmarks_input_size =
            gesture_embed_hand_world_landmarks_input_shape
                .iter()
                .fold(1, |a, b| a * b);

        // tensors to classifications
        let mut tensors_to_classification = TensorsToClassification::new();
        add_tensors_to_classifications!(
            tensors_to_classification,
            self,
            self.canned_classify_model_resources,
            classification_options,
        );
        if let Some(ref r) = self.custom_classify_resources {
            add_tensors_to_classifications!(
                tensors_to_classification,
                self,
                r,
                custom_classification_options,
            );
        }

        // init contexts
        let gesture_embed_execution_ctx = self.gesture_embed_graph.init_execution_context()?;
        let canned_classify_execution_ctx = self.canned_classify_graph.init_execution_context()?;
        let custom_classify_execution_ctx = if let Some(ref g) = self.custom_classify_graph {
            Some(g.init_execution_context()?)
        } else {
            None
        };
        let hand_landmarker_session = self.hand_landmarker.new_session()?;
        Ok(GestureRecognizerSession {
            gesture_recognizer: self,
            gesture_embed_execution_ctx,
            canned_classify_execution_ctx,
            custom_classify_execution_ctx,
            hand_landmarker_session,
            gesture_embed_hand_landmarks_input_shape,
            gesture_embed_handedness_input_shape,
            gesture_embed_hand_world_landmarks_input_shape,
            canned_classify_input_shape,
            custom_classify_input_shape,
            gesture_embed_hand_landmarks_input_buffer: vec![
                0.;
                gesture_embed_hand_landmarks_input_size
            ],
            gesture_embed_hand_world_landmarks_input_buffer:
                vec![0.; gesture_embed_hand_world_landmarks_input_size],
            gesture_embed_handedness_input_buffer: [0.],
            gesture_embed_out_buffer: vec![0.; self.gesture_embed_out_size],
            tensors_to_classification,
        })
    }

    /// Recognize one image using a new task session.
    #[inline(always)]
    pub fn recognize(&self, input: &impl ImageToTensor) -> Result<GestureRecognizerResults, Error> {
        self.new_session()?.recognize(input)
    }

    /// Recognize video stream using a new task session, and collect all results to [`Vec`].
    #[inline(always)]
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
    gesture_recognizer: &'model GestureRecognizer,
    gesture_embed_execution_ctx: GraphExecutionContext<'model>,
    canned_classify_execution_ctx: GraphExecutionContext<'model>,
    custom_classify_execution_ctx: Option<GraphExecutionContext<'model>>,

    hand_landmarker_session: HandLandmarkerSession<'model>,

    gesture_embed_hand_landmarks_input_shape: &'model [usize],
    gesture_embed_handedness_input_shape: &'model [usize],
    gesture_embed_hand_world_landmarks_input_shape: &'model [usize],
    canned_classify_input_shape: &'model [usize],
    custom_classify_input_shape: Option<&'model [usize]>,
    gesture_embed_hand_landmarks_input_buffer: Vec<f32>,
    gesture_embed_hand_world_landmarks_input_buffer: Vec<f32>,
    gesture_embed_handedness_input_buffer: [f32; 1],
    gesture_embed_out_buffer: Vec<f32>,

    tensors_to_classification: TensorsToClassification<'model>,
}

impl<'model> GestureRecognizerSession<'model> {
    /// Recognize one image using this session.
    #[inline(always)]
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
                self.gesture_recognizer.gesture_embed_handedness_input_index,
                TensorType::F32,
                self.gesture_embed_handedness_input_shape,
                &self.gesture_embed_handedness_input_buffer,
            )?;
            self.gesture_embed_execution_ctx.set_input(
                self.gesture_recognizer
                    .gesture_embed_hand_landmarks_input_index,
                TensorType::F32,
                self.gesture_embed_hand_landmarks_input_shape,
                self.gesture_embed_hand_landmarks_input_buffer.as_slice(),
            )?;
            self.gesture_embed_execution_ctx.set_input(
                self.gesture_recognizer
                    .gesture_embed_hand_world_landmarks_input_index,
                TensorType::F32,
                self.gesture_embed_hand_world_landmarks_input_shape,
                self.gesture_embed_hand_world_landmarks_input_buffer
                    .as_slice(),
            )?;

            self.gesture_embed_execution_ctx.compute()?;
            self.gesture_embed_execution_ctx
                .get_output(0, &mut self.gesture_embed_out_buffer)?;

            self.canned_classify_execution_ctx.set_input(
                0,
                TensorType::F32,
                self.canned_classify_input_shape,
                self.gesture_embed_out_buffer.as_slice(),
            )?;
            self.canned_classify_execution_ctx.compute()?;
            let output_buffer = self.tensors_to_classification.output_buffer(0);
            self.canned_classify_execution_ctx
                .get_output(0, output_buffer)?;

            if let Some(ref mut ctx) = self.custom_classify_execution_ctx {
                ctx.set_input(
                    0,
                    TensorType::F32,
                    self.custom_classify_input_shape.unwrap(),
                    self.gesture_embed_out_buffer.as_slice(),
                )?;
                ctx.compute()?;
                let output_buffer = self.tensors_to_classification.output_buffer(1);
                ctx.get_output(0, output_buffer)?;
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
    #[inline(always)]
    pub fn recognize_for_video<InputVideoData: VideoData>(
        &mut self,
        video_data: InputVideoData,
    ) -> Result<VideoResultsIter<Self, InputVideoData>, Error> {
        Ok(VideoResultsIter::new(self, video_data))
    }
}

impl<'model> super::TaskSession for GestureRecognizerSession<'model> {
    type Result = GestureRecognizerResults;

    #[inline]
    fn process_next(
        &mut self,
        _process_options: &super::ImageProcessingOptions,
        video_data: &mut impl VideoData,
    ) -> Result<Option<Self::Result>, Error> {
        // todo: video track optimize
        if let Some(frame) = video_data.next_frame()? {
            return self.recognize(&frame).map(|r| Some(r));
        }
        Ok(None)
    }
}

#[inline(always)]
fn handedness_to_tensor(category: &Category) -> f32 {
    if category.index == 0 {
        category.score
    } else {
        1.0 - category.score
    }
}
