mod builder;
mod hand_landmark;
mod result;

use super::{HandDetector, HandDetectorBuilder, HandDetectorSession};
pub use builder::HandLandmarkerBuilder;
pub use hand_landmark::HandLandmark;
pub use result::{HandLandmarkResult, HandLandmarkResults};

use crate::model::ModelResourceTrait;
use crate::postprocess::{CategoriesFilter, NormalizedRect, TensorsToLandmarks, VideoResultsIter};
use crate::preprocess::vision::{ImageToTensor, ImageToTensorInfo, VideoData};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs hand landmark on images and video frames.
pub struct HandLandmarker {
    build_options: HandLandmarkerBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    hand_detector: HandDetector,

    handedness_buf_index: usize,
    score_buf_index: usize,
    landmarks_buf_index: usize,
    world_landmarks_buf_index: usize,

    // only one input and one output
    input_tensor_type: TensorType,
}

impl HandLandmarker {
    const LANDMARKS_NORMALIZE_Z: f32 = 0.4;

    detector_impl!(HandLandmarkerSession, HandLandmarkResults);

    hand_landmark_options_get_impl!();

    /// Get the subtask: hand detector.
    #[inline(always)]
    pub fn subtask_hand_detector(&self) -> &HandDetector {
        &self.hand_detector
    }

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<HandLandmarkerSession, Error> {
        let image_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);

        // todo: parse index from metadata
        let hand_labels = self.model_resource.output_tensor_labels_locale(0, "")?.0;
        let categories_filter =
            CategoriesFilter::new_full(f32::MIN, hand_labels, Some(hand_labels));

        let landmarks_out =
            get_type_and_quantization!(self.model_resource, self.landmarks_buf_index);
        let landmarks_shape = model_resource_check_and_get_impl!(
            self.model_resource,
            output_tensor_shape,
            self.landmarks_buf_index
        );
        let mut tensors_to_landmarks =
            TensorsToLandmarks::new(HandLandmark::NAMES.len(), landmarks_out, landmarks_shape)?;
        tensors_to_landmarks
            .set_image_size(image_to_tensor_info.width(), image_to_tensor_info.height());
        tensors_to_landmarks.set_normalize_z(Self::LANDMARKS_NORMALIZE_Z);

        let world_landmarks_out =
            get_type_and_quantization!(self.model_resource, self.world_landmarks_buf_index);
        let world_landmarks_shape = model_resource_check_and_get_impl!(
            self.model_resource,
            output_tensor_shape,
            self.world_landmarks_buf_index
        );
        let tensors_to_world_landmarks = TensorsToLandmarks::new(
            HandLandmark::NAMES.len(),
            world_landmarks_out,
            world_landmarks_shape,
        )?;

        let hand_detector_session = self.hand_detector.new_session()?;
        let execution_ctx = self.graph.init_execution_context()?;

        Ok(HandLandmarkerSession {
            hand_landmarker: self,
            execution_ctx,
            hand_detector_session,
            image_to_tensor_info,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
            score_of_hand_presence: [0.],
            score_of_handedness: [0.],
            categories_filter,
            tensors_to_landmarks,
            tensors_to_world_landmarks,
        })
    }
}

/// Session to run inference.
/// If process multiple images or videos, reuse it can get better performance.
pub struct HandLandmarkerSession<'model> {
    hand_landmarker: &'model HandLandmarker,
    execution_ctx: GraphExecutionContext<'model>,

    hand_detector_session: HandDetectorSession<'model>,

    image_to_tensor_info: &'model ImageToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_buffer: Vec<u8>,
    score_of_hand_presence: [f32; 1],
    score_of_handedness: [f32; 1],
    categories_filter: CategoriesFilter<'model>,
    tensors_to_landmarks: TensorsToLandmarks,
    tensors_to_world_landmarks: TensorsToLandmarks,
}

impl<'model> HandLandmarkerSession<'model> {
    const DETECTION_TO_RECT_ROTATION_OPTION: Option<(f32, usize, usize)> =
        Some((90. * std::f32::consts::PI / 180.0, 0, 2));

    /// Detect one image using this task session.
    #[inline(always)]
    pub fn detect(&mut self, input: &impl ImageToTensor) -> Result<HandLandmarkResults, Error> {
        let (img_w, img_h) = input.image_size();
        let hand_detection_result = self.hand_detector_session.detect(input)?;
        let mut hand_landmark_results = Vec::with_capacity(hand_detection_result.detections.len());

        for d in hand_detection_result.detections.iter() {
            // get roi
            let hand_rect = NormalizedRect::from_detection(
                &d,
                Self::DETECTION_TO_RECT_ROTATION_OPTION,
                img_w,
                img_h,
                false,
            )
            .transform(img_w, img_h, 2.6, 2.6, 0.0, -0.5, None, true);

            // image to tensor
            input.to_tensor(
                self.image_to_tensor_info,
                &super::ImageProcessingOptions::from_normalized_rect(&hand_rect),
                &mut self.input_buffer,
            )?;

            // set input and compute
            self.execution_ctx.set_input(
                0,
                self.hand_landmarker.input_tensor_type,
                self.input_tensor_shape,
                self.input_buffer.as_slice(),
            )?;
            self.execution_ctx.compute()?;

            // check hand presence score
            self.execution_ctx.get_output(
                self.hand_landmarker.score_buf_index,
                &mut self.score_of_hand_presence,
            )?;
            if self.score_of_hand_presence[0] < self.hand_landmarker.min_hand_presence_confidence()
            {
                continue;
            }

            // get handedness, left or right
            self.execution_ctx.get_output(
                self.hand_landmarker.handedness_buf_index,
                &mut self.score_of_handedness,
            )?;
            let category = if self.score_of_handedness[0] > 0.5 {
                self.categories_filter
                    .create_category(0, self.score_of_handedness[0])
                    .unwrap()
            } else {
                self.categories_filter
                    .create_category(1, 1. - self.score_of_handedness[0])
                    .unwrap()
            };

            // get landmarks
            self.execution_ctx.get_output(
                self.hand_landmarker.landmarks_buf_index,
                self.tensors_to_landmarks.landmark_buffer(),
            )?;
            let mut hand_landmarks = self.tensors_to_landmarks.result(true);
            self.execution_ctx.get_output(
                self.hand_landmarker.world_landmarks_buf_index,
                self.tensors_to_world_landmarks.landmark_buffer(),
            )?;
            let mut hand_world_landmarks = self.tensors_to_world_landmarks.result(false);

            // do projection
            crate::postprocess::projection_normalized_landmarks(
                &mut hand_landmarks,
                &hand_rect,
                false,
            );
            crate::postprocess::projection_world_landmark(&mut hand_world_landmarks, &hand_rect);

            hand_landmark_results.push(HandLandmarkResult {
                handedness: category,
                hand_landmarks,
                hand_world_landmarks,
            });
        }

        Ok(HandLandmarkResults(hand_landmark_results))
    }

    /// Detect input video stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn detect_for_video<InputVideoData: VideoData>(
        &mut self,
        video_data: InputVideoData,
    ) -> Result<VideoResultsIter<Self, InputVideoData>, Error> {
        Ok(VideoResultsIter::new(self, video_data))
    }
}

impl<'model> super::TaskSession for HandLandmarkerSession<'model> {
    type Result = HandLandmarkResults;

    #[inline]
    fn process_next(
        &mut self,
        _process_options: &super::ImageProcessingOptions,
        video_data: &mut impl VideoData,
    ) -> Result<Option<Self::Result>, Error> {
        // todo: video track optimize
        if let Some(frame) = video_data.next_frame()? {
            return self.detect(&frame).map(|r| Some(r));
        }
        Ok(None)
    }
}
