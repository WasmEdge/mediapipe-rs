mod builder;
mod face_landmark_blendshapes;
mod face_landmark_connections;
mod result;

use super::{
    FaceDetector, FaceDetectorBuilder, FaceDetectorSession
};
pub use builder::FaceLandmarkerBuilder;
pub use face_landmark_blendshapes::FaceLandmarkBlendshapes;
pub use face_landmark_connections::FaceLandmarkConnections;
pub use result::{FaceLandmarkResult, FaceLandmarkResults};

use crate::model::ModelResourceTrait;
use crate::postprocess::{NormalizedRect, TensorsToLandmarks, VideoResultsIter};
use crate::preprocess::vision::{ImageToTensor, ImageToTensorInfo, VideoData};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs face landmark on images and video frames.
pub struct FaceLandmarker {
    build_options: FaceLandmarkerBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    face_detector: FaceDetector,

    score_buf_index: usize,
    landmarks_buf_index: usize,

    // only one input and one output
    input_tensor_type: TensorType,
}

impl FaceLandmarker {
    const LANDMARKS_NORMALIZE_Z: f32 = 1.0;

    detector_impl!(FaceLandmarkerSession, FaceLandmarkResults);

    face_landmark_options_get_impl!();

    /// Get the subtask: face detector.
    #[inline(always)]
    pub fn subtask_face_detector(&self) -> &FaceDetector {
        &self.face_detector
    }

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<FaceLandmarkerSession, Error> {
        let image_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);

        let landmarks_out =
            get_type_and_quantization!(self.model_resource, self.landmarks_buf_index);
        let landmarks_shape = model_resource_check_and_get_impl!(
            self.model_resource,
            output_tensor_shape,
            self.landmarks_buf_index
        );

        // 468 is the standard number of facial landmarks used in MediaPipe's Face Mesh model (kMeshLandmarksNum).
        // For models including the iris, this number increases to 478.
        // Reference: https://github.com/google/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h
        let mut tensors_to_landmarks =
            TensorsToLandmarks::new(468, landmarks_out, landmarks_shape)?;
        tensors_to_landmarks
            .set_image_size(image_to_tensor_info.width(), image_to_tensor_info.height());
        tensors_to_landmarks.set_normalize_z(Self::LANDMARKS_NORMALIZE_Z);

        let face_detector_session = self.face_detector.new_session()?;
        let execution_ctx = self.graph.init_execution_context()?;

        Ok(FaceLandmarkerSession {
            face_landmarker: self,
            execution_ctx,
            face_detector_session,
            image_to_tensor_info,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
            score_of_face_presence: [0.],
            tensors_to_landmarks,
        })
    }
}

/// Session to run inference.
/// If process multiple images or videos, reuse it can get better performance.
pub struct FaceLandmarkerSession<'model> {
    face_landmarker: &'model FaceLandmarker,
    execution_ctx: GraphExecutionContext<'model>,

    face_detector_session: FaceDetectorSession<'model>,

    image_to_tensor_info: &'model ImageToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_buffer: Vec<u8>,
    score_of_face_presence: [f32; 1],
    tensors_to_landmarks: TensorsToLandmarks,
}

impl<'model> FaceLandmarkerSession<'model> {
    const DETECTION_TO_RECT_ROTATION_OPTION: Option<(f32, usize, usize)> = Some((0.0, 0, 1));

    /// Detect one image using this task session.
    #[inline(always)]
    pub fn detect(&mut self, input: &impl ImageToTensor) -> Result<FaceLandmarkResults, Error> {
        let (img_w, img_h) = input.image_size();
        let face_detection_result = self.face_detector_session.detect(input)?;
        let mut face_landmark_results = Vec::with_capacity(face_detection_result.detections.len());

        for d in face_detection_result.detections.iter() {
            // get roi
            let face_rect = NormalizedRect::from_detection(
                &d,
                Self::DETECTION_TO_RECT_ROTATION_OPTION,
                img_w,
                img_h,
                false,
            )
            .transform(img_w, img_h, 1.5, 1.5, 0.0, 0.0, None, false);

            // image to tensor
            input.to_tensor(
                self.image_to_tensor_info,
                &super::ImageProcessingOptions::from_normalized_rect(&face_rect),
                &mut self.input_buffer,
            )?;

            // set input and compute
            self.execution_ctx.set_input(
                0,
                self.face_landmarker.input_tensor_type,
                self.input_tensor_shape,
                self.input_buffer.as_slice(),
            )?;
            self.execution_ctx.compute()?;

            // check face presence score
            self.execution_ctx.get_output(
                self.face_landmarker.score_buf_index,
                &mut self.score_of_face_presence,
            )?;
            if self.score_of_face_presence[0] < self.face_landmarker.min_face_presence_confidence()
            {
                continue;
            }

            // get landmarks
            self.execution_ctx.get_output(
                self.face_landmarker.landmarks_buf_index,
                self.tensors_to_landmarks.landmark_buffer(),
            )?;
            let mut face_landmarks = self.tensors_to_landmarks.result(true);

            // do projection
            crate::postprocess::projection_normalized_landmarks(
                &mut face_landmarks,
                &face_rect,
                false,
            );

            face_landmark_results.push(FaceLandmarkResult {
                face_landmarks,
            });
        }

        Ok(FaceLandmarkResults(face_landmark_results))
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

impl<'model> super::TaskSession for FaceLandmarkerSession<'model> {
    type Result = FaceLandmarkResults;

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
