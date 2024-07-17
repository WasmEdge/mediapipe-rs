#[derive(Clone)]
pub(crate) struct FaceLandmarkOptions {
    /// The maximum number of faces can be detected by the FaceLandmarker.
    pub num_faces: i32,

    /// The minimum confidence score for the face detection to be considered successful.
    pub min_face_detection_confidence: f32,

    /// The minimum confidence score of face presence score in the face landmark detection.
    pub min_face_presence_confidence: f32,

    /// The minimum confidence score for the face tracking to be considered successful.
    pub min_tracking_confidence: f32,

    /// Whether Face Landmarker outputs face blendshapes.
    /// Face blendshapes are used for rendering the 3D face model.
    pub output_face_blendshapes: bool,

    /// Whether FaceLandmarker outputs the facial transformation matrix.
    /// FaceLandmarker uses the matrix to transform the face landmarks from a canonical face model
    /// to the detected face, so users can apply effects on the detected landmarks.
    pub output_facial_transformation_matrixes: bool,
}

impl Default for FaceLandmarkOptions {
    #[inline(always)]
    fn default() -> Self {
        Self {
            num_faces: 1,
            min_face_detection_confidence: 0.5,
            min_face_presence_confidence: 0.5,
            min_tracking_confidence: 0.5,
            output_face_blendshapes: false,
            output_facial_transformation_matrixes: false,
        }
    }
}

macro_rules! face_landmark_options_impl {
    () => {
        /// Set the maximum number of faces can be detected by the FaceLandmarker.
        #[inline(always)]
        pub fn num_faces(mut self, num_faces: i32) -> Self {
            self.face_landmark_options.num_faces = num_faces;
            self
        }

        /// Set the minimum confidence score for the face detection to be considered successful.
        #[inline(always)]
        pub fn min_face_detection_confidence(mut self, min_face_detection_confidence: f32) -> Self {
            self.face_landmark_options.min_face_detection_confidence =
                min_face_detection_confidence;
            self
        }

        /// Set the minimum confidence score of face presence score in the face landmark detection.
        #[inline(always)]
        pub fn min_face_presence_confidence(mut self, min_face_presence_confidence: f32) -> Self {
            self.face_landmark_options.min_face_presence_confidence = min_face_presence_confidence;
            self
        }

        /// Set the minimum confidence score for the face tracking to be considered successful.
        #[inline(always)]
        pub fn min_tracking_confidence(mut self, min_tracking_confidence: f32) -> Self {
            self.face_landmark_options.min_tracking_confidence = min_tracking_confidence;
            self
        }

        /// Set whether FaceLandmarker outputs face blendshapes.
        pub fn output_face_blendshapes(mut self, output_face_blendshapes: bool) -> Self {
            self.face_landmark_options.output_face_blendshapes = output_face_blendshapes;
            self
        }

        /// Set whether FaceLandmarker outputs the facial transformation matrix.
        pub fn output_facial_transformation_matrixes(
            mut self,
            output_facial_transformation_matrixes: bool,
        ) -> Self {
            self.face_landmark_options.output_facial_transformation_matrixes =
                output_facial_transformation_matrixes;
            self
        }
    };
}

macro_rules! face_landmark_options_check {
    ( $self:ident ) => {{
        if $self.face_landmark_options.num_faces == 0 {
            return Err(crate::Error::ArgumentError(
                "The number of max faces cannot be zero".into(),
            ));
        }
        if $self.face_landmark_options.min_face_presence_confidence < 0.
            || $self.face_landmark_options.min_face_presence_confidence > 1.
        {
            return Err(crate::Error::ArgumentError(format!(
                "The min_face_presence_confidence must in range [0.0, 1.0], but got `{}`",
                $self.face_landmark_options.min_face_presence_confidence
            )));
        }
        if $self.face_landmark_options.min_face_detection_confidence < 0.
            || $self.face_landmark_options.min_face_detection_confidence > 1.
        {
            return Err(crate::Error::ArgumentError(format!(
                "The min_face_detection_confidence must in range [0.0, 1.0], but got `{}`",
                $self.face_landmark_options.min_face_detection_confidence
            )));
        }
    }};
}

macro_rules! face_landmark_options_get_impl {
    () => {
        /// Get the maximum number of faces can be detected by the FaceLandmarker.
        #[inline(always)]
        pub fn num_faces(&self) -> i32 {
            self.build_options.face_landmark_options.num_faces
        }

        /// Get the minimum confidence score for the face detection to be considered successful.
        #[inline(always)]
        pub fn min_face_detection_confidence(&self) -> f32 {
            self.build_options
                .face_landmark_options
                .min_face_detection_confidence
        }

        /// Get the minimum confidence score of face presence score in the face landmark detection.
        #[inline(always)]
        pub fn min_face_presence_confidence(&self) -> f32 {
            self.build_options
                .face_landmark_options
                .min_face_presence_confidence
        }

        /// Get the minimum confidence score for the face tracking to be considered successful.
        #[inline(always)]
        pub fn min_tracking_confidence(&self) -> f32 {
            self.build_options
                .face_landmark_options
                .min_tracking_confidence
        }

        /// Get whether FaceLandmarker outputs face blendshapes.
        pub fn output_face_blendshapes(&self) -> bool {
            self.build_options.face_landmark_options.output_face_blendshapes
        }

        /// Get whether FaceLandmarker outputs the facial transformation matrix.
        pub fn output_facial_transformation_matrixes(&self) -> bool {
            self.build_options
                .face_landmark_options
                .output_facial_transformation_matrixes
        }
    };
}
