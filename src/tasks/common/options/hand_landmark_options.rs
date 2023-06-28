#[derive(Clone)]
pub(crate) struct HandLandmarkOptions {
    /// The maximum number of hands can be detected by the HandLandmarker.
    pub num_hands: i32,

    /// The minimum confidence score for the hand detection to be considered successful.
    pub min_hand_detection_confidence: f32,

    /// The minimum confidence score of hand presence score in the hand landmark detection.
    pub min_hand_presence_confidence: f32,

    /// The minimum confidence score for the hand tracking to be considered successful.
    pub min_tracking_confidence: f32,
}

impl Default for HandLandmarkOptions {
    #[inline(always)]
    fn default() -> Self {
        Self {
            num_hands: 1,
            min_hand_detection_confidence: 0.5,
            min_hand_presence_confidence: 0.5,
            min_tracking_confidence: 0.5,
        }
    }
}

macro_rules! hand_landmark_options_impl {
    () => {
        /// Set the maximum number of hands can be detected by the HandLandmarker.
        #[inline(always)]
        pub fn num_hands(mut self, num_hands: i32) -> Self {
            self.hand_landmark_options.num_hands = num_hands;
            self
        }

        /// Set the minimum confidence score for the hand detection to be considered successful.
        #[inline(always)]
        pub fn min_hand_detection_confidence(mut self, min_hand_detection_confidence: f32) -> Self {
            self.hand_landmark_options.min_hand_detection_confidence =
                min_hand_detection_confidence;
            self
        }

        /// Set the minimum confidence score of hand presence score in the hand landmark detection.
        #[inline(always)]
        pub fn min_hand_presence_confidence(mut self, min_hand_presence_confidence: f32) -> Self {
            self.hand_landmark_options.min_hand_presence_confidence = min_hand_presence_confidence;
            self
        }

        /// Set the minimum confidence score for the hand tracking to be considered successful.
        #[inline(always)]
        pub fn min_tracking_confidence(mut self, min_tracking_confidence: f32) -> Self {
            self.hand_landmark_options.min_tracking_confidence = min_tracking_confidence;
            self
        }
    };
}

macro_rules! hand_landmark_options_check {
    ( $self:ident ) => {{
        if $self.hand_landmark_options.num_hands == 0 {
            return Err(crate::Error::ArgumentError(
                "The number of max hands cannot be zero".into(),
            ));
        }
        if $self.hand_landmark_options.min_hand_presence_confidence < 0.
            || $self.hand_landmark_options.min_hand_presence_confidence > 1.
        {
            return Err(crate::Error::ArgumentError(format!(
                "The min_hand_presence_confidence must in range [0.0, 1.0], but got `{}`",
                $self.hand_landmark_options.min_hand_presence_confidence
            )));
        }
        if $self.hand_landmark_options.min_hand_detection_confidence < 0.
            || $self.hand_landmark_options.min_hand_detection_confidence > 1.
        {
            return Err(crate::Error::ArgumentError(format!(
                "The min_hand_detection_confidence must in range [0.0, 1.0], but got `{}`",
                $self.hand_landmark_options.min_hand_detection_confidence
            )));
        }
    }};
}

macro_rules! hand_landmark_options_get_impl {
    () => {
        /// Get the maximum number of hands can be detected by the HandLandmarker.
        #[inline(always)]
        pub fn num_hands(&self) -> i32 {
            self.build_options.hand_landmark_options.num_hands
        }

        /// Get the minimum confidence score for the hand detection to be considered successful.
        #[inline(always)]
        pub fn min_hand_detection_confidence(&self) -> f32 {
            self.build_options
                .hand_landmark_options
                .min_hand_detection_confidence
        }

        /// Get the minimum confidence score of hand presence score in the hand landmark detection.
        #[inline(always)]
        pub fn min_hand_presence_confidence(&self) -> f32 {
            self.build_options
                .hand_landmark_options
                .min_hand_presence_confidence
        }

        /// Get the minimum confidence score for the hand tracking to be considered successful.
        #[inline(always)]
        pub fn min_tracking_confidence(&self) -> f32 {
            self.build_options
                .hand_landmark_options
                .min_tracking_confidence
        }
    };
}
