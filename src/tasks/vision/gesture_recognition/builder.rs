use super::*;
use crate::model::ZipFiles;
use crate::tasks::common::{BaseTaskOptions, ClassificationOptions, HandLandmarkOptions};

/// Configure the build options of a new **Gesture Recognition** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct GestureRecognizerBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) classification_options: ClassificationOptions,
    pub(super) custom_classification_options: ClassificationOptions,
    pub(super) hand_landmark_options: HandLandmarkOptions,
}

impl Default for GestureRecognizerBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl GestureRecognizerBuilder {
    base_task_options_impl!(GestureRecognizer);

    classification_options_impl!();

    hand_landmark_options_impl!();

    /// **Set options for custom classification model! (if custom model exists)**
    ///
    /// Set the locale to use for display names specified through the TFLite Model Metadata, if any.
    /// Defaults to English.
    #[inline(always)]
    pub fn custom_classifier_display_names_locale(mut self, display_names_locale: String) -> Self {
        self.custom_classification_options.display_names_locale = display_names_locale;
        self
    }

    /// **Set options for custom classification model! (if custom model exists)**
    ///
    /// Set the maximum number of top-scored classification results to return.
    /// If < 0, all available results will be returned.
    /// If 0, an invalid argument error is returned.
    #[inline(always)]
    pub fn custom_classifier_max_results(mut self, max_results: i32) -> Self {
        self.custom_classification_options.max_results = max_results;
        self
    }

    /// **Set options for custom classification model! (if custom model exists)**
    ///
    /// Set score threshold to override the one provided in the model metadata (if any).
    /// Results below this value are rejected.
    #[inline(always)]
    pub fn custom_classifier_score_threshold(mut self, score_threshold: f32) -> Self {
        self.custom_classification_options.score_threshold = score_threshold;
        self
    }

    /// **Set options for custom classification model! (if custom model exists)**
    ///
    /// Set the allow list of category names.
    /// If non-empty, detection results whose category name is not in this set will be filtered out.
    /// Duplicate or unknown category names are ignored.
    /// Mutually exclusive with category_deny_list.
    #[inline(always)]
    pub fn custom_classifier_category_allow_list(
        mut self,
        category_allow_list: Vec<String>,
    ) -> Self {
        self.custom_classification_options.category_allow_list = category_allow_list;
        self
    }

    /// **Set options for custom classification model! (if custom model exists)**
    ///
    /// Set the deny list of category names.
    /// If non-empty, detection results whose category name is in this set will be filtered out.
    /// Duplicate or unknown category names are ignored.
    /// Mutually exclusive with category_allow_list.
    #[inline(always)]
    pub fn custom_classifier_category_deny_list(mut self, category_deny_list: Vec<String>) -> Self {
        self.custom_classification_options.category_deny_list = category_deny_list;
        self
    }

    const HAND_LANDMARK_SUBTASK_CANDIDATE_NAMES: &'static [&'static str] =
        &["hand_landmarker.task"];
    const HAND_GESTURE_CANDIDATE_NAMES: &'static [&'static str] = &["hand_gesture_recognizer.task"];

    const GESTURE_EMBEDDER_CANDIDATE_NAMES: &'static [&'static str] = &["gesture_embedder.tflite"];
    const GESTURE_CANNED_GESTURE_CLASSIFIER_CANDIDATE_NAMES: &'static [&'static str] =
        &["canned_gesture_classifier.tflite"];
    const GESTURE_CUSTOM_GESTURE_CLASSIFIER_CANDIDATE_NAMES: &'static [&'static str] =
        &["custom_gesture_classifier.tflite"];

    const TASK_NAME: &'static str = "GestureRecognizer";

    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            classification_options: Default::default(),
            custom_classification_options: Default::default(),
            hand_landmark_options: Default::default(),
        }
    }

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(self, buffer: impl AsRef<[u8]>) -> Result<GestureRecognizer, Error> {
        classification_options_check!(self, classification_options);
        classification_options_check!(self, custom_classification_options);
        hand_landmark_options_check!(self);
        let buf = buffer.as_ref();

        let zip_file = ZipFiles::new(buf)?;
        let hand_gesture_bundle_file = crate::model::search_file_in_zip(
            &zip_file,
            Self::HAND_GESTURE_CANDIDATE_NAMES,
            Self::TASK_NAME,
        )?;

        // subtask: landmark
        let landmark_task_file = crate::model::search_file_in_zip(
            &zip_file,
            Self::HAND_LANDMARK_SUBTASK_CANDIDATE_NAMES,
            Self::TASK_NAME,
        )?;
        let hand_landmarker = HandLandmarkerBuilder {
            base_task_options: BaseTaskOptions {
                device: self.base_task_options.device,
            },
            hand_landmark_options: self.hand_landmark_options.clone(),
        }
        .build_from_buffer(landmark_task_file)?;

        let zip_file = ZipFiles::new(hand_gesture_bundle_file)?;
        // search files, build graph and check model

        let gesture_embed_file = crate::model::search_file_in_zip(
            &zip_file,
            Self::GESTURE_EMBEDDER_CANDIDATE_NAMES,
            Self::TASK_NAME,
        )?;
        let gesture_embed_model_resources = crate::model::parse_model(gesture_embed_file)?;
        gesture_embed_model_resources.check_tensor_counts(Some(3), 1)?;
        // now only support fp32 type for embed model
        for i in 0..3 {
            gesture_embed_model_resources.check_input_tensor_type(i, TensorType::F32)?;
        }
        gesture_embed_model_resources.check_output_tensor_type(0, TensorType::F32)?;
        let gesture_embed_out_size = gesture_embed_model_resources
            .expect_output_tensor_shape(0)?
            .iter()
            .product::<usize>();
        let gesture_embed_graph = crate::tasks::common::build_graph(
            gesture_embed_model_resources.as_ref(),
            self.base_task_options.device,
            gesture_embed_file,
        )?;

        let canned_file = crate::model::search_file_in_zip(
            &zip_file,
            Self::GESTURE_CANNED_GESTURE_CLASSIFIER_CANDIDATE_NAMES,
            Self::TASK_NAME,
        )?;
        let canned_classifier = GestureClassifier::build(
            canned_file,
            self.base_task_options.device,
            gesture_embed_out_size,
        )?;

        let custom_classifier = Self::GESTURE_CUSTOM_GESTURE_CLASSIFIER_CANDIDATE_NAMES
            .iter()
            .find_map(|name| zip_file.get_file(name))
            .map(|file| {
                GestureClassifier::build(
                    file,
                    self.base_task_options.device,
                    gesture_embed_out_size,
                )
            })
            .transpose()?;

        Ok(GestureRecognizer {
            build_options: self,
            gesture_embed_model_resources,
            gesture_embed_graph,
            canned_classifier,
            custom_classifier,
            hand_landmarker,
            gesture_embed_out_size,
        })
    }
}
