pub(crate) struct ClassificationOptions {
    /// The maximum number of top-scored classification results to return.
    /// `None` returns all available results. `Some(0)` is an invalid argument.
    pub max_results: Option<usize>,

    /// Score threshold to override the one provided in the model metadata (if
    /// any). Results below this value are rejected.
    pub score_threshold: f32,

    /// The locale to use for display names specified through the TFLite Model
    /// Metadata, if any. Defaults to English.
    pub display_names_locale: String,

    /// The allow list of category names. If non-empty, detection results whose
    /// category name is not in this set will be filtered out. Duplicate or unknown
    /// category names are ignored. Mutually exclusive with category_deny_list.
    pub category_allow_list: Vec<String>,

    /// The deny list of category names. If non-empty, detection results whose
    /// category name is in this set will be filtered out. Duplicate or unknown
    /// category names are ignored. Mutually exclusive with category_allow_list.
    pub category_deny_list: Vec<String>,
}

impl Default for ClassificationOptions {
    fn default() -> Self {
        Self {
            display_names_locale: "en".into(),
            max_results: None,
            score_threshold: -1.0f32,
            category_allow_list: Vec::new(),
            category_deny_list: Vec::new(),
        }
    }
}

macro_rules! classification_options_impl {
    () => {
        /// Set the locale to use for display names specified through the TFLite Model Metadata, if any.
        /// Defaults to English.
        #[inline(always)]
        pub fn display_names_locale(mut self, display_names_locale: String) -> Self {
            self.classification_options.display_names_locale = display_names_locale;
            self
        }

        /// Set the maximum number of top-scored classification results to return.
        /// By default all available results are returned. `0` is an invalid argument.
        #[inline(always)]
        pub fn max_results(mut self, max_results: usize) -> Self {
            self.classification_options.max_results = Some(max_results);
            self
        }

        /// Set score threshold to override the one provided in the model metadata (if any).
        /// Results below this value are rejected.
        #[inline(always)]
        pub fn score_threshold(mut self, score_threshold: f32) -> Self {
            self.classification_options.score_threshold = score_threshold;
            self
        }

        /// Set the allow list of category names.
        /// If non-empty, detection results whose category name is not in this set will be filtered out.
        /// Duplicate or unknown category names are ignored.
        /// Mutually exclusive with category_deny_list.
        #[inline(always)]
        pub fn category_allow_list(mut self, category_allow_list: Vec<String>) -> Self {
            self.classification_options.category_allow_list = category_allow_list;
            self
        }

        /// Set the deny list of category names.
        /// If non-empty, detection results whose category name is in this set will be filtered out.
        /// Duplicate or unknown category names are ignored.
        /// Mutually exclusive with category_allow_list.
        #[inline(always)]
        pub fn category_deny_list(mut self, category_deny_list: Vec<String>) -> Self {
            self.classification_options.category_deny_list = category_deny_list;
            self
        }
    };
}

impl ClassificationOptions {
    pub(crate) fn check(&self) -> Result<(), crate::Error> {
        if self.max_results == Some(0) {
            return Err(crate::Error::ArgumentError(
                "The number of max results cannot be zero".into(),
            ));
        }
        if !self.category_allow_list.is_empty() && !self.category_deny_list.is_empty() {
            return Err(crate::Error::ArgumentError(
                "Cannot use both `category_allow_list` and `category_deny_list`".into(),
            ));
        }
        Ok(())
    }
}

macro_rules! classification_options_get_impl {
    () => {
        /// Get the maximum number of top-scored classification results to return.
        /// `None` returns all available results.
        #[inline(always)]
        pub fn max_results(&self) -> Option<usize> {
            self.build_options.classification_options.max_results
        }

        /// Get score threshold.
        #[inline(always)]
        pub fn score_threshold(&self) -> f32 {
            self.build_options.classification_options.score_threshold
        }

        /// Set the locale to use for display names.
        #[inline(always)]
        pub fn display_names_locale(&self) -> &String {
            &self
                .build_options
                .classification_options
                .display_names_locale
        }

        /// Get the allow list of category names.
        #[inline(always)]
        pub fn category_allow_list(&self) -> &Vec<String> {
            &self
                .build_options
                .classification_options
                .category_allow_list
        }

        /// Get the deny list of category names.
        #[inline(always)]
        pub fn category_deny_list(&self) -> &Vec<String> {
            &self.build_options.classification_options.category_deny_list
        }
    };
}
