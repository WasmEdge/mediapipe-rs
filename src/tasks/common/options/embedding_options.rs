pub struct EmbeddingOptions {
    /// Whether to normalize the returned feature vector with L2 norm.
    /// Use this option only if the model does not already contain a native L2_NORMALIZATION TF Lite Op.
    /// In most cases, this is already the case and L2 norm is thus achieved through TF Lite inference.
    pub l2_normalize: bool,

    /// Whether the returned embedding should be quantized to bytes via scalar quantization.
    /// Embeddings are implicitly assumed to be unit-norm and
    /// therefore any dimension is guaranteed to have a value in \[-1.0, 1.0\].
    /// Use the l2_normalize option if this is not the case.
    pub quantize: bool,
}

impl Default for EmbeddingOptions {
    #[inline(always)]
    fn default() -> Self {
        Self {
            l2_normalize: false,
            quantize: false,
        }
    }
}

macro_rules! embedding_options_impl {
    () => {
        /// Set whether to normalize the returned feature vector with L2 norm.
        /// Use this option only if the model does not already contain a native L2_NORMALIZATION TF Lite Op.
        /// In most cases, this is already the case and L2 norm is thus achieved through TF Lite inference.
        #[inline(always)]
        pub fn l2_normalize(mut self, l2_normalize: bool) -> Self {
            self.embedding_options.l2_normalize = l2_normalize;
            self
        }

        /// Set whether the returned embedding should be quantized to bytes via scalar quantization.
        /// Embeddings are implicitly assumed to be unit-norm and
        /// therefore any dimension is guaranteed to have a value in \[-1.0, 1.0\].
        /// Use the l2_normalize option if this is not the case.
        #[inline(always)]
        pub fn quantize(mut self, quantize: bool) -> Self {
            self.embedding_options.quantize = quantize;
            self
        }
    };
}

macro_rules! embedding_options_get_impl {
    () => {
        /// Get the task whether to normalize the returned feature vector with L2 norm.
        #[inline(always)]
        pub fn l2_normalize(&self) -> bool {
            self.build_options.embedding_options.l2_normalize
        }

        /// Get the task whether the returned embedding should be quantized to bytes via scalar quantization.
        #[inline(always)]
        pub fn quantize(&self) -> bool {
            self.build_options.embedding_options.quantize
        }
    };
}
