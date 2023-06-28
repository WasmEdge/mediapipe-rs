/// Embedding result for a given embedder head.
///
/// One and only one of the two 'float_embedding' and 'quantized_embedding' will
/// contain data, based on whether or not the embedder was configured to perform scalar quantization.
#[derive(Debug)]
pub struct Embedding {
    /// The index of the embedder head (i.e. output tensor) this embedding comes from. This is useful for multi-head models.
    pub head_index: usize,
    /// The optional name of the embedder head, as provided in the TFLite Model
    /// Metadata [1] if present. This is useful for multi-head models.
    ///
    /// [1]: https://www.tensorflow.org/lite/convert/metadata
    pub head_name: Option<String>,

    /// Floating-point embedding. Empty if the embedder was configured to perform scalar-quantization.
    pub float_embedding: Vec<f32>,

    /// Scalar-quantized embedding. Empty if the embedder was not configured to perform scalar quantization.
    pub quantized_embedding: Vec<i8>,
}

/// Defines embedding results of a model.
pub struct EmbeddingResult {
    /// The embedding results for each head of the model.
    pub embeddings: Vec<Embedding>,

    /// The optional timestamp (in milliseconds) of the start of the chunk of data
    /// corresponding to these results.
    ///
    /// This is only used for classification on time series (e.g. audio
    /// classification). In these use cases, the amount of data to process might
    /// exceed the maximum size that the model can process: to solve this, the
    /// input data is split into multiple chunks starting at different timestamps.
    pub timestamp_ms: Option<u64>,
}

macro_rules! check_size_eq {
    ( $len_1:expr, $len_2:expr ) => {
        if $len_1 != $len_2 {
            return Err(crate::Error::ArgumentError(format!(
                "Cannot compute cosine similarity between embeddings of different sizes ({} vs. {})",
                $len_1,
                $len_2
            )));
        }
    };
}

impl Embedding {
    /// Utility function to compute cosine similarity [1] between two embeddings. May
    /// return an InvalidArgumentError if e.g. the embeddings are of different types
    /// (quantized vs. float), have different sizes, or have a an L2-norm of 0.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Cosine_similarity
    /// [2]: https://github.com/google/mediapipe/blob/master/mediapipe/tasks/cc/components/utils/cosine_similarity.cc
    ///
    pub fn cosine_similarity(&self, other: &Self) -> Result<f64, crate::Error> {
        if !self.float_embedding.is_empty() && !other.float_embedding.is_empty() {
            check_size_eq!(self.float_embedding.len(), other.float_embedding.len());
            return Self::cosine_similarity_inner(
                self.float_embedding.as_slice(),
                other.float_embedding.as_slice(),
            );
        }
        if !self.quantized_embedding.is_empty() && !other.quantized_embedding.is_empty() {
            check_size_eq!(
                self.quantized_embedding.len(),
                other.quantized_embedding.len()
            );
            return Self::cosine_similarity_inner(
                self.quantized_embedding.as_slice(),
                other.quantized_embedding.as_slice(),
            );
        }
        Err(crate::Error::ArgumentError(
            "Cannot compute cosine similarity between quantized and float embeddings".into(),
        ))
    }

    fn cosine_similarity_inner<T: Sized + Copy + Into<f64>>(
        u: &[T],
        v: &[T],
    ) -> Result<f64, crate::Error> {
        assert_eq!(u.len(), v.len());
        let len = u.len();

        let mut dot_product = 0f64;
        let mut norm_u = 0f64;
        let mut norm_v = 0f64;
        for i in 0..len {
            let u = u[i].into();
            let v = v[i].into();
            dot_product += u * v;
            norm_u += u * u;
            norm_v += v * v;
        }

        if norm_u <= 0f64 || norm_v <= 0f64 {
            return Err(crate::Error::ArgumentError(format!(
                "Cannot compute cosine similarity on embedding with 0 norm"
            )));
        }

        Ok(dot_product / (norm_u * norm_v).sqrt())
    }
}
