use crate::postprocess::Category;
use std::fmt::{Display, Formatter};

/// Defines classification results for a given classifier head.
#[derive(Debug)]
pub struct Classifications {
    /// The index of the classifier head (i.e. output tensor) these categories
    /// refer to. This is useful for multi-head models.
    pub head_index: usize,
    /// The optional name of the classifier head, as provided in the TFLite Model
    /// Metadata [1] if present. This is useful for multi-head models.
    ///
    /// [1]: https://www.tensorflow.org/lite/convert/metadata
    pub head_name: Option<String>,
    /// The array of predicted categories, usually sorted by descending scores,
    /// e.g. from high to low probability.
    pub categories: Vec<Category>,
}

/// Defines classification results of a model.
#[derive(Debug)]
pub struct ClassificationResult {
    /// The classification results for each head of the model.
    pub classifications: Vec<Classifications>,
    /// The optional timestamp (in milliseconds) of the start of the chunk of data
    /// corresponding to these results.
    ///
    /// This is only used for classification on time series (e.g. audio
    /// classification). In these use cases, the amount of data to process might
    /// exceed the maximum size that the model can process: to solve this, the
    /// input data is split into multiple chunks starting at different timestamps.
    pub timestamp_ms: Option<u64>,
}

impl Display for ClassificationResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ClassificationResult:")?;
        if let Some(t) = self.timestamp_ms {
            writeln!(f, "  Timestamp: {} ms", t)?;
        }

        if self.classifications.is_empty() {
            return writeln!(f, "  No Classification");
        }
        for i in 0..self.classifications.len() {
            writeln!(f, "  Classification #{}:", i)?;
            let c = self.classifications.get(i).unwrap();
            if let Some(ref name) = c.head_name {
                writeln!(f, "    Head name: {}", name)?;
                writeln!(f, "    Head index: {}", c.head_index)?;
            }
            for j in 0..c.categories.len() {
                writeln!(f, "    Category #{}:", j)?;
                write!(f, "{}", c.categories.get(j).unwrap())?;
            }
        }
        Ok(())
    }
}
