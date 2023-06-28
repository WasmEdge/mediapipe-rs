use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

/// Defines a single classification result.
///
/// The label maps packed into the TFLite Model Metadata [1] are used to populate
/// the 'category_name' and 'display_name' fields.
///
/// [1]: https://www.tensorflow.org/lite/convert/metadata
#[derive(Debug)]
pub struct Category {
    /// The index of the category in the classification model output.
    pub index: u32,

    /// The score for this category, e.g. (but not necessarily) a probability in \[0,1\].
    pub score: f32,

    /// The optional ID for the category, read from the label map packed in the
    /// TFLite Model Metadata if present. Not necessarily human-readable.
    pub category_name: Option<String>,

    /// The optional human-readable name for the category, read from the label map
    /// packed in the TFLite Model Metadata if present.
    pub display_name: Option<String>,
}

impl Display for Category {
    #[inline(always)]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(ref name) = self.category_name {
            writeln!(f, "      Category name: \"{}\"", name)?;
        } else {
            writeln!(f, "      Category name: None")?;
        }
        if let Some(ref name) = self.display_name {
            writeln!(f, "      Display name:  \"{}\"", name)?;
        } else {
            writeln!(f, "      Display name:  None")?;
        }
        writeln!(f, "      Score:         {}", self.score)?;
        writeln!(f, "      Index:         {}", self.index)
    }
}

impl Eq for Category {}

impl PartialEq<Self> for Category {
    fn eq(&self, other: &Self) -> bool {
        self.index.eq(&other.index)
    }
}

impl PartialOrd<Self> for Category {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for Category {
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.partial_cmp(&self.score).unwrap()
    }
}
