use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};

use crate::postprocess::ClassificationResult;
use crate::tasks::vision::results::HandLandmarkResult;

/// The gesture recognition result from GestureRecognizer
#[derive(Debug)]
pub struct GestureRecognizerResult {
    /// Recognized hand gestures with sorted order such that the winning label is the first item in the list.
    pub gestures: ClassificationResult,
    /// hand landmark result
    pub hand_landmark: HandLandmarkResult,
}

/// The gesture recognition result list from GestureRecognizer
#[derive(Debug)]
pub struct GestureRecognizerResults(pub Vec<GestureRecognizerResult>);

impl Deref for GestureRecognizerResults {
    type Target = Vec<GestureRecognizerResult>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GestureRecognizerResults {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for GestureRecognizerResults {
    type Item = GestureRecognizerResult;
    type IntoIter = std::vec::IntoIter<GestureRecognizerResult>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl From<Vec<GestureRecognizerResult>> for GestureRecognizerResults {
    fn from(value: Vec<GestureRecognizerResult>) -> Self {
        GestureRecognizerResults(value)
    }
}

impl From<GestureRecognizerResults> for Vec<GestureRecognizerResult> {
    fn from(value: GestureRecognizerResults) -> Self {
        value.0
    }
}

impl Display for GestureRecognizerResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            writeln!(f, "No GestureRecognizerResult.")?;
        } else {
            for (i, r) in self.iter().enumerate() {
                writeln!(f, "GestureRecognizerResult #{}:", i)?;
                write!(f, "{}", r)?;
            }
        }
        Ok(())
    }
}

impl Display for GestureRecognizerResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Gestures: ")?;
        if let Some(t) = self.gestures.timestamp_ms {
            writeln!(f, "  Timestamp: {} ms", t)?;
        }
        if self.gestures.classifications.is_empty() {
            writeln!(f, "  No Classification")?;
        } else {
            for (i, c) in self.gestures.classifications.iter().enumerate() {
                writeln!(f, "  Classification #{}:", i)?;
                for (j, c) in c.categories.iter().enumerate() {
                    writeln!(f, "    Category #{}:", j)?;
                    write!(f, "{}", c)?;
                }
            }
        }
        write!(f, "{}", self.hand_landmark)?;
        Ok(())
    }
}
