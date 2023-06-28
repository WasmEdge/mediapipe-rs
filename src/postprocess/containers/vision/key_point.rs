use std::fmt::{Display, Formatter};

/// A keypoint, defined by the coordinates (x, y), normalized
/// by the image dimensions.
#[derive(Debug)]
pub struct NormalizedKeypoint {
    /// x in normalized image coordinates.
    pub x: f32,
    /// y in normalized image coordinates.
    pub y: f32,
    /// optional label of the keypoint.
    pub label: Option<String>,
    /// optional score of the keypoint.
    pub score: Option<f32>,
}

impl Display for NormalizedKeypoint {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "      Coordinates: ({},{})", self.x, self.y)?;
        if let Some(ref l) = self.label {
            writeln!(f, "      Label:       \"{}\"", l)?;
        } else {
            writeln!(f, "      Label:       None")?;
        }
        if let Some(ref s) = self.score {
            writeln!(f, "      Score:       \"{}\"", s)
        } else {
            writeln!(f, "      Score:       None")
        }
    }
}
