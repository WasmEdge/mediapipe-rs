use super::NormalizedRect;
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};

/// Landmark represents a point in 3D space with x, y, z coordinates. The
/// landmark coordinates are in meters. z represents the landmark depth, and the
/// smaller the value the closer the world landmark is to the camera.
#[derive(Debug)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Landmark visibility. Should stay unset if not supported.
    /// Float score of whether landmark is visible or occluded by other objects.
    /// Landmark considered as invisible also if it is not present on the screen
    /// (out of scene bounds). Depending on the model, visibility value is either a
    /// sigmoid or an argument of sigmoid.
    pub visibility: Option<f32>,
    /// Landmark presence. Should stay unset if not supported.
    /// Float score of whether landmark is present on the scene (located within
    /// scene bounds). Depending on the model, presence value is either a result of
    /// sigmoid or an argument of sigmoid function to get landmark presence probability.
    pub presence: Option<f32>,
    /// Landmark name. Should stay unset if not supported.
    pub name: Option<String>,
}

/// A list of Landmarks.
#[derive(Debug)]
pub struct Landmarks(pub Vec<Landmark>);

impl Deref for Landmarks {
    type Target = Vec<Landmark>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Landmarks {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for Landmarks {
    type Item = Landmark;
    type IntoIter = std::vec::IntoIter<Landmark>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

/// A normalized version of above Landmark struct. All coordinates should be within [0, 1].
pub type NormalizedLandmark = Landmark;

/// A list of NormalizedLandmarks.
pub type NormalizedLandmarks = Landmarks;

impl Landmark {
    pub const LANDMARK_TOLERANCE: f32 = 1e-6;
}

impl Eq for Landmark {}

impl PartialEq for Landmark {
    fn eq(&self, other: &Self) -> bool {
        return (self.x - other.x).abs() < Self::LANDMARK_TOLERANCE
            && (self.y - other.y) < Self::LANDMARK_TOLERANCE
            && (self.z - other.z) < Self::LANDMARK_TOLERANCE;
    }
}

impl Display for Landmark {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "      x:       {}", self.x)?;
        writeln!(f, "      y:       {}", self.y)?;
        writeln!(f, "      z:       {}", self.z)
    }
}

impl Display for Landmarks {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Landmarks:")?;
        if self.is_empty() {
            return writeln!(f, "  No Landmark");
        }
        for i in 0..self.len() {
            writeln!(f, "    Landmark #{}:", i)?;
            let l = self.get(i).unwrap();
            write!(f, "{}", l)?;
        }
        Ok(())
    }
}

pub(crate) fn projection_normalized_landmarks(
    landmarks: &mut NormalizedLandmarks,
    normalized_rect: &NormalizedRect,
    mut ignore_rotation: bool,
) {
    if normalized_rect.rotation.is_none() {
        ignore_rotation = true;
    }

    let mut cos = 0.;
    let mut sin = 0.;
    if !ignore_rotation {
        let angle = normalized_rect.rotation.unwrap();
        cos = angle.cos();
        sin = angle.sin();
    }
    let rect_x_min = max_f32!(0.0, normalized_rect.x_center - normalized_rect.width * 0.5);
    let rect_y_min = max_f32!(0.0, normalized_rect.y_center - normalized_rect.height * 0.5);
    let rect_x_max = min_f32!(1.0, normalized_rect.x_center + normalized_rect.width * 0.5);
    let rect_y_max = min_f32!(1.0, normalized_rect.y_center + normalized_rect.height * 0.5);
    let width = rect_x_max - rect_x_min;
    let height = rect_y_max - rect_y_min;

    for l in landmarks.iter_mut() {
        if !ignore_rotation {
            let x = l.x - 0.5;
            let y = l.y - 0.5;
            l.x = 0.5 + x * cos - y * sin;
            l.y = 0.5 + x * sin + y * cos;
        }
        l.x = rect_x_min + l.x * width;
        l.y = rect_y_min + l.y * height;

        l.z *= normalized_rect.width;
    }
}

pub(crate) fn projection_world_landmark(
    landmarks: &mut Landmarks,
    normalized_rect: &NormalizedRect,
) {
    if let Some(angle) = normalized_rect.rotation {
        let cos = angle.cos();
        let sin = angle.sin();
        for l in landmarks.iter_mut() {
            let x = cos * l.x - sin * l.y;
            let y = sin * l.x + cos * l.y;
            l.x = x;
            l.y = y;
        }
    }
}
