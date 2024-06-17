use super::HandLandmark;
use crate::postprocess::utils::{draw_landmarks_with_options, DefaultPixel, DrawLandmarksOptions};
use crate::postprocess::{Category, Landmarks, NormalizedLandmarks};
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};

/// A single hand landmark detection result.
#[derive(Debug)]
pub struct HandLandmarkResult {
    /// Classification of handedness.
    pub handedness: Category,
    /// Detected hand landmarks in normalized image coordinates.
    pub hand_landmarks: NormalizedLandmarks,
    /// Detected hand landmarks in world coordinates.
    pub hand_world_landmarks: Landmarks,
}

impl HandLandmarkResult {
    /// Draw this detection result to image with default options
    #[inline(always)]
    pub fn draw<I>(&self, img: &mut I)
    where
        I: image::GenericImage,
        I::Pixel: 'static + DefaultPixel,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        let mut options = DrawLandmarksOptions::default();
        options.connections = HandLandmark::CONNECTIONS;
        draw_landmarks_with_options(img, &self.hand_landmarks, &options);
    }

    /// Draw this detection result to image with options
    #[inline(always)]
    pub fn draw_with_options<I>(&self, img: &mut I, options: &DrawLandmarksOptions<I::Pixel>)
    where
        I: image::GenericImage,
        I::Pixel: 'static,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        draw_landmarks_with_options(img, &self.hand_landmarks, options);
    }
}

/// The hand landmarks detection result from HandLandmark
#[derive(Debug)]
pub struct HandLandmarkResults(pub Vec<HandLandmarkResult>);

impl Deref for HandLandmarkResults {
    type Target = Vec<HandLandmarkResult>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HandLandmarkResults {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for HandLandmarkResults {
    type Item = HandLandmarkResult;
    type IntoIter = std::vec::IntoIter<HandLandmarkResult>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Display for HandLandmarkResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Handedness: ")?;
        writeln!(f, "    Category #0:")?;
        write!(f, "{}", self.handedness)?;
        writeln!(f, "  Landmarks:")?;
        for (i, l) in self.hand_landmarks.iter().enumerate() {
            writeln!(
                f,
                "    Normalized Landmark #{} ({}):",
                i,
                HandLandmark::NAMES[i]
            )?;
            write!(f, "{}", l)?;
        }
        writeln!(f, "  WorldLandmarks:")?;
        for (i, l) in self.hand_world_landmarks.iter().enumerate() {
            writeln!(f, "    Landmark #{} ({}):", i, HandLandmark::NAMES[i])?;
            write!(f, "{}", l)?;
        }
        Ok(())
    }
}

impl Display for HandLandmarkResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            writeln!(f, "No HandLandmarkResult.")?;
        } else {
            for (i, r) in self.iter().enumerate() {
                writeln!(f, "HandLandmarkResult #{}", i)?;
                write!(f, "{}", r)?;
            }
        }
        Ok(())
    }
}
