use crate::postprocess::impl_result_list;
use crate::postprocess::utils::{draw_landmarks_with_options, DefaultPixel, DrawLandmarksOptions};
use crate::postprocess::NormalizedLandmarks;
use std::fmt::{Display, Formatter};

/// A single face landmark detection result.
#[derive(Debug)]
pub struct FaceLandmarkResult {
    /// Detected face landmarks in normalized image coordinates.
    pub face_landmarks: NormalizedLandmarks,
}

impl FaceLandmarkResult {
    /// Draw this detection result to image with default options
    #[inline(always)]
    pub fn draw<I>(&self, img: &mut I)
    where
        I: image::GenericImage,
        I::Pixel: 'static + DefaultPixel,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        let options = DrawLandmarksOptions::default();
        draw_landmarks_with_options(img, &self.face_landmarks, &options);
    }

    /// Draw this detection result to image with options
    #[inline(always)]
    pub fn draw_with_options<I>(&self, img: &mut I, options: &DrawLandmarksOptions<I::Pixel>)
    where
        I: image::GenericImage,
        I::Pixel: 'static,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        draw_landmarks_with_options(img, &self.face_landmarks, options);
    }
}

/// The face landmarks detection result from FaceLandmark
#[derive(Debug)]
pub struct FaceLandmarkResults(pub Vec<FaceLandmarkResult>);

impl_result_list!(FaceLandmarkResults, FaceLandmarkResult);

impl Display for FaceLandmarkResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Landmarks:")?;
        for (i, l) in self.face_landmarks.iter().enumerate() {
            writeln!(f, "    Normalized Landmark #{}:", i)?;
            write!(f, "{}", l)?;
        }
        Ok(())
    }
}

impl Display for FaceLandmarkResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            writeln!(f, "No FaceLandmarkResult.")?;
        } else {
            for (i, r) in self.iter().enumerate() {
                writeln!(f, "FaceLandmarkResult #{}", i)?;
                write!(f, "{}", r)?;
            }
        }
        Ok(())
    }
}
