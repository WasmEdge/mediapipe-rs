use crate::postprocess::utils::{draw_detection_with_options, DefaultPixel, DrawDetectionsOptions};
use crate::postprocess::{Category, NormalizedKeypoint, Rect};
use std::fmt::{Display, Formatter};

/// Detection for a single bounding box.
#[derive(Debug)]
pub struct Detection {
    /// A vector of detected categories.
    pub categories: Vec<Category>,
    /// The bounding box location.
    pub bounding_box: Rect<f32>,
    /// Optional list of key points associated with the detection. Key points
    /// represent interesting points related to the detection. For example, the
    /// key points represent the eye, ear and mouth from face detection model. Or
    /// in the template matching detection, e.g. KNIFT, they can represent the
    /// feature points for template matching.
    pub key_points: Option<Vec<NormalizedKeypoint>>,
}

/// Detection results of a model.
#[derive(Debug)]
pub struct DetectionResult {
    /// A vector of Detections.
    pub detections: Vec<Detection>,
}

impl DetectionResult {
    /// Draw this detection result to image with default options
    #[inline(always)]
    pub fn draw<I>(&self, img: &mut I)
    where
        I: image::GenericImage,
        I::Pixel: 'static + DefaultPixel,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        draw_detection_with_options(img, self, &Default::default());
    }

    /// Draw this detection result to image with options
    #[inline(always)]
    pub fn draw_with_options<I>(&self, img: &mut I, options: &DrawDetectionsOptions<I::Pixel>)
    where
        I: image::GenericImage,
        I::Pixel: 'static,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        draw_detection_with_options(img, self, options);
    }
}

impl Display for DetectionResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DetectionResult:")?;
        if self.detections.is_empty() {
            return writeln!(f, "  No Detection");
        }
        for i in 0..self.detections.len() {
            writeln!(f, "  Detection #{}:", i)?;
            let d = self.detections.get(i).unwrap();
            write!(f, "    {}", d.bounding_box)?;
            for (id, c) in d.categories.iter().enumerate() {
                writeln!(f, "    Category #{}:", id)?;
                write!(f, "{}", c)?;
            }
            if let Some(ref key_points) = d.key_points {
                for (id, key_point) in key_points.iter().enumerate() {
                    writeln!(f, "    KeyPoint #{}:", id)?;
                    write!(f, "{}", key_point)?;
                }
            }
        }
        Ok(())
    }
}
