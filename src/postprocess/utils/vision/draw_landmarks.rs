use super::DefaultPixel;
use crate::postprocess::{Landmark, NormalizedLandmarks};
use image::{GenericImage, Pixel};
use imageproc::drawing;

/// draw landmarks options
#[derive(Debug)]
pub struct DrawLandmarksOptions<'a, P: Pixel> {
    pub line_colors: Vec<P>,
    pub landmark_colors: Vec<P>,
    pub connections: &'a [(usize, usize)],
    pub landmark_radius_percent: f32,
    pub presence_threshold: f32,
    pub visibility_threshold: f32,
    // todo: add other options
}

impl<'a, P: Pixel> DrawLandmarksOptions<'a, P> {
    pub fn new(line_colors: Vec<P>, landmark_colors: Vec<P>) -> Self {
        Self {
            line_colors,
            landmark_colors,
            connections: &[],
            landmark_radius_percent: 0.01,
            presence_threshold: 0.5,
            visibility_threshold: 0.5,
        }
    }

    pub fn connections(self, connections: &[(usize, usize)]) -> DrawLandmarksOptions<'_, P> {
        DrawLandmarksOptions {
            line_colors: self.line_colors,
            landmark_colors: self.landmark_colors,
            landmark_radius_percent: self.landmark_radius_percent,
            presence_threshold: self.presence_threshold,
            visibility_threshold: self.visibility_threshold,
            connections,
        }
    }

    #[inline]
    pub fn presence_threshold(mut self, presence_threshold: f32) -> Self {
        self.presence_threshold = presence_threshold;
        self
    }

    #[inline]
    pub fn visibility_threshold(mut self, visibility_threshold: f32) -> Self {
        self.visibility_threshold = visibility_threshold;
        self
    }

    #[inline]
    pub fn landmark_radius_percent(mut self, landmark_radius_percent: f32) -> Self {
        self.landmark_radius_percent = landmark_radius_percent;
        self
    }
}

impl<'a, P: Pixel + DefaultPixel> Default for DrawLandmarksOptions<'a, P> {
    fn default() -> Self {
        Self {
            line_colors: vec![DefaultPixel::default()],
            landmark_colors: Vec::default(),
            connections: &[],
            landmark_radius_percent: 0.01,
            presence_threshold: 0.5,
            visibility_threshold: 0.5,
        }
    }
}

/// draw landmarks to image with default options
#[inline]
pub fn draw_landmarks<I>(img: &mut I, normalized_landmarks: &NormalizedLandmarks)
where
    I: GenericImage,
    I::Pixel: 'static + DefaultPixel,
{
    draw_landmarks_with_options::<I>(img, normalized_landmarks, &Default::default())
}

fn passes_threshold<P: Pixel>(landmark: &Landmark, options: &DrawLandmarksOptions<P>) -> bool {
    landmark
        .visibility
        .is_none_or(|v| v >= options.visibility_threshold)
        && landmark
            .presence
            .is_none_or(|p| p >= options.presence_threshold)
}

/// draw landmarks to image with options
pub fn draw_landmarks_with_options<I>(
    img: &mut I,
    normalized_landmarks: &NormalizedLandmarks,
    options: &DrawLandmarksOptions<I::Pixel>,
) where
    I: GenericImage,
    I::Pixel: 'static,
{
    let img_w = img.width() as f32;
    let img_h = img.height() as f32;
    let img_min = if img_h > img_w { img_w } else { img_h };
    let landmark_radius = (img_min * options.landmark_radius_percent) as i32;

    let default_color = if !options.line_colors.is_empty() {
        options.line_colors[0]
    } else {
        options.landmark_colors[0]
    };

    for (c_id, (id_start, id_end)) in options.connections.iter().enumerate() {
        let l_start = normalized_landmarks.get(*id_start).unwrap();
        let l_end = normalized_landmarks.get(*id_end).unwrap();
        if !passes_threshold(l_start, options) || !passes_threshold(l_end, options) {
            continue;
        }
        let color = match options.line_colors.get(c_id) {
            Some(c) => *c,
            None => default_color,
        };
        drawing::draw_line_segment_mut(
            img,
            (l_start.x * img_w, l_start.y * img_h),
            (l_end.x * img_w, l_end.y * img_h),
            color,
        );
    }

    for (l_id, normalized_landmark) in normalized_landmarks.iter().rev().enumerate() {
        if !passes_threshold(normalized_landmark, options) {
            continue;
        }
        let color = match options.landmark_colors.get(l_id) {
            Some(c) => *c,
            None => default_color,
        };
        drawing::draw_filled_circle_mut(
            img,
            (
                (normalized_landmark.x * img_w) as i32,
                (normalized_landmark.y * img_h) as i32,
            ),
            landmark_radius,
            color,
        );
    }
}
