use super::DefaultPixel;
use crate::postprocess::NormalizedLandmarks;
use image::{GenericImage, Pixel};
use imageproc::drawing;

/// draw landmarks options
#[derive(Debug)]
pub struct DrawLandmarksOptions<'a, P: Pixel> {
    pub line_color: P,
    pub landmark_color: P,
    pub connections: &'a [(usize, usize)],
    pub landmark_radius: i32,
    pub presence_threshold: f32,
    pub visibility_threshold: f32,
    // todo: add other options
}

impl<'a, P: Pixel> DrawLandmarksOptions<'a, P> {
    #[inline(always)]
    pub fn new(line_color: P, landmark_color: P) -> Self {
        Self {
            line_color,
            landmark_color,
            connections: &[],
            landmark_radius: 5,
            presence_threshold: 0.5,
            visibility_threshold: 0.5,
        }
    }

    #[inline(always)]
    pub fn connections(self, connections: &[(usize, usize)]) -> DrawLandmarksOptions<P> {
        DrawLandmarksOptions {
            line_color: self.line_color,
            landmark_color: self.landmark_color,
            landmark_radius: self.landmark_radius,
            presence_threshold: self.presence_threshold,
            visibility_threshold: self.visibility_threshold,
            connections,
        }
    }

    #[inline(always)]
    pub fn presence_threshold(mut self, presence_threshold: f32) -> Self {
        self.presence_threshold = presence_threshold;
        self
    }

    #[inline(always)]
    pub fn visibility_threshold(mut self, visibility_threshold: f32) -> Self {
        self.visibility_threshold = visibility_threshold;
        self
    }
}

impl<'a, P: Pixel + DefaultPixel> Default for DrawLandmarksOptions<'a, P> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            line_color: DefaultPixel::default(),
            landmark_color: DefaultPixel::default(),
            connections: &[],
            landmark_radius: 5,
            presence_threshold: 0.5,
            visibility_threshold: 0.5,
        }
    }
}

/// draw landmarks to image with default options
#[inline(always)]
pub fn draw_landmarks<I>(img: &mut I, normalized_landmarks: &NormalizedLandmarks)
where
    I: GenericImage,
    I::Pixel: 'static + DefaultPixel,
{
    draw_landmarks_with_options::<I>(img, normalized_landmarks, &Default::default())
}

macro_rules! check_threshold {
    ( $l:ident, $options:ident ) => {
        if let Some(v) = $l.visibility {
            if v < $options.visibility_threshold {
                continue;
            }
        }
        if let Some(p) = $l.presence {
            if p < $options.presence_threshold {
                continue;
            }
        }
    };
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
    for (id_start, id_end) in options.connections {
        let l_start = normalized_landmarks.get(*id_start).unwrap();
        let l_end = normalized_landmarks.get(*id_end).unwrap();
        check_threshold!(l_start, options);
        check_threshold!(l_end, options);
        drawing::draw_line_segment_mut(
            img,
            (l_start.x * img_w, l_start.y * img_h),
            (l_end.x * img_w, l_end.y * img_h),
            options.line_color,
        );
    }

    for normalized_landmark in normalized_landmarks.iter().rev() {
        check_threshold!(normalized_landmark, options);
        drawing::draw_filled_circle_mut(
            img,
            (
                (normalized_landmark.x * img_w) as i32,
                (normalized_landmark.y * img_h) as i32,
            ),
            options.landmark_radius,
            options.landmark_color,
        );
    }
}
