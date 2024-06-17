use super::DefaultPixel;
use crate::postprocess::DetectionResult;
use ab_glyph::{FontArc, PxScale};
use image::{GenericImage, Pixel};
use imageproc::{definitions::Clamp, drawing};

/// draw detection results options
#[derive(Debug)]
pub struct DrawDetectionsOptions<'font, P: Pixel> {
    pub rect_colors: Vec<P>,

    pub draw_keypoint: bool,
    pub keypoint_colors: Vec<P>,
    pub keypoint_radius_percent: f32,

    pub draw_label: bool,
    pub font: &'font FontArc,
    pub font_color: P,
    pub font_scale: f32,
}

impl<P: Pixel + DefaultPixel> Default for DrawDetectionsOptions<'static, P> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            rect_colors: vec![P::default()],

            draw_keypoint: true,
            keypoint_colors: Vec::default(),
            keypoint_radius_percent: 0.01,

            draw_label: true,
            font: super::default_font(),
            font_color: P::white(),
            font_scale: 0.05,
        }
    }
}

/// draw detection results to image with default options
#[inline(always)]
pub fn draw_detection<I>(img: &mut I, detection_result: &DetectionResult)
where
    I: GenericImage,
    I::Pixel: 'static + DefaultPixel,
    <I::Pixel as Pixel>::Subpixel: Into<f32> + Clamp<f32>,
{
    draw_detection_with_options::<I>(img, detection_result, &Default::default())
}

/// draw detection results to image with options
pub fn draw_detection_with_options<I>(
    img: &mut I,
    detection_result: &DetectionResult,
    options: &DrawDetectionsOptions<I::Pixel>,
) where
    I: GenericImage,
    <I::Pixel as Pixel>::Subpixel: Into<f32> + Clamp<f32>,
    I::Pixel: 'static,
{
    let img_w = img.width() as f32;
    let img_h = img.height() as f32;
    let img_min = if img_h > img_w { img_w } else { img_h };
    let keypoint_radius = (img_min * options.keypoint_radius_percent) as i32;
    let default_keypoint_color = if options.keypoint_colors.is_empty() {
        options.rect_colors[0]
    } else {
        options.keypoint_colors[0]
    };
    let font_scale = PxScale {
        x: img_min * options.font_scale,
        y: img_min * options.font_scale,
    };

    for (d_id, d) in detection_result.detections.iter().rev().enumerate() {
        let left = d.bounding_box.left * img_w;
        let right = d.bounding_box.right * img_w;
        let top = d.bounding_box.top * img_h;
        let bottom = d.bounding_box.bottom * img_h;
        let rect = imageproc::rect::Rect::at(left as i32, top as i32)
            .of_size((right - left) as u32, (bottom - top) as u32);
        let rect_color = match options.rect_colors.get(d_id) {
            Some(c) => *c,
            None => options.rect_colors[0],
        };
        drawing::draw_hollow_rect_mut(img, rect, rect_color);

        if options.draw_keypoint {
            if let Some(ref ks) = d.key_points {
                for (k_id, k) in ks.iter().enumerate() {
                    let x = k.x * img_w;
                    let y = k.y * img_h;
                    let color = match options.keypoint_colors.get(k_id) {
                        Some(c) => *c,
                        None => default_keypoint_color,
                    };
                    drawing::draw_filled_circle_mut(
                        img,
                        (x as i32, y as i32),
                        keypoint_radius,
                        color,
                    );
                }
            }
        }

        if options.draw_label && !d.categories.is_empty() {
            let mut y = top;
            for c in d.categories.iter() {
                let score_p = c.score * 100.;
                let text = match c.display_name.as_ref() {
                    Some(n) => format!("{:.2}% {}", score_p, n),
                    None => match c.category_name.as_ref() {
                        Some(n) => format!("{:.2}% {}", score_p, n),
                        None => format!("{:.4}%", score_p),
                    },
                };
                let font_region_size = drawing::text_size(font_scale, options.font, text.as_str());
                drawing::draw_filled_rect_mut(
                    img,
                    imageproc::rect::Rect::at(left as i32, y as i32)
                        .of_size(font_region_size.0 as u32 + 1, font_region_size.1 as u32 + 1),
                    rect_color,
                );
                drawing::draw_text_mut(
                    img,
                    options.font_color,
                    left as i32,
                    y as i32,
                    font_scale,
                    options.font,
                    text.as_str(),
                );

                y += font_region_size.1 as f32 + 5.;
            }
        }
    }
}
