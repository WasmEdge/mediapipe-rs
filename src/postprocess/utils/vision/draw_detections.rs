use super::DefaultPixel;
use crate::postprocess::DetectionResult;
use image::{GenericImage, Pixel};
use imageproc::drawing;

/// draw detection results options
#[derive(Debug)]
pub struct DrawDetectionsOptions<P: Pixel> {
    pub border_color: P,
    pub draw_keypoint: bool,
    pub keypoint_color: P,
    pub keypoint_radius: i32,
    // todo: add other options
}

impl<P: Pixel> DrawDetectionsOptions<P> {
    #[inline(always)]
    pub fn new(border_color: P, keypoint_color: P) -> Self {
        Self {
            border_color,
            draw_keypoint: false,
            keypoint_color,
            keypoint_radius: 5,
        }
    }
}

impl<P: Pixel + DefaultPixel> Default for DrawDetectionsOptions<P> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            border_color: DefaultPixel::default(),
            draw_keypoint: false,
            keypoint_color: DefaultPixel::default(),
            keypoint_radius: 5,
        }
    }
}

/// draw detection results to image with default options
#[inline(always)]
pub fn draw_detection<I>(img: &mut I, detection_result: &DetectionResult)
where
    I: GenericImage,
    I::Pixel: 'static + DefaultPixel,
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
    I::Pixel: 'static,
{
    let img_w = img.width() as f32;
    let img_h = img.height() as f32;
    for d in detection_result.detections.iter().rev() {
        let left = d.bounding_box.left * img_w;
        let right = d.bounding_box.right * img_w;
        let top = d.bounding_box.top * img_h;
        let bottom = d.bounding_box.bottom * img_h;
        let rect = imageproc::rect::Rect::at(left as i32, top as i32)
            .of_size((right - left) as u32, (bottom - top) as u32);
        drawing::draw_hollow_rect_mut(img, rect, options.border_color);

        if options.draw_keypoint {
            if let Some(ref ks) = d.key_points {
                for k in ks {
                    let x = k.x * img_w;
                    let y = k.y * img_h;
                    drawing::draw_filled_circle_mut(
                        img,
                        (x as i32, y as i32),
                        options.keypoint_radius,
                        options.keypoint_color,
                    );
                }
            }
        }

        // todo: draw label
    }
}
