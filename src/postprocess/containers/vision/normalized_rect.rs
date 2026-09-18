use super::{Detection, Rect};

/// A rectangle with rotation in normalized coordinates. The values of box center
/// location and size are within [0, 1].
#[derive(Debug)]
pub struct NormalizedRect {
    /// Location of the center of the rectangle in image coordinates.
    /// The (0.0, 0.0) point is at the (top, left) corner.
    pub x_center: f32,
    pub y_center: f32,

    /// Size of the rectangle.
    pub height: f32,
    pub width: f32,

    /// Rotation angle is clockwise in radians. [default = 0.0]
    pub rotation: Option<f32>,

    /// Optional unique id to help associate different NormalizedRects to each other.
    pub rect_id: Option<u64>,
}

impl NormalizedRect {
    #[inline(always)]
    fn init_from_rect(rect: &Rect<f32>) -> Self {
        let width = rect.right - rect.left;
        let height = rect.bottom - rect.top;
        Self {
            x_center: rect.left + width / 2.,
            y_center: rect.top + height / 2.,
            width,
            height,
            rotation: None,
            rect_id: None,
        }
    }

    /// create from detection
    /// * rotation_option: (angle in radians, start_key_point_index, end_key_point_index)
    ///   angle is counter-clockwise
    /// * use_keypoint: if true, use key points, else use detection box
    pub(crate) fn from_detection(
        detection: &Detection,
        rotation_option: Option<(f32, usize, usize)>,
        img_w: u32,
        img_h: u32,
        use_keypoint: bool,
    ) -> Self {
        let mut r = if use_keypoint {
            assert!(detection.key_points.as_ref().unwrap().len() > 1);
            let mut rect = Rect {
                left: f32::MAX,
                top: f32::MAX,
                right: f32::MIN,
                bottom: f32::MIN,
            };
            for k in detection.key_points.as_ref().unwrap() {
                rect.left = k.x.min(rect.left);
                rect.top = k.y.min(rect.top);
                rect.right = k.x.max(rect.right);
                rect.bottom = k.y.max(rect.bottom);
            }
            Self::init_from_rect(&rect)
        } else {
            Self::init_from_rect(&detection.bounding_box)
        };
        if let Some((angle, s_id, e_id)) = rotation_option {
            let key_points = detection.key_points.as_ref().unwrap();
            let x0 = key_points[s_id].x * img_w as f32;
            let y0 = key_points[s_id].y * img_h as f32;
            let x1 = key_points[e_id].x * img_w as f32;
            let y1 = key_points[e_id].y * img_h as f32;
            let rotation = angle - (-(y1 - y0)).atan2(x1 - x0);

            // change range to -pi,pi
            r.rotation = Some(Self::normalize_radians(rotation))
        }
        r
    }

    /// geometric transformation
    ///
    /// rotation is counter-clockwise in radians
    /// to_square_long: true: use long side, false: use short side
    pub(crate) fn transform(
        &self,
        (img_w, img_h): (u32, u32),
        (scale_x, scale_y): (f32, f32),
        (shift_x, shift_y): (f32, f32),
        rotation: Option<f32>,
        to_square_long: bool,
    ) -> Self {
        let img_w = img_w as f32;
        let img_h = img_h as f32;

        let self_r = self.rotation.unwrap_or(0.);
        let rotation = if let Some(r) = rotation {
            Self::normalize_radians(self_r + r)
        } else {
            self_r
        };
        let mut width = self.width;
        let mut height = self.height;
        let (x_center, y_center) = if rotation == 0. {
            (
                self.x_center + width * shift_x,
                self.y_center + height * shift_y,
            )
        } else {
            let cos_r = rotation.cos();
            let sin_r = rotation.sin();
            let x_shift =
                (img_w * width * shift_x * cos_r - img_h * height * shift_y * sin_r) / img_w;
            let y_shift =
                (img_w * width * shift_x * sin_r + img_h * height * shift_y * cos_r) / img_h;
            (self.x_center + x_shift, self.y_center + y_shift)
        };

        if to_square_long {
            // long
            let long_side = (width * img_w).max(height * img_h);
            width = long_side / img_w;
            height = long_side / img_h;
        } else {
            // short
            let short_side = (width * img_w).min(height * img_h);
            width = short_side / img_w;
            height = short_side / img_h;
        }

        Self {
            x_center,
            y_center,
            width: width * scale_x,
            height: height * scale_y,
            rotation: Some(rotation),
            rect_id: None,
        }
    }

    #[inline(always)]
    fn normalize_radians(angle: f32) -> f32 {
        angle
            - 2. * std::f32::consts::PI
                * ((angle - (-std::f32::consts::PI)) / (2. * std::f32::consts::PI)).floor()
    }
}
