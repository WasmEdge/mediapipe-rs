use crate::postprocess::NormalizedRect;

/// Image crop params
#[derive(Debug, Clone)]
pub struct CropRect {
    pub x_min: f32,
    pub y_min: f32,
    pub width: f32,
    pub height: f32,
}

impl CropRect {
    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Result<Self, crate::Error> {
        if !(0. ..=1.).contains(&top) {
            return Err(crate::Error::ArgumentError(format!(
                "Rect top must in range [0, 1], but got `{}`",
                top
            )));
        }
        if !(0. ..=1.).contains(&bottom) {
            return Err(crate::Error::ArgumentError(format!(
                "Rect bottom must in range [0, 1], but got `{}`",
                bottom
            )));
        }
        if !(0. ..=1.).contains(&left) {
            return Err(crate::Error::ArgumentError(format!(
                "Rect left must in range [0, 1], but got `{}`",
                left
            )));
        }
        if !(0. ..=1.).contains(&right) {
            return Err(crate::Error::ArgumentError(format!(
                "Rect right must in range [0, 1], but got `{}`",
                right
            )));
        }
        let width = right - left;
        if width <= 0. {
            return Err(crate::Error::ArgumentError(format!(
                "Rect left must less than right, but got `left({})` >= `right({})`",
                left, right
            )));
        }
        let height = bottom - top;
        if height <= 0. {
            return Err(crate::Error::ArgumentError(format!(
                "Rect top must less than bottom, but got `top({})` >= `bottom({})`",
                top, bottom
            )));
        }

        Ok(Self {
            x_min: left,
            y_min: top,
            width,
            height,
        })
    }
}

impl<'a> From<&'a NormalizedRect> for CropRect {
    fn from(value: &'a NormalizedRect) -> Self {
        let mut width = value.width;
        let mut height = value.height;
        let x_min = (value.x_center - width / 2.).max(0.);
        let y_min = (value.y_center - height / 2.).max(0.);
        if x_min + width > 1. {
            width = 1. - x_min;
        }
        if y_min + height > 1. {
            height = 1. - y_min;
        }
        Self {
            x_min,
            y_min,
            width,
            height,
        }
    }
}

#[cfg(test)]
mod test {
    use super::CropRect;
    #[test]
    fn test_crop_rect() {
        assert!(CropRect::new(0.1, 0.2, 0.5, 0.7,).is_ok());
        assert!(CropRect::new(-1., 1., 1., 1.,).is_err());
        assert!(CropRect::new(1.1, 1., 1., 1.,).is_err());
        assert!(CropRect::new(0.5, 0.4, 1., 0.3,).is_err());
        assert!(CropRect::new(0.9, 0.4, 0.4, 1.,).is_err());
    }

    #[test]
    fn test_crop_rect_rejects_nan() {
        assert!(CropRect::new(f32::NAN, 0.2, 0.5, 0.7).is_err());
        assert!(CropRect::new(0.1, f32::NAN, 0.5, 0.7).is_err());
        assert!(CropRect::new(0.1, 0.2, f32::NAN, 0.7).is_err());
        assert!(CropRect::new(0.1, 0.2, 0.5, f32::NAN).is_err());
    }
}
