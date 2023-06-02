use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Sub};

/// Defines a rectangle, used e.g. as part of detection results or as input
/// region-of-interest.
#[derive(Debug, Clone)]
pub struct Rect<T: Sized + Clone + Copy + Send + Sync + Debug + Display + Add + Sub + 'static> {
    pub left: T,
    pub top: T,
    pub right: T,
    pub bottom: T,
}

impl<T> Display for Rect<T>
where
    T: Sized + Clone + Copy + Send + Sync + Debug + Display + Add + Sub + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Box: (left: {}, top: {}, right: {}, bottom: {})",
            self.left, self.top, self.right, self.bottom
        )
    }
}

impl PartialEq<Self> for Rect<u32> {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left
            && self.top == other.top
            && self.right == other.right
            && self.bottom == other.bottom
    }
}

impl Eq for Rect<u32> {}

impl PartialEq<Self> for Rect<f32> {
    fn eq(&self, other: &Self) -> bool {
        const RECT_TOLERANCE: f32 = 1e-4;

        (self.left - other.left).abs() < RECT_TOLERANCE
            && (self.top - other.top).abs() < RECT_TOLERANCE
            && (self.right - other.right).abs() < RECT_TOLERANCE
            && (self.bottom - other.bottom).abs() < RECT_TOLERANCE
    }
}

impl Eq for Rect<f32> {}

impl Rect<u32> {
    pub fn to_rect(&self, image_height: u32, image_width: u32) -> Rect<f32> {
        Rect {
            left: (self.left as f32) / image_width as f32,
            top: (self.top as f32) / image_height as f32,
            right: (self.right as f32) / image_width as f32,
            bottom: (self.bottom as f32) / image_height as f32,
        }
    }
}

impl Rect<f32> {
    #[inline(always)]
    pub fn to_rect(&self, image_height: u32, image_width: u32) -> Rect<u32> {
        Rect {
            left: (self.left * image_width as f32) as u32,
            top: (self.top * image_height as f32) as u32,
            right: (self.right * image_width as f32) as u32,
            bottom: (self.bottom * image_height as f32) as u32,
        }
    }

    #[inline(always)]
    pub fn intersect(&self, other: &Rect<f32>) -> Option<Rect<f32>> {
        if !(other.left > self.right
            || other.right < self.left
            || other.top > self.bottom
            || other.bottom < self.top)
        {
            Some(Rect {
                left: max_f32!(self.left, other.left),
                right: min_f32!(self.right, other.right),
                top: max_f32!(self.top, other.top),
                bottom: min_f32!(self.bottom, other.bottom),
            })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn union(&self, other: &Rect<f32>) -> Rect<f32> {
        Rect {
            left: min_f32!(self.left, other.left),
            top: min_f32!(self.top, other.top),
            right: max_f32!(self.right, other.right),
            bottom: max_f32!(self.bottom, other.bottom),
        }
    }

    #[inline(always)]
    pub fn width(&self) -> f32 {
        self.right - self.left
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        self.bottom - self.top
    }

    #[inline(always)]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_area() {
        let r = Rect {
            left: -1.,
            right: 1.,
            top: -1.,
            bottom: 1.,
        };
        assert_eq!(r.area(), 4.);
    }

    #[test]
    fn test_intersect() {
        let r1 = Rect {
            left: 10.,
            top: 10.,
            right: 30.,
            bottom: 30.,
        };
        let r2 = Rect {
            left: 20.,
            top: 20.,
            right: 50.,
            bottom: 50.,
        };
        let r3 = Rect {
            left: 70.,
            top: 70.,
            right: 90.,
            bottom: 90.,
        };
        let r4 = Rect {
            left: 15.,
            top: 5.,
            right: 25.,
            bottom: 35.,
        };

        assert_eq!(
            r1.intersect(&r2).unwrap(),
            Rect {
                left: 20.,
                top: 20.,
                right: 30.,
                bottom: 30.,
            }
        );
        assert!(r1.intersect(&r3).is_none());
        assert_eq!(
            r1.intersect(&r4).unwrap(),
            Rect {
                left: 15.,
                top: 10.,
                right: 25.,
                bottom: 30.,
            }
        );
    }
}
