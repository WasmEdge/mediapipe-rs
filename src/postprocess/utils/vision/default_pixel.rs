use image::{Luma, LumaA, Rgb, Rgba};

pub trait DefaultPixel {
    /// Return default color, now is `red`.
    fn default() -> Self;

    /// Return `white` color.
    fn white() -> Self;
}

impl DefaultPixel for Rgb<u8> {
    #[inline(always)]
    fn default() -> Self {
        Rgb::from([255u8, 0u8, 0u8])
    }

    #[inline(always)]
    fn white() -> Self {
        Rgb::from([255u8, 255u8, 255u8])
    }
}

impl DefaultPixel for Rgba<u8> {
    #[inline(always)]
    fn default() -> Self {
        Rgba::from([255u8, 0u8, 0u8, 1u8])
    }

    #[inline(always)]
    fn white() -> Self {
        Rgba::from([255u8, 255u8, 255u8, 1u8])
    }
}

impl DefaultPixel for Luma<u8> {
    #[inline(always)]
    fn default() -> Self {
        // red to gray
        Luma::from([77u8])
    }

    #[inline(always)]
    fn white() -> Self {
        Luma::from([255u8])
    }
}

impl DefaultPixel for LumaA<u8> {
    #[inline(always)]
    fn default() -> Self {
        // red to gray
        LumaA::from([77, 1u8])
    }

    #[inline(always)]
    fn white() -> Self {
        LumaA::from([255u8, 1u8])
    }
}
