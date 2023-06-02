use image::{Luma, LumaA, Rgb, Rgba};

pub trait DefaultPixel {
    fn default() -> Self;
}

impl DefaultPixel for Rgb<u8> {
    #[inline(always)]
    fn default() -> Self {
        Rgb::from([255u8, 0u8, 0u8])
    }
}

impl DefaultPixel for Rgba<u8> {
    #[inline(always)]
    fn default() -> Self {
        Rgba::from([255u8, 0u8, 0u8, 1u8])
    }
}

impl DefaultPixel for Luma<u8> {
    #[inline(always)]
    fn default() -> Self {
        Luma::from([255])
    }
}

impl DefaultPixel for LumaA<u8> {
    #[inline(always)]
    fn default() -> Self {
        LumaA::from([255, 1])
    }
}
