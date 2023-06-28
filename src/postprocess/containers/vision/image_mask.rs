use image::{ImageBuffer, Luma};

/// each pixel represents the prediction confidence, usually in the [0, 1] range.
pub type ImageConfidenceMask = ImageBuffer<Luma<f32>, Vec<f32>>;

/// each pixel represents the class which the pixel in the original image was predicted to belong to.
pub type ImageCategoryMask = ImageBuffer<Luma<u8>, Vec<u8>>;
