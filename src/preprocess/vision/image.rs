extern crate image as image_crate;

use super::*;
pub(super) use image_crate::{
    imageops, DynamicImage, EncodableLayout, GenericImageView, ImageBuffer, Pixel, Rgb, RgbImage,
};

const IMAGE_RESIZE_FILTER: imageops::FilterType = imageops::FilterType::Gaussian;

macro_rules! get_rgb_mean_std_from_info {
    ( $info:ident ) => {{
        let r_mean = $info.normalization_options.0.get(0).unwrap();
        let r_std = $info.normalization_options.1.get(0).unwrap();
        let g_mean = $info.normalization_options.0.get(1).unwrap_or(r_mean);
        let g_std = $info.normalization_options.1.get(1).unwrap_or(r_std);
        let b_mean = $info.normalization_options.0.get(1).unwrap_or(r_mean);
        let b_std = $info.normalization_options.1.get(1).unwrap_or(r_std);
        (r_mean, r_std, g_mean, g_std, b_mean, b_std)
    }};
}

impl ImageToTensor for DynamicImage {
    #[inline(always)]
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        info: &ImageToTensorInfo,
        process_options: &ImageProcessingOptions,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        match info.color_space {
            ImageColorSpaceType::GRAYSCALE => {
                unimplemented!()
            }
            // we treat unknown as rgb8
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                if let Some(rgb) = self.as_rgb8() {
                    rgb.to_tensor(info, process_options, output_buffer)
                } else {
                    self.to_rgb8()
                        .to_tensor(info, process_options, output_buffer)
                }
            }
        }
    }

    /// return image size: (weight, height)
    #[inline(always)]
    fn image_size(&self) -> (u32, u32) {
        self.dimensions()
    }
}

impl ImageToTensor for RgbImage {
    #[inline]
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        info: &ImageToTensorInfo,
        process_options: &ImageProcessingOptions,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        let mut tmp_rgb_img;

        let mut rgb_img = if let Some(ref roi) = process_options.region_of_interest {
            // check roi
            let weight = self.width() as f32;
            let height = self.height() as f32;
            let x = (roi.x_min * weight) as u32;
            let y = (roi.y_min * height) as u32;
            let w = (roi.width * weight) as u32;
            let h = (roi.height * height) as u32;
            tmp_rgb_img = imageops::crop_imm(self, x, y, w, h).to_image();
            let abs = process_options.rotation.abs();
            if abs > 0.01 {
                if (abs - std::f32::consts::PI).abs() < 0.01 {
                    imageops::rotate180_in_place(&mut tmp_rgb_img);
                } else {
                    tmp_rgb_img = ops_inner::rotate_any(&tmp_rgb_img, process_options.rotation);
                }
            }
            &tmp_rgb_img
        } else {
            let abs = process_options.rotation.abs();
            if abs > 0.01 {
                if (abs - std::f32::consts::PI).abs() < 0.01 {
                    tmp_rgb_img = imageops::rotate180(self);
                } else {
                    tmp_rgb_img = ops_inner::rotate_any(self, process_options.rotation);
                }
                &tmp_rgb_img
            } else {
                self
            }
        };

        let width = info.width();
        let height = info.height();
        if width != rgb_img.width() || height != rgb_img.height() {
            tmp_rgb_img = imageops::resize(rgb_img, width, height, IMAGE_RESIZE_FILTER);
            rgb_img = &tmp_rgb_img;
        }

        if info.color_space == ImageColorSpaceType::GRAYSCALE {
            // todo: gray image
            unimplemented!()
        }

        rgb8_image_buffer_to_tensor(rgb_img, info, output_buffer)
    }

    /// return image size: (weight, height)
    #[inline(always)]
    fn image_size(&self) -> (u32, u32) {
        self.dimensions()
    }
}

#[inline(always)]
pub(super) fn rgb8_image_buffer_to_tensor<'t, Container>(
    img: &'t ImageBuffer<Rgb<u8>, Container>,
    info: &ImageToTensorInfo,
    output_buffer: &mut impl AsMut<[u8]>,
) -> Result<(), Error>
where
    Container: std::ops::Deref<Target = [u8]>,
{
    debug_assert!(
        img.width() == info.width()
            && img.height() == info.height()
            && info.color_space != ImageColorSpaceType::GRAYSCALE
    );

    info.normalization_options.0.get(0).unwrap_or(&1f32);

    let data_layout = info.image_data_layout;
    let res = output_buffer.as_mut();
    let mut res_index = 0;
    match info.tensor_type {
        TensorType::F32 => {
            let (r_mean, r_std, g_mean, g_std, b_mean, b_std) = get_rgb_mean_std_from_info!(info);
            let bytes = img.as_bytes();
            debug_assert_eq!(res.len(), bytes.len() * std::mem::size_of::<f32>());

            let hw = (img.width() * img.height()) as usize;
            return match data_layout {
                ImageDataLayout::NHWC => {
                    let mut i = 0;
                    while i < bytes.len() {
                        let f = ((bytes[i] as f32) - r_mean) / r_std;
                        res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                        res_index += 4;
                        let f = ((bytes[i + 1] as f32) - g_mean) / g_std;
                        res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                        res_index += 4;
                        let f = ((bytes[i + 2] as f32) - b_mean) / b_std;
                        res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                        res_index += 4;
                        i += 3;
                    }
                    Ok(())
                }
                ImageDataLayout::NCHW | ImageDataLayout::CHWN => {
                    for start in 0..3 {
                        let mut i = start as usize;
                        while i < hw {
                            let f = ((bytes[i] as f32) - r_mean) / r_std;
                            res[res_index..res_index + 4].copy_from_slice(&f.to_ne_bytes());
                            res_index += 4;
                            i += 3;
                        }
                    }
                    Ok(())
                }
            };
        }
        TensorType::U8 => {
            let bytes = img.as_bytes();
            debug_assert_eq!(res.len(), bytes.len());
            return match data_layout {
                ImageDataLayout::NHWC => {
                    // just copy
                    res.copy_from_slice(bytes);
                    Ok(())
                }
                // batch is always 1 now
                ImageDataLayout::NCHW | ImageDataLayout::CHWN => {
                    let hw = (img.width() * img.height()) as usize;
                    for c in 0..3 {
                        let mut i = c as usize;
                        while i < hw {
                            res[res_index] = bytes[i];
                            res_index += 1;
                            i += c;
                        }
                    }
                    Ok(())
                }
            };
        }
        _ => unimplemented!(),
    }
}

mod ops_inner {
    use super::*;

    /// Rotate an image any radians clockwise.
    /// angle is in radians
    #[inline]
    pub fn rotate_any<I: GenericImageView>(
        image: &I,
        angle: f32,
    ) -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where
        I::Pixel: 'static,
    {
        let (width, height) = image.dimensions();

        let cos = angle.cos();
        let sin = angle.sin();
        let new_width = ((cos * width as f32).abs() + (sin * height as f32).abs()) as u32;
        let new_height = ((sin * width as f32).abs() + (cos * height as f32).abs()) as u32;

        let mut out = ImageBuffer::new(new_width, new_height);
        rotate_any_in(image, &mut out, angle);
        out
    }

    #[inline]
    fn rotate_any_in<I, Container>(
        image: &I,
        destination: &mut ImageBuffer<I::Pixel, Container>,
        angle: f32,
    ) where
        I: GenericImageView,
        I::Pixel: 'static,
        Container: std::ops::DerefMut<Target = [<I::Pixel as Pixel>::Subpixel]>,
    {
        let (dst_w, dst_h) = destination.dimensions();
        let (src_w, src_h) = image.dimensions();

        //  x_old = cos(angle) * (x - dst_w / 2) + sin(angle) * (y - dst_h / 2) + src_w / 2
        //  y_old = - sin(angle) * (x - dst_w / 2) + cos(angle) * (y - dst_h / 2) + src_h / 2

        let cos = angle.cos();
        let sin = angle.sin();
        let src_w_div2 = src_w as f32 / 2.;
        let src_h_div2 = src_h as f32 / 2.;
        let dst_w_div2 = dst_w as f32 / 2.;
        let dst_h_div2 = dst_h as f32 / 2.;
        for x in 0..dst_w {
            let add_x_old = cos * (x as f32 - dst_w_div2) + src_w_div2 + sin * (-dst_h_div2);
            let add_y_old = -sin * (x as f32 - dst_w_div2) + src_h_div2 + cos * (-dst_h_div2);
            for y in 0..dst_h {
                let x_old = add_x_old + sin * (y as f32);
                if x_old < 0. {
                    continue;
                }
                let x_old = x_old as u32;
                if x_old >= src_w {
                    continue;
                }

                let y_old = add_y_old + cos * (y as f32);
                if y_old < 0. {
                    continue;
                }
                let y_old = y_old as u32;
                if y_old >= src_h {
                    continue;
                }

                let pixel = image.get_pixel(x_old, y_old);
                destination.put_pixel(x, y, pixel);
            }
        }
    }
}
