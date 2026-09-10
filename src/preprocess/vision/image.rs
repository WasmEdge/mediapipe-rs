extern crate image as image_crate;

use super::*;
pub(super) use image_crate::{
    imageops, DynamicImage, EncodableLayout, GenericImageView, ImageBuffer, Pixel, Rgb, RgbImage,
};

const IMAGE_RESIZE_FILTER: imageops::FilterType = imageops::FilterType::Gaussian;

/// Returns per-channel (mean, std) for RGB. Channels absent from the metadata reuse the first value.
fn rgb_mean_std(info: &ImageToTensorInfo) -> Result<([f32; 3], [f32; 3]), Error> {
    let (mean, std) = (&info.normalization_options.0, &info.normalization_options.1);
    let (Some(&mean0), Some(&std0)) = (mean.first(), std.first()) else {
        return Err(Error::ModelInconsistentError(
            "Float32 image input requires NormalizationOptions (mean/std) in model metadata".into(),
        ));
    };
    let pick = |v: &[f32], i: usize, default: f32| v.get(i).copied().unwrap_or(default);
    Ok((
        [mean0, pick(mean, 1, mean0), pick(mean, 2, mean0)],
        [std0, pick(std, 1, std0), pick(std, 2, std0)],
    ))
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
    let shape = &info.tensor_shape;
    if shape.batch != 1
        || shape.channels != 3
        || img.width() != info.width()
        || img.height() != info.height()
    {
        return Err(Error::ModelInconsistentError(format!(
            "Expect a `1x{}x{}x3` RGB image tensor, but got batch `{}`, `{}x{}`, channels `{}`",
            img.height(),
            img.width(),
            shape.batch,
            shape.height,
            shape.width,
            shape.channels
        )));
    }

    let data_layout = info.image_data_layout;
    let res = output_buffer.as_mut();
    let bytes = img.as_bytes();
    let hw = (img.width() * img.height()) as usize;
    let expected_len = shape.elem_size() * tensor_byte_size!(info.tensor_type);
    if res.len() < expected_len {
        return Err(Error::ArgumentError(format!(
            "Expect output buffer at least `{}` bytes, but got `{}`",
            expected_len,
            res.len()
        )));
    }
    match info.tensor_type {
        TensorType::F32 => {
            let (means, stds) = rgb_mean_std(info)?;
            let mut out = res.chunks_exact_mut(std::mem::size_of::<f32>());
            let mut put = |value: u8, c: usize| {
                let f = (value as f32 - means[c]) / stds[c];
                out.next().unwrap().copy_from_slice(&f.to_ne_bytes());
            };
            match data_layout {
                ImageDataLayout::NHWC => {
                    for px in bytes.chunks_exact(3) {
                        put(px[0], 0);
                        put(px[1], 1);
                        put(px[2], 2);
                    }
                }
                // batch is always 1 now
                ImageDataLayout::NCHW | ImageDataLayout::CHWN => {
                    for c in 0..3 {
                        for p in 0..hw {
                            put(bytes[p * 3 + c], c);
                        }
                    }
                }
            }
            Ok(())
        }
        TensorType::U8 => {
            match data_layout {
                ImageDataLayout::NHWC => res[..bytes.len()].copy_from_slice(bytes),
                // batch is always 1 now
                ImageDataLayout::NCHW | ImageDataLayout::CHWN => {
                    let mut out = res.iter_mut();
                    for c in 0..3 {
                        for p in 0..hw {
                            *out.next().unwrap() = bytes[p * 3 + c];
                        }
                    }
                }
            }
            Ok(())
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

#[cfg(test)]
mod test {
    use super::*;

    fn info(
        layout: ImageDataLayout,
        tensor_type: TensorType,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> ImageToTensorInfo {
        info_with_shape(layout, tensor_type, mean, std, (1, 2, 1, 3))
    }

    fn info_with_shape(
        layout: ImageDataLayout,
        tensor_type: TensorType,
        mean: Vec<f32>,
        std: Vec<f32>,
        (batch, width, height, channels): (usize, usize, usize, usize),
    ) -> ImageToTensorInfo {
        ImageToTensorInfo {
            image_data_layout: layout,
            color_space: ImageColorSpaceType::RGB,
            tensor_type,
            tensor_shape: ImageLikeTensorShape {
                batch,
                width,
                height,
                channels,
            },
            stats_min: vec![],
            stats_max: vec![],
            normalization_options: (mean, std),
        }
    }

    fn to_f32(buf: &[u8]) -> Vec<f32> {
        buf.chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_rgb8_to_f32_per_channel_normalization() {
        let img = RgbImage::from_raw(2, 1, vec![10, 20, 30, 40, 50, 60]).unwrap();
        let mut buf = vec![0u8; 6 * 4];

        let nhwc = info(
            ImageDataLayout::NHWC,
            TensorType::F32,
            vec![0., 1., 2.],
            vec![1., 2., 4.],
        );
        rgb8_image_buffer_to_tensor(&img, &nhwc, &mut buf).unwrap();
        assert_eq!(to_f32(&buf), [10., 9.5, 7., 40., 24.5, 14.5]);

        let nchw = info(
            ImageDataLayout::NCHW,
            TensorType::F32,
            vec![0., 1., 2.],
            vec![1., 2., 4.],
        );
        rgb8_image_buffer_to_tensor(&img, &nchw, &mut buf).unwrap();
        assert_eq!(to_f32(&buf), [10., 40., 9.5, 24.5, 7., 14.5]);

        let single = info(ImageDataLayout::NHWC, TensorType::F32, vec![10.], vec![10.]);
        rgb8_image_buffer_to_tensor(&img, &single, &mut buf).unwrap();
        assert_eq!(to_f32(&buf), [0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn test_rgb8_to_f32_requires_normalization_options() {
        let img = RgbImage::from_raw(2, 1, vec![0; 6]).unwrap();
        let mut buf = vec![0u8; 6 * 4];
        let missing = info(ImageDataLayout::NHWC, TensorType::F32, vec![], vec![]);
        assert!(rgb8_image_buffer_to_tensor(&img, &missing, &mut buf).is_err());
        let mut short = vec![0u8; 6 * 4 - 1];
        let ok = info(ImageDataLayout::NHWC, TensorType::F32, vec![0.], vec![1.]);
        assert!(rgb8_image_buffer_to_tensor(&img, &ok, &mut short).is_err());
    }

    #[test]
    fn test_rgb8_rejects_unsupported_tensor_shapes() {
        let img = RgbImage::from_raw(2, 1, vec![0; 6]).unwrap();
        let mut buf = vec![0u8; 2 * 6 * 4];
        for shape in [(2, 2, 1, 3), (1, 2, 1, 1), (1, 2, 1, 4), (1, 1, 2, 3)] {
            let info = info_with_shape(
                ImageDataLayout::NHWC,
                TensorType::F32,
                vec![0.],
                vec![1.],
                shape,
            );
            assert!(
                rgb8_image_buffer_to_tensor(&img, &info, &mut buf).is_err(),
                "{:?}",
                shape
            );
        }
    }

    #[test]
    fn test_rgb8_to_u8_layouts() {
        let img = RgbImage::from_raw(2, 1, vec![10, 20, 30, 40, 50, 60]).unwrap();
        let mut buf = vec![0u8; 6];

        let nhwc = info(ImageDataLayout::NHWC, TensorType::U8, vec![], vec![]);
        rgb8_image_buffer_to_tensor(&img, &nhwc, &mut buf).unwrap();
        assert_eq!(buf, [10, 20, 30, 40, 50, 60]);

        let nchw = info(ImageDataLayout::NCHW, TensorType::U8, vec![], vec![]);
        rgb8_image_buffer_to_tensor(&img, &nchw, &mut buf).unwrap();
        assert_eq!(buf, [10, 40, 20, 50, 30, 60]);
    }
}
