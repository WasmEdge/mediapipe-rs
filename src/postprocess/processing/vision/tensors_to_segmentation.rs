// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/calculators/tflite/tflite_tensors_to_segmentation_calculator.cc

use super::*;
use crate::postprocess::{Activation, ImageCategoryMask, ImageConfidenceMask};
use crate::preprocess::vision::{ImageDataLayout, ImageLikeTensorShape};

/// Converts a segmentation output tensor to masks. The tensor is interpreted as
/// `[1, height, width, channels]` or `[height, width, channels]`, as MediaPipe does; TFLite
/// metadata does not describe the output layout.
pub(crate) struct TensorsToSegmentation {
    activation: Activation,
    tensor_buffer: OutputBuffer,
    tensor_shape: ImageLikeTensorShape,
}

impl TensorsToSegmentation {
    #[inline(always)]
    pub(crate) fn new(
        activation: Activation,
        tensor_buf_info: (TensorType, Option<QuantizationParameters>),
        tensor_shape: &[usize],
    ) -> Result<Self, crate::Error> {
        let tensor_shape = ImageLikeTensorShape::parse(ImageDataLayout::NHWC, tensor_shape)?;
        if tensor_shape.batch != 1 {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Unsupported batch size `{}`, now only support batch size = 1",
                tensor_shape.batch
            )));
        }

        let elem_size = tensor_shape.elem_size();
        Ok(Self {
            activation,
            tensor_buffer: OutputBuffer::new(tensor_buf_info, elem_size)?,
            tensor_shape,
        })
    }

    #[inline(always)]
    pub(crate) fn tensor_buffer(&mut self) -> &mut OutputBuffer {
        &mut self.tensor_buffer
    }

    pub(crate) fn category_mask(&mut self) -> ImageCategoryMask {
        let tensor = self.tensor_buffer.as_f32_mut();
        let mut res = ImageCategoryMask::new(
            self.tensor_shape.width as u32,
            self.tensor_shape.height as u32,
        );
        let channels = self.tensor_shape.channels;
        for (p, scores) in res.pixels_mut().zip(tensor.chunks_exact(channels)) {
            p.0[0] = if channels == 1 {
                (scores[0] > 0.5) as u8
            } else {
                let mut max_v = scores[0];
                let mut max_c = 0;
                for (c, &v) in scores.iter().enumerate().skip(1) {
                    if v > max_v {
                        max_v = v;
                        max_c = c;
                    }
                }
                max_c as u8
            };
        }
        res
    }

    pub(crate) fn confidence_masks(&mut self) -> Vec<ImageConfidenceMask> {
        let tensor = self.tensor_buffer.as_f32_mut();
        let channels = self.tensor_shape.channels;

        // apply activation
        match self.activation {
            Activation::None => { /* do nothing */ }
            Activation::SIGMOID => tensor.sigmoid_inplace(),
            Activation::SOFTMAX => {
                if channels > 1 {
                    for scores in tensor.chunks_exact_mut(channels) {
                        scores.softmax_inplace();
                    }
                }
            }
        };

        let mut res = Vec::with_capacity(channels);
        for c in 0..channels {
            res.push(ImageConfidenceMask::new(
                self.tensor_shape.width as u32,
                self.tensor_shape.height as u32,
            ));
        }
        let mut pixels = res.iter_mut().map(|c| c.pixels_mut()).collect::<Vec<_>>();
        for scores in tensor.chunks_exact(channels) {
            for (mask, &score) in pixels.iter_mut().zip(scores) {
                if let Some(p) = mask.next() {
                    p.0[0] = score;
                }
            }
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn segmentation(activation: Activation, channels: usize) -> TensorsToSegmentation {
        TensorsToSegmentation::new(activation, (TensorType::F32, None), &[1, 1, 2, channels])
            .unwrap()
    }

    #[test]
    fn test_rejects_batched_output() {
        assert!(TensorsToSegmentation::new(
            Activation::None,
            (TensorType::F32, None),
            &[2, 1, 2, 3]
        )
        .is_err());
    }

    #[test]
    fn test_category_mask_argmax_and_threshold() {
        let mut s = segmentation(Activation::None, 3);
        s.tensor_buffer
            .as_f32_mut()
            .copy_from_slice(&[0.1, 0.7, 0.2, 0.5, 0.1, 0.4]);
        assert_eq!(s.category_mask().into_raw(), vec![1, 0]);

        let mut s = segmentation(Activation::None, 1);
        s.tensor_buffer.as_f32_mut().copy_from_slice(&[0.9, 0.2]);
        assert_eq!(s.category_mask().into_raw(), vec![1, 0]);
    }

    #[test]
    fn test_confidence_masks_softmax_per_pixel() {
        let mut s = segmentation(Activation::SOFTMAX, 2);
        s.tensor_buffer
            .as_f32_mut()
            .copy_from_slice(&[0., 0., 1., 1.]);
        let masks = s.confidence_masks();
        assert_eq!(masks.len(), 2);
        assert_eq!(masks[0].as_raw(), &[0.5, 0.5]);
        assert_eq!(masks[1].as_raw(), &[0.5, 0.5]);
    }
}
