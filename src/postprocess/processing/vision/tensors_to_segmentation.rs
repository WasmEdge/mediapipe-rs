// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/calculators/tflite/tflite_tensors_to_segmentation_calculator.cc

use super::*;
use crate::postprocess::{Activation, ImageCategoryMask, ImageConfidenceMask};
use crate::preprocess::vision::{ImageDataLayout, ImageLikeTensorShape};

pub(crate) struct TensorsToSegmentation {
    activation: Activation,
    tensor_buffer: OutputBuffer,
    image_data_layout: ImageDataLayout,
    tensor_shape: ImageLikeTensorShape,
}

impl TensorsToSegmentation {
    #[inline(always)]
    pub(crate) fn new(
        activation: Activation,
        tensor_buf_info: (TensorType, Option<QuantizationParameters>),
        image_data_layout: ImageDataLayout,
        tensor_shape: &[usize],
    ) -> Result<Self, crate::Error> {
        let tensor_shape = ImageLikeTensorShape::parse(image_data_layout, tensor_shape)?;
        if tensor_shape.batch != 1 {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Unsupported batch size `{}`, now only support batch size = 1",
                tensor_shape.batch
            )));
        }

        let elem_size = tensor_shape.elem_size();
        Ok(Self {
            activation,
            tensor_buffer: empty_output_buffer!(tensor_buf_info, elem_size),
            image_data_layout,
            tensor_shape,
        })
    }

    #[inline(always)]
    pub(crate) fn tenor_buffer(&mut self) -> &mut [u8] {
        self.tensor_buffer.data_buffer.as_mut_slice()
    }

    pub(crate) fn category_mask(&mut self) -> ImageCategoryMask {
        let tensor = output_buffer_mut_slice!(self.tensor_buffer);
        let mut res = ImageCategoryMask::new(
            self.tensor_shape.width as u32,
            self.tensor_shape.height as u32,
        );
        let channels = self.tensor_shape.channels;
        let mut index = 0;
        match self.image_data_layout {
            ImageDataLayout::NHWC => {
                for p in res.pixels_mut() {
                    if channels == 1 {
                        p.0[0] = if tensor[index] > 0.5 { 1 } else { 0 };
                    } else {
                        let mut max_v = tensor[index];
                        let mut max_c = 0;
                        for c in 1..channels {
                            if tensor[index + c] > max_v {
                                max_v = tensor[index + c];
                                max_c = c;
                            }
                        }
                        p.0[0] = max_c as u8;
                    }

                    index += channels;
                }
            }
            ImageDataLayout::NCHW => {
                unimplemented!()
            }
            ImageDataLayout::CHWN => {
                unimplemented!()
            }
        }
        res
    }

    pub(crate) fn confidence_masks(&mut self) -> Vec<ImageConfidenceMask> {
        let tensor = output_buffer_mut_slice!(self.tensor_buffer);
        let channels = self.tensor_shape.channels;

        // apply activation
        match self.activation {
            Activation::None => { /* do nothing */ }
            Activation::SIGMOID => tensor.sigmoid_inplace(),
            Activation::SOFTMAX => match self.image_data_layout {
                ImageDataLayout::NHWC => {
                    if channels > 1 {
                        let mut index = 0;
                        while index < tensor.len() {
                            tensor[index..index + channels].softmax_inplace();
                            index += channels;
                        }
                    }
                }
                ImageDataLayout::NCHW => {
                    unimplemented!()
                }
                ImageDataLayout::CHWN => {
                    unimplemented!()
                }
            },
        };

        let mut res = Vec::with_capacity(channels);
        for c in 0..channels {
            res.push(ImageConfidenceMask::new(
                self.tensor_shape.width as u32,
                self.tensor_shape.height as u32,
            ));
        }
        let mut pixels = res.iter_mut().map(|c| c.pixels_mut()).collect::<Vec<_>>();
        match self.image_data_layout {
            ImageDataLayout::NHWC => {
                let mut index = 0;
                let len = pixels[0].len();
                for i in 0..len {
                    for c in 0..channels {
                        pixels[c].next().unwrap().0[0] = tensor[index];
                        index += 1;
                    }
                }
            }
            ImageDataLayout::NCHW => {
                unimplemented!()
            }
            ImageDataLayout::CHWN => {
                unimplemented!()
            }
        }
        res
    }
}
