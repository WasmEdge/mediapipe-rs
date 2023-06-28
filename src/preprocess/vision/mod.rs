mod image;

#[cfg(feature = "ffmpeg")]
mod ffmpeg;
#[cfg(feature = "ffmpeg")]
pub use ffmpeg::FFMpegVideoData;

use super::*;
use crate::tasks::vision::ImageProcessingOptions;
use crate::TensorType;

/// Every type implement the [`ImageToTensor`] trait can be used as vision tasks input.
/// Now the builtin impl: image crate images.
pub trait ImageToTensor {
    /// convert image to tensors, save to output_buffers
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        to_tensor_info: &ImageToTensorInfo,
        process_options: &ImageProcessingOptions,
        output_buffers: &mut T,
    ) -> Result<(), Error>;

    /// return image size: (weight, height)
    fn image_size(&self) -> (u32, u32);

    /// return the current timestamp (ms)
    /// video frame must return a valid timestamp
    fn timestamp_ms(&self) -> Option<u64> {
        return None;
    }
}

/// Used for video data. Every video data implement the [`VideoData`] can be used as vision tasks input.
/// Now builtin impl: [`FFMpegVideoData`].
///
/// Now rust stable cannot use [Generic Associated Types](https://rust-lang.github.io/rfcs/1598-generic_associated_types.html)
pub trait VideoData {
    type Frame<'frame>: ImageToTensor
    where
        Self: 'frame;

    fn next_frame(&mut self) -> Result<Option<Self::Frame<'_>>, Error>;
}

/// Data layout in memory for image tensor. ```NCHW```, ```NHWC```, ```CHWN```.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum ImageDataLayout {
    NCHW,
    NHWC,
    CHWN,
}

/// Image Color Type.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum ImageColorSpaceType {
    RGB,
    GRAYSCALE,
    UNKNOWN,
}

/// Necessary information for the image to tensor.
#[derive(Debug)]
pub struct ImageToTensorInfo {
    pub image_data_layout: ImageDataLayout,
    pub color_space: ImageColorSpaceType,
    pub tensor_type: TensorType,
    pub tensor_shape: ImageLikeTensorShape,
    pub stats_min: Vec<f32>,
    pub stats_max: Vec<f32>,
    /// (mean,std), len can be 1 or 3
    pub normalization_options: (Vec<f32>, Vec<f32>),
}

/// The tensor shape which contains image information.
#[derive(Debug, Copy, Clone)]
pub struct ImageLikeTensorShape {
    pub batch: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

impl ImageToTensorInfo {
    #[inline(always)]
    pub fn width(&self) -> u32 {
        self.tensor_shape.width as u32
    }

    #[inline(always)]
    pub fn height(&self) -> u32 {
        self.tensor_shape.height as u32
    }
}

impl ImageLikeTensorShape {
    /// Parse a tensor shape for given image data layout.
    pub fn parse(data_layout: ImageDataLayout, shape: &[usize]) -> Result<Self, Error> {
        match shape.len() {
            2 => Ok(Self {
                batch: 1,
                width: shape[1],
                height: shape[0],
                channels: 1,
            }),
            3 => match data_layout {
                ImageDataLayout::NCHW | ImageDataLayout::CHWN => Ok(Self {
                    batch: 1,
                    width: shape[2],
                    height: shape[1],
                    channels: shape[0],
                }),
                ImageDataLayout::NHWC => Ok(Self {
                    batch: 1,
                    width: shape[1],
                    height: shape[0],
                    channels: shape[2],
                }),
            },
            4 => match data_layout {
                ImageDataLayout::NCHW => Ok(Self {
                    batch: shape[0],
                    width: shape[3],
                    height: shape[2],
                    channels: shape[1],
                }),
                ImageDataLayout::NHWC => Ok(Self {
                    batch: shape[0],
                    width: shape[2],
                    height: shape[1],
                    channels: shape[3],
                }),
                ImageDataLayout::CHWN => Ok(Self {
                    batch: shape[3],
                    width: shape[2],
                    height: shape[1],
                    channels: shape[0],
                }),
            },
            _ => Err(Error::ArgumentError(format!(
                "Expect shape len is 2, 3 or 4, but got shape `{:?}`",
                shape
            ))),
        }
    }

    /// Get the number of tensor elements.
    #[inline(always)]
    pub fn elem_size(&self) -> usize {
        self.batch * self.width * self.height * self.channels
    }
}
