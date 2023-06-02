mod common;

#[cfg(feature = "audio")]
pub mod audio;

#[cfg(feature = "text")]
pub mod text;

#[cfg(feature = "vision")]
pub mod vision;

use crate::Error;

#[derive(Debug)]
enum ToTensorInfoInner<'buf> {
    #[cfg(feature = "audio")]
    Audio(audio::AudioToTensorInfo),
    #[cfg(feature = "vision")]
    Image(vision::ImageToTensorInfo),
    #[cfg(feature = "text")]
    Text(text::TextToTensorInfo<'buf>),

    None(#[cfg(not(feature = "text"))] std::marker::PhantomData<&'buf ()>),
}

/// Necessary information for the media type to tensor. Such as Image to Tensors, Text to Tensors, Audio to Tensors, etc.
#[derive(Debug)]
pub struct ToTensorInfo<'buf> {
    inner: ToTensorInfoInner<'buf>,
}

impl<'buf> ToTensorInfo<'buf> {
    #[inline(always)]
    pub(crate) fn new_none() -> Self {
        #[cfg(not(feature = "text"))]
        return Self {
            inner: ToTensorInfoInner::None(Default::default()),
        };
        #[cfg(feature = "text")]
        return Self {
            inner: ToTensorInfoInner::None(),
        };
    }

    #[cfg(feature = "audio")]
    #[inline(always)]
    pub(crate) fn new_audio(audio_to_tensor_info: audio::AudioToTensorInfo) -> Self {
        Self {
            inner: ToTensorInfoInner::Audio(audio_to_tensor_info),
        }
    }

    #[cfg(feature = "vision")]
    #[inline(always)]
    pub(crate) fn new_image(image_to_tensor_info: vision::ImageToTensorInfo) -> Self {
        Self {
            inner: ToTensorInfoInner::Image(image_to_tensor_info),
        }
    }

    #[cfg(feature = "text")]
    #[inline(always)]
    pub(crate) fn new_text(text_to_tensor_info: text::TextToTensorInfo<'buf>) -> Self {
        Self {
            inner: ToTensorInfoInner::Text(text_to_tensor_info),
        }
    }

    /// Try convert to [`audio::AudioToTensorInfo`], if the model has no audio preprocess information, will return an error.
    #[cfg(feature = "audio")]
    #[inline(always)]
    pub fn try_to_audio(&self) -> Result<&audio::AudioToTensorInfo, Error> {
        match &self.inner {
            ToTensorInfoInner::Audio(a) => Ok(a),
            _ => {
                return Err(Error::ModelInconsistentError(format!(
                    "Expect Audio to Tensor Info, but got `{:?}`",
                    self.inner
                )));
            }
        }
    }

    /// Try convert to [`vision::ImageToTensorInfo`], if the model has no image preprocess information, will return an error.
    #[cfg(feature = "vision")]
    #[inline(always)]
    pub fn try_to_image(&self) -> Result<&vision::ImageToTensorInfo, Error> {
        match &self.inner {
            ToTensorInfoInner::Image(i) => Ok(i),
            _ => {
                return Err(Error::ModelInconsistentError(format!(
                    "Expect Image to Tensor Info, but got `{:?}`",
                    self.inner
                )));
            }
        }
    }

    /// Try convert to [`text::TextToTensorInfo`], if the model has no text preprocess information, will return an error.
    #[cfg(feature = "text")]
    #[inline(always)]
    pub fn try_to_text(&self) -> Result<&text::TextToTensorInfo<'buf>, Error> {
        match &self.inner {
            ToTensorInfoInner::Text(t) => Ok(t),
            _ => {
                return Err(Error::ModelInconsistentError(format!(
                    "Expect Text to Tensor Info, but got `{:?}`",
                    self.inner
                )));
            }
        }
    }
}
