#![allow(unused)]

use std::collections::HashMap;

pub(crate) use memory_text_file::MemoryTextFile;
pub(crate) use zip::ZipFiles;

use crate::postprocess::{Activation, QuantizationParameters};
#[cfg(feature = "audio")]
use crate::preprocess::audio::AudioToTensorInfo;
#[cfg(feature = "text")]
use crate::preprocess::text::TextToTensorInfo;
#[cfg(feature = "vision")]
use crate::preprocess::vision::{ImageColorSpaceType, ImageDataLayout, ImageToTensorInfo};
use crate::preprocess::ToTensorInfo;
use crate::{Error, GraphEncoding, TensorType};

/// Abstraction for model resources.
/// Users can use this trait to get information for models, such as data layout, model backend, etc.
/// Now it supports ```TensorFlowLite``` backend.
pub(crate) trait ModelResourceTrait {
    fn model_backend(&self) -> GraphEncoding;

    fn input_tensor_count(&self) -> usize;

    fn output_tensor_count(&self) -> usize;

    fn input_tensor_type(&self, index: usize) -> Option<TensorType>;

    fn output_tensor_type(&self, index: usize) -> Option<TensorType>;

    fn input_tensor_shape(&self, index: usize) -> Option<&[usize]>;

    fn output_tensor_shape(&self, index: usize) -> Option<&[usize]>;

    fn output_tensor_name_to_index(&self, name: &str) -> Option<usize>;

    fn output_tensor_quantization_parameters(&self, index: usize)
        -> Option<QuantizationParameters>;

    fn output_tensor_labels_locale(
        &self,
        index: usize,
        locale: &str,
    ) -> Result<(&[u8], Option<&[u8]>), Error>;

    #[cfg(feature = "vision")]
    fn output_bounding_box_properties(&self, index: usize, slice: &mut [usize]) -> bool;

    fn to_tensor_info(&self, input_index: usize) -> Option<&ToTensorInfo>;

    fn output_activation(&self) -> Activation;
}

#[inline]
pub(crate) fn parse_model(buf: &[u8]) -> Result<Box<dyn ModelResourceTrait + 'static>, Error> {
    if buf.len() < 8 {
        return Err(Error::ModelParseError(format!(
            "Model buffer is tool short!"
        )));
    }

    match &buf[4..8] {
        tflite::TfLiteModelResource::HEAD_MAGIC => {
            let tf_model_resource = tflite::TfLiteModelResource::new(buf)?;
            Ok(Box::new(tf_model_resource))
        }
        _ => Err(Error::ModelParseError(format!(
            "Cannot parse this head magic `{:?}`",
            &buf[..8]
        ))),
    }
}

macro_rules! model_resource_check_and_get_impl {
    ( $model_resource:expr, $func_name:ident, $index:expr ) => {
        $model_resource
            .$func_name($index)
            .ok_or(crate::Error::ModelInconsistentError(format!(
                "Model resource has no information for `{}` at index `{}`.",
                stringify!($func_name),
                $index
            )))?
    };
}

macro_rules! tensor_byte_size {
    ($tensor_type:expr) => {
        match $tensor_type {
            crate::TensorType::F32 => 4,
            crate::TensorType::U8 => 1,
            crate::TensorType::I32 => 4,
            crate::TensorType::F16 => 2,
        }
    };
}

macro_rules! tensor_bytes {
    ( $tensor_type:expr, $tensor_shape:ident ) => {{
        let mut b = tensor_byte_size!($tensor_type);
        for s in $tensor_shape {
            b *= s;
        }
        b
    }};
}

macro_rules! check_quantization_parameters {
    ( $tensor_type:ident, $q:ident, $i:expr ) => {
        if $tensor_type == crate::TensorType::U8 && $q.is_none() {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Missing tensor quantization parameters for output `{}`",
                $i
            )));
        }
    };
}

macro_rules! get_type_and_quantization {
    ( $model_resource:expr, $index:expr ) => {{
        let t = model_resource_check_and_get_impl!($model_resource, output_tensor_type, $index);
        let q = $model_resource.output_tensor_quantization_parameters($index);
        check_quantization_parameters!(t, q, $index);
        (t, q)
    }};
}

macro_rules! search_file_in_zip {
    ( $zip_files:expr, $buf:expr, $candidate_list:expr, $task_name:expr ) => {{
        let mut search_result = None;
        for name in $candidate_list {
            if let Some(r) = $zip_files.get_file_offset(*name) {
                search_result = Some(r);
                break;
            }
        }
        if let Some(r) = search_result {
            &$buf[r]
        } else {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Cannot find model asset file for `{}` task, candidate list is `{:?}`",
                $task_name, $candidate_list
            )));
        }
    }};
}

macro_rules! check_tensor_type {
    ( $model_resource:ident, $index:expr, $func:ident, $tensor_type:expr ) => {{
        let tensor_type = model_resource_check_and_get_impl!($model_resource, $func, $index);
        if tensor_type != $tensor_type {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect {} `{}` type is {:?}, but got `{:?}`",
                stringify!($func).split("_").next().unwrap(),
                $index,
                $tensor_type,
                tensor_type
            )));
        }
    }};
}

mod memory_text_file;
mod tflite;
mod zip;
