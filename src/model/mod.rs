use std::collections::HashMap;

pub(crate) use memory_text_file::MemoryTextFile;
pub(crate) use zip::ZipFiles;

#[cfg(feature = "vision")]
use crate::postprocess::Activation;
use crate::postprocess::QuantizationParameters;
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

    #[cfg(feature = "vision")]
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

    #[cfg(feature = "vision")]
    fn output_activation(&self) -> Activation;

    // Result-returning accessors and checks shared by the task builders and sessions.
    fn expect_input_tensor_type(&self, index: usize) -> Result<TensorType, Error> {
        self.input_tensor_type(index)
            .ok_or_else(|| missing_info("input_tensor_type", index))
    }

    fn expect_output_tensor_type(&self, index: usize) -> Result<TensorType, Error> {
        self.output_tensor_type(index)
            .ok_or_else(|| missing_info("output_tensor_type", index))
    }

    fn expect_input_tensor_shape(&self, index: usize) -> Result<&[usize], Error> {
        self.input_tensor_shape(index)
            .ok_or_else(|| missing_info("input_tensor_shape", index))
    }

    fn expect_output_tensor_shape(&self, index: usize) -> Result<&[usize], Error> {
        self.output_tensor_shape(index)
            .ok_or_else(|| missing_info("output_tensor_shape", index))
    }

    #[cfg(feature = "vision")]
    fn expect_output_tensor_index(&self, name: &str) -> Result<usize, Error> {
        self.output_tensor_name_to_index(name)
            .ok_or_else(|| missing_info("output_tensor_name_to_index", name))
    }

    fn expect_to_tensor_info(&self, index: usize) -> Result<&ToTensorInfo, Error> {
        self.to_tensor_info(index)
            .ok_or_else(|| missing_info("to_tensor_info", index))
    }

    /// Output tensor type with its quantization parameters, which U8 tensors must have.
    fn output_type_and_quantization(
        &self,
        index: usize,
    ) -> Result<(TensorType, Option<QuantizationParameters>), Error> {
        let tensor_type = self.expect_output_tensor_type(index)?;
        let quantization = self.output_tensor_quantization_parameters(index);
        if tensor_type == TensorType::U8 && quantization.is_none() {
            return Err(Error::ModelInconsistentError(format!(
                "Missing tensor quantization parameters for output `{}`",
                index
            )));
        }
        Ok((tensor_type, quantization))
    }

    fn check_tensor_counts(
        &self,
        expect_inputs: Option<usize>,
        expect_outputs: usize,
    ) -> Result<(), Error> {
        if let Some(expect_inputs) = expect_inputs {
            let inputs = self.input_tensor_count();
            if inputs != expect_inputs {
                return Err(Error::ModelInconsistentError(format!(
                    "Expect model input tensor count `{}`, but got `{}`",
                    expect_inputs, inputs
                )));
            }
        }
        let outputs = self.output_tensor_count();
        if outputs != expect_outputs {
            return Err(Error::ModelInconsistentError(format!(
                "Expect model output tensor count `{}`, but got `{}`",
                expect_outputs, outputs
            )));
        }
        Ok(())
    }

    #[cfg(feature = "vision")]
    fn check_input_tensor_type(&self, index: usize, expect: TensorType) -> Result<(), Error> {
        check_tensor_type(
            "input",
            index,
            expect,
            self.expect_input_tensor_type(index)?,
        )
    }

    #[cfg(feature = "vision")]
    fn check_output_tensor_type(&self, index: usize, expect: TensorType) -> Result<(), Error> {
        check_tensor_type(
            "output",
            index,
            expect,
            self.expect_output_tensor_type(index)?,
        )
    }

    /// Check that output tensor `index` holds one `F32` value.
    #[cfg(feature = "vision")]
    fn check_scalar_f32_output(&self, index: usize) -> Result<(), Error> {
        self.check_output_tensor_type(index, TensorType::F32)?;
        let shape = self.expect_output_tensor_shape(index)?;
        if !shape.iter().all(|d| *d == 1) {
            return Err(Error::ModelInconsistentError(format!(
                "Expect output `{}` to hold one value, but got shape `{:?}`",
                index, shape
            )));
        }
        Ok(())
    }
}

pub(crate) fn parse_model(buf: &[u8]) -> Result<Box<dyn ModelResourceTrait + 'static>, Error> {
    if buf.len() < 8 {
        return Err(Error::ModelParseError(
            "Model buffer is too short!".to_string(),
        ));
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

fn missing_info(name: &str, index: impl std::fmt::Display) -> Error {
    Error::ModelInconsistentError(format!(
        "Model resource has no information for `{}` at index `{}`.",
        name, index
    ))
}

#[cfg(feature = "vision")]
fn check_tensor_type(
    kind: &str,
    index: usize,
    expect: TensorType,
    got: TensorType,
) -> Result<(), Error> {
    if got != expect {
        return Err(Error::ModelInconsistentError(format!(
            "Expect {} `{}` type is {:?}, but got `{:?}`",
            kind, index, expect, got
        )));
    }
    Ok(())
}

#[cfg(any(feature = "vision", feature = "audio"))]
pub(crate) fn tensor_byte_size(tensor_type: TensorType) -> usize {
    match tensor_type {
        TensorType::F32 | TensorType::I32 => 4,
        TensorType::U8 => 1,
        TensorType::F16 => 2,
    }
}

#[cfg(any(feature = "vision", feature = "audio"))]
pub(crate) fn tensor_bytes(tensor_type: TensorType, shape: &[usize]) -> usize {
    tensor_byte_size(tensor_type) * shape.iter().product::<usize>()
}

/// Find the first file in `zip_files` whose name is in `candidates`.
#[cfg(feature = "vision")]
pub(crate) fn search_file_in_zip<'buf>(
    zip_files: &ZipFiles<'buf>,
    candidates: &[&str],
    task_name: &str,
) -> Result<&'buf [u8], Error> {
    candidates
        .iter()
        .find_map(|name| zip_files.get_file(name))
        .ok_or_else(|| {
            Error::ModelInconsistentError(format!(
                "Cannot find model asset file for `{}` task, candidate list is `{:?}`",
                task_name, candidates
            ))
        })
}

mod memory_text_file;
mod tflite;
mod zip;
