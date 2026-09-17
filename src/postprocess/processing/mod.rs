#![allow(unused)]

use crate::postprocess::ops::*;
use crate::{Error, GraphExecutionContext, TensorType};

/// Copy output tensor `index` from `ctx` into `buf` and check that it fills `buf` exactly.
pub(crate) fn fetch_output<T>(
    ctx: &GraphExecutionContext,
    index: usize,
    buf: &mut [T],
) -> Result<(), Error> {
    let expected = std::mem::size_of_val(buf);
    let written = ctx.get_output(index, buf)?;
    if written != expected {
        return Err(Error::ModelInconsistentError(format!(
            "Model output `{}` bytes size is `{}`, but got `{}`",
            index, expected, written
        )));
    }
    Ok(())
}

enum OutputStorage {
    F32(Vec<f32>),
    U8 {
        bytes: Vec<u8>,
        quantization: QuantizationParameters,
        dequantized: Vec<f32>,
    },
}

/// Receives one output tensor from wasi-nn and exposes it as `f32` values.
pub(crate) struct OutputBuffer {
    storage: OutputStorage,
}

impl OutputBuffer {
    pub(crate) fn new(
        (tensor_type, quantization): (TensorType, Option<QuantizationParameters>),
        elem_count: usize,
    ) -> Result<Self, Error> {
        let storage = match (tensor_type, quantization) {
            (TensorType::F32, _) => OutputStorage::F32(vec![0.; elem_count]),
            (TensorType::U8, Some(quantization)) => OutputStorage::U8 {
                bytes: vec![0; elem_count],
                quantization,
                dequantized: vec![0.; elem_count],
            },
            (TensorType::U8, None) => {
                return Err(Error::ModelInconsistentError(
                    "Missing quantization parameters for U8 output tensor".into(),
                ));
            }
            (t, _) => {
                return Err(Error::ModelInconsistentError(format!(
                    "Unsupported output tensor type `{:?}`, expect F32 or U8",
                    t
                )));
            }
        };
        Ok(Self { storage })
    }

    /// Set the number of elements expected from the next `fetch`.
    pub(crate) fn resize(&mut self, elem_count: usize) {
        match &mut self.storage {
            OutputStorage::F32(v) => v.resize(elem_count, 0.),
            OutputStorage::U8 {
                bytes, dequantized, ..
            } => {
                bytes.resize(elem_count, 0);
                dequantized.resize(elem_count, 0.);
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match &self.storage {
            OutputStorage::F32(v) => v.len(),
            OutputStorage::U8 { bytes, .. } => bytes.len(),
        }
    }

    /// Copy output tensor `index` from `ctx` and check that it fills this buffer exactly.
    pub(crate) fn fetch(&mut self, ctx: &GraphExecutionContext, index: usize) -> Result<(), Error> {
        match &mut self.storage {
            OutputStorage::F32(v) => fetch_output(ctx, index, v.as_mut_slice()),
            OutputStorage::U8 { bytes, .. } => fetch_output(ctx, index, bytes.as_mut_slice()),
        }
    }

    /// Output values as `f32`. U8 tensors are dequantized on every call.
    pub(crate) fn as_f32_mut(&mut self) -> &mut [f32] {
        match &mut self.storage {
            OutputStorage::F32(v) => v.as_mut_slice(),
            OutputStorage::U8 {
                bytes,
                quantization,
                dequantized,
            } => {
                bytes
                    .as_slice()
                    .dequantize_to_buf(*quantization, dequantized);
                dequantized.as_mut_slice()
            }
        }
    }
}

mod common;
pub(crate) use common::*;

#[cfg(feature = "vision")]
mod vision;
#[cfg(feature = "vision")]
pub(crate) use vision::*;
