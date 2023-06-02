#![allow(unused)]

use crate::postprocess::ops::*;
use crate::TensorType;

struct OutputBuffer {
    data_buffer: Vec<u8>,
    tensor_type: TensorType,
    quantization_parameters: Option<(QuantizationParameters, Vec<f32>)>,
}

macro_rules! output_buffer_mut_slice {
    ( $out:expr ) => {
        match $out.tensor_type {
            TensorType::U8 => {
                let (q, f) = $out.quantization_parameters.as_mut().unwrap();
                $out.data_buffer.as_slice().dequantize_to_buf(*q, f);
                f.as_mut_slice()
            }
            TensorType::F32 => unsafe {
                core::slice::from_raw_parts_mut(
                    $out.data_buffer.as_mut_slice().as_ptr() as *mut f32,
                    $out.data_buffer.len() >> 2,
                )
            },
            _ => {
                todo!("FP16, I32")
            }
        }
    };
}

macro_rules! empty_output_buffer {
    ( $x:ident ) => {
        match $x.1 {
            Some(q) => OutputBuffer {
                data_buffer: vec![],
                tensor_type: $x.0,
                quantization_parameters: Some((q, vec![])),
            },
            None => OutputBuffer {
                data_buffer: vec![],
                tensor_type: $x.0,
                quantization_parameters: None,
            },
        }
    };

    ( $x:ident, $elem_size:expr ) => {{
        let bytes_size = tensor_byte_size!($x.0) * $elem_size;
        match $x.1 {
            Some(q) => OutputBuffer {
                data_buffer: vec![0; bytes_size],
                tensor_type: $x.0,
                quantization_parameters: Some((q, vec![0f32; $elem_size])),
            },
            None => OutputBuffer {
                data_buffer: vec![0; bytes_size],
                tensor_type: $x.0,
                quantization_parameters: None,
            },
        }
    }};
}

macro_rules! realloc_output_buffer {
    ( $self:expr, $new_size:expr ) => {
        let new_size = $new_size;
        if let Some(ref mut t) = $self.quantization_parameters {
            if t.1.len() < new_size {
                t.1.resize(new_size, 0f32);
            }
        }
        let s = tensor_byte_size!($self.tensor_type) * new_size;
        if $self.data_buffer.len() < s {
            $self.data_buffer.resize(s, 0);
        }
    };
}

mod common;
pub(crate) use common::*;

#[cfg(feature = "vision")]
mod vision;
#[cfg(feature = "vision")]
pub(crate) use vision::*;
