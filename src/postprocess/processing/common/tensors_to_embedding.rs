// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/tasks/cc/components/calculators/tensors_to_embeddings_calculator.cc

use super::*;
use crate::postprocess::{Embedding, EmbeddingResult};

pub struct TensorsToEmbedding {
    quantize: bool,
    l2_normalize: bool,

    outputs: Vec<OutputBuffer>,
    head_names: Vec<Option<String>>,
}

impl TensorsToEmbedding {
    #[inline(always)]
    pub(crate) fn new(quantize: bool, l2_normalize: bool) -> Self {
        Self {
            quantize,
            l2_normalize,
            outputs: Vec::new(),
            head_names: Vec::new(),
        }
    }

    #[inline(always)]
    pub(crate) fn add_output_cfg(
        &mut self,
        tensor_buf: (TensorType, Option<QuantizationParameters>),
        tensor_shape: &[usize],
        head_name: Option<String>,
    ) -> Result<(), crate::Error> {
        let elem_size = tensor_shape.iter().fold(1, |a, b| a * b);
        self.outputs.push(OutputBuffer::new(tensor_buf, elem_size)?);
        self.head_names.push(head_name);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn output_buffer(&mut self, index: usize) -> &mut OutputBuffer {
        &mut self.outputs[index]
    }

    pub(crate) fn result(&mut self, timestamp_ms: Option<u64>) -> EmbeddingResult {
        let embeddings_count = self.outputs.len();
        let mut embeddings = Vec::with_capacity(embeddings_count);

        for id in 0..embeddings_count {
            let tensor = self.outputs[id].as_f32_mut();

            let mut float_embedding;
            let mut quantized_embedding;
            if self.quantize {
                float_embedding = Vec::new();
                quantized_embedding = Vec::with_capacity(tensor.len());
                let scale = if self.l2_normalize {
                    Self::get_inverse_l2_norm(tensor)
                } else {
                    1.0
                };
                for t in tensor {
                    quantized_embedding.push(Self::quantize_value((*t) * scale));
                }
            } else {
                quantized_embedding = Vec::new();
                if self.l2_normalize {
                    float_embedding = Vec::with_capacity(tensor.len());
                    let inv_l2_norm = Self::get_inverse_l2_norm(tensor);
                    for t in tensor {
                        float_embedding.push((*t) * inv_l2_norm);
                    }
                } else {
                    float_embedding = Vec::from(tensor);
                }
            }

            embeddings.push(Embedding {
                head_index: id,
                head_name: self.head_names[id].clone(),
                float_embedding,
                quantized_embedding,
            })
        }

        EmbeddingResult {
            embeddings,
            timestamp_ms,
        }
    }

    /// Scales by 128 and saturates to the i8 range before narrowing.
    #[inline(always)]
    fn quantize_value(value: f32) -> i8 {
        ((value * 128.).round() as i32).clamp(-128, 127) as i8
    }

    /// Computes the inverse L2 norm of the provided array of values. Returns 1.0 in case all values are 0.
    fn get_inverse_l2_norm(values: &[f32]) -> f32 {
        let mut squared_l2_norm = 0.0;
        for v in values {
            let value = *v;
            squared_l2_norm += v * v;
        }

        let mut inv_l2_norm = 1.0;
        if squared_l2_norm > 0.0 {
            inv_l2_norm = 1.0 / squared_l2_norm.sqrt();
        }
        return inv_l2_norm;
    }
}

#[cfg(test)]
mod test {
    use super::TensorsToEmbedding;

    #[test]
    fn test_quantize_value_saturates() {
        assert_eq!(TensorsToEmbedding::quantize_value(0.5), 64);
        assert_eq!(TensorsToEmbedding::quantize_value(-0.5), -64);
        assert_eq!(TensorsToEmbedding::quantize_value(1.0), 127);
        assert_eq!(TensorsToEmbedding::quantize_value(-1.0), -128);
        assert_eq!(TensorsToEmbedding::quantize_value(3.0), 127);
        assert_eq!(TensorsToEmbedding::quantize_value(-3.0), -128);
    }
}
