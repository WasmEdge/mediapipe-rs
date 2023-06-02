/// Quantization parameters corresponding to the zero_point and scale value.
#[derive(Debug, Copy, Clone)]
pub struct QuantizationParameters {
    pub scale: f32,
    pub zero_point: i32,
}

pub(crate) trait Dequantize {
    fn dequantize(&self, quantization_parameters: QuantizationParameters) -> Vec<f32>;

    fn dequantize_to_buf(
        &self,
        quantization_parameters: QuantizationParameters,
        out_buf: &mut [f32],
    );
}

impl Dequantize for &[u8] {
    #[inline(always)]
    fn dequantize(&self, quantization_parameters: QuantizationParameters) -> Vec<f32> {
        let mut res = Vec::with_capacity(self.len());
        self.dequantize_to_buf(quantization_parameters, res.as_mut_slice());
        res
    }

    #[inline(always)]
    fn dequantize_to_buf(
        &self,
        quantization_parameters: QuantizationParameters,
        out_buf: &mut [f32],
    ) {
        for i in 0..self.len() {
            out_buf[i] = quantization_parameters.scale
                * (self[i] as i32 - quantization_parameters.zero_point) as f32;
        }
    }
}
