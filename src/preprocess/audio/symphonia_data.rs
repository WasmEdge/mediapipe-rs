use super::*;
use symphonia_core::audio::{AudioBuffer, AudioBufferRef, Signal};
use symphonia_core::codecs::Decoder;
use symphonia_core::conv::IntoSample;
use symphonia_core::formats::FormatReader;
use symphonia_core::sample::Sample;

/// Audio Data which using the `symphonia` crate as a decoder.
pub struct SymphoniaAudioData {
    format_reader: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
}

impl SymphoniaAudioData {
    /// Create a new Symphonia Audio Data.
    #[inline(always)]
    pub fn new(format_reader: Box<dyn FormatReader>, decoder: Box<dyn Decoder>) -> Self {
        Self {
            format_reader,
            decoder,
        }
    }
}

/// Convert every channel to `f32` samples in `[-1.0, 1.0]`. Return `(sample_rate, num_samples)`.
fn output_to_buffer<S>(audio: &AudioBuffer<S>, sample_buffer: &mut Vec<Vec<f32>>) -> (usize, usize)
where
    S: Sample + IntoSample<f32>,
{
    let spec = audio.spec();
    let num_channels = spec.channels.count();
    sample_buffer.resize_with(num_channels, Vec::new);
    for (c, output_buffer) in sample_buffer.iter_mut().enumerate() {
        output_buffer.clear();
        output_buffer.extend(audio.chan(c).iter().map(|s| (*s).into_sample()));
    }
    (spec.rate as usize, audio.frames())
}

impl AudioData for SymphoniaAudioData {
    #[inline]
    fn next_frame(
        &mut self,
        sample_buffer: &mut Vec<Vec<f32>>,
    ) -> Result<Option<(usize, usize)>, Error> {
        match self.format_reader.next_packet() {
            Ok(p) => {
                let (sample_rate, num_samples) = match self.decoder.decode(&p)? {
                    AudioBufferRef::U8(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::U16(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::U24(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::U32(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::S8(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::S16(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::S24(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::S32(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::F32(r) => output_to_buffer(&r, sample_buffer),
                    AudioBufferRef::F64(r) => output_to_buffer(&r, sample_buffer),
                };
                Ok(Some((sample_rate, num_samples)))
            }
            Err(e) => {
                if let symphonia_core::errors::Error::IoError(e) = &e {
                    // end of stream
                    if e.kind() == std::io::ErrorKind::UnexpectedEof {
                        return Ok(None);
                    }
                }
                return Err(Error::from(e));
            }
        }
    }
}
