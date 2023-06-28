use super::*;
use symphonia_core::audio::{AudioBufferRef, Signal};
use symphonia_core::codecs::Decoder;
use symphonia_core::formats::FormatReader;

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

macro_rules! output_to_buffer {
    ( $audio:ident, $sample_buffer:ident, $tp:ty ) => {{
        let max = <$tp>::MAX as f32;
        let spec = $audio.spec();
        let sample_rate = spec.rate;
        let num_channels = spec.channels.count();
        let mut num_samples = 0;

        for c in 0..num_channels {
            if $sample_buffer.len() <= c {
                $sample_buffer.push(Vec::new());
            }
            let output_buffer = $sample_buffer.get_mut(c).unwrap();

            let samples = $audio.chan(c);
            num_samples = samples.len();

            if output_buffer.len() < num_samples {
                output_buffer.resize(num_samples, 0.);
            }

            for i in 0..num_samples {
                output_buffer[i] = samples[i] as f32 / max;
            }
        }
        Ok(Some((sample_rate as usize, num_samples)))
    }};
}

impl AudioData for SymphoniaAudioData {
    #[inline]
    fn next_frame(
        &mut self,
        sample_buffer: &mut Vec<Vec<f32>>,
    ) -> Result<Option<(usize, usize)>, Error> {
        match self.format_reader.next_packet() {
            Ok(p) => match self.decoder.decode(&p)? {
                AudioBufferRef::U8(r) => {
                    output_to_buffer!(r, sample_buffer, u8)
                }
                AudioBufferRef::S16(r) => {
                    output_to_buffer!(r, sample_buffer, i16)
                }
                AudioBufferRef::U16(r) => {
                    output_to_buffer!(r, sample_buffer, u16)
                }
                AudioBufferRef::U32(r) => {
                    output_to_buffer!(r, sample_buffer, u32)
                }
                _ => unimplemented!(),
            },
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
