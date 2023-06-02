use super::*;
use common::ffmpeg_input::FFMpegInput;

type FFMpegAudioDataInner = FFMpegInput<ffmpeg_next::decoder::Audio, ffmpeg_next::frame::Audio>;

/// Audio Data which using the `FFMpeg` library as a decoder.
pub struct FFMpegAudioData(FFMpegAudioDataInner);

impl FFMpegAudioData {
    /// Create a new instance from FFMpeg input.
    #[inline(always)]
    pub fn new(input: ffmpeg_next::format::context::Input) -> Result<Self, Error> {
        FFMpegAudioDataInner::new(input).map(|i| Self(i))
    }
}

impl std::ops::Deref for FFMpegAudioData {
    type Target = FFMpegAudioDataInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for FFMpegAudioData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

macro_rules! output_to_buffer {
    ( $self:ident, $num_channels:ident, $num_samples:ident, $sample_buffer:ident, $tp:ty ) => {{
        let max_value = <$tp>::MAX as f32;
        for c in 0..$num_channels {
            if $sample_buffer.len() <= c {
                $sample_buffer.push(Vec::with_capacity($num_samples));
            }
            let output = $sample_buffer.get_mut(c).unwrap();
            if output.len() < $num_samples {
                output.resize($num_samples, 0.);
            }
            let samples = $self.frame.plane::<$tp>(c);
            for i in 0..$num_samples {
                output[i] = samples[i] as f32 / max_value;
            }
        }
    }};
}

macro_rules! output_tuple_to_buffer {
    ( $samples:ident, $channel:tt, $num_samples:ident, $sample_buffer:ident, $max:ident ) => {
        if $sample_buffer.len() <= $channel {
            $sample_buffer.push(Vec::with_capacity($num_samples));
        }
        let buffer = $sample_buffer.get_mut($channel).unwrap();
        if buffer.len() < $num_samples {
            buffer.resize($num_samples, 0.);
        }
        for i in 0..$num_samples {
            buffer[i] = $samples[i].$channel as f32 / $max;
        }
    };
}

macro_rules! process_samples {
    ( $format:ident, $self:ident, $num_channels:ident, $num_samples:ident, $sample_buffer:ident, $tp:ty ) => {
        match $format {
            ffmpeg_next::format::sample::Type::Packed => match $num_channels {
                1 => {
                    output_to_buffer!($self, $num_channels, $num_samples, $sample_buffer, $tp);
                }
                2 => {
                    let samples = $self.frame.plane::<($tp, $tp)>(0);
                    let max = <$tp>::MAX as f32;
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, max);
                }
                3 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp)>(0);
                    let max = <$tp>::MAX as f32;
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, max);
                }
                4 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp)>(0);
                    let max = <$tp>::MAX as f32;
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, max);
                }
                5 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp, $tp)>(0);
                    let max = <$tp>::MAX as f32;
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 4, $num_samples, $sample_buffer, max);
                }
                6 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp, $tp, $tp)>(0);
                    let max = <$tp>::MAX as f32;
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 4, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 5, $num_samples, $sample_buffer, max);
                }
                7 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp, $tp, $tp, $tp)>(0);
                    let max = <$tp>::MAX as f32;
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 4, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 5, $num_samples, $sample_buffer, max);
                    output_tuple_to_buffer!(samples, 6, $num_samples, $sample_buffer, max);
                }
                _ => {
                    return Err(Error::ArgumentError(format!(
                        "unsupported number of channels `{}`",
                        $num_channels
                    )))
                }
            },
            ffmpeg_next::format::sample::Type::Planar => {
                output_to_buffer!($self, $num_channels, $num_samples, $sample_buffer, $tp);
            }
        }
    };
}

impl AudioData for FFMpegAudioData {
    /// return (sample_rate, num_samples), save the sample in sample_buffer,
    /// sample data must be range in ```[-1.0,1.0]```.
    fn next_frame(
        &mut self,
        sample_buffer: &mut Vec<Vec<f32>>,
    ) -> Result<Option<(usize, usize)>, Error> {
        if !self.receive_frame()? {
            return Ok(None);
        }

        let sample_rate = self.frame.rate() as usize;
        let num_channels = self.frame.channels() as usize;
        let num_samples = self.frame.samples();

        match self.frame.format() {
            ffmpeg_next::format::Sample::U8(tp) => {
                process_samples!(tp, self, num_channels, num_samples, sample_buffer, u8);
            }
            ffmpeg_next::format::Sample::I16(tp) => {
                process_samples!(tp, self, num_channels, num_samples, sample_buffer, i16);
            }
            ffmpeg_next::format::Sample::I32(tp) => {
                process_samples!(tp, self, num_channels, num_samples, sample_buffer, i32);
            }
            ffmpeg_next::format::Sample::I64(_) => {
                unimplemented!()
            }
            ffmpeg_next::format::Sample::F32(_) => {
                unimplemented!()
            }
            ffmpeg_next::format::Sample::F64(_) => {
                unimplemented!()
            }
            ffmpeg_next::format::Sample::None => {
                return Err(Error::ArgumentError(
                    "Unsupported ffmpeg sample format `None`".into(),
                ));
            }
        }
        return Ok(Some((sample_rate, num_samples)));
    }
}
