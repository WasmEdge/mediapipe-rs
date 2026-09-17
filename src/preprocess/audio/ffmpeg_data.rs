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
    ( $self:ident, $num_channels:ident, $num_samples:ident, $sample_buffer:ident, $tp:ty, $conv:expr ) => {{
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
                output[i] = $conv(samples[i]);
            }
        }
    }};
}

macro_rules! output_tuple_to_buffer {
    ( $samples:ident, $channel:tt, $num_samples:ident, $sample_buffer:ident, $conv:expr ) => {
        if $sample_buffer.len() <= $channel {
            $sample_buffer.push(Vec::with_capacity($num_samples));
        }
        let buffer = $sample_buffer.get_mut($channel).unwrap();
        if buffer.len() < $num_samples {
            buffer.resize($num_samples, 0.);
        }
        for i in 0..$num_samples {
            buffer[i] = $conv($samples[i].$channel);
        }
    };
}

macro_rules! process_samples {
    ( $format:ident, $self:ident, $num_channels:ident, $num_samples:ident, $sample_buffer:ident, $tp:ty, $conv:expr ) => {
        match $format {
            ffmpeg_next::format::sample::Type::Packed => match $num_channels {
                1 => {
                    output_to_buffer!(
                        $self,
                        $num_channels,
                        $num_samples,
                        $sample_buffer,
                        $tp,
                        $conv
                    );
                }
                2 => {
                    let samples = $self.frame.plane::<($tp, $tp)>(0);
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, $conv);
                }
                3 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp)>(0);
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, $conv);
                }
                4 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp)>(0);
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, $conv);
                }
                5 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp, $tp)>(0);
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 4, $num_samples, $sample_buffer, $conv);
                }
                6 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp, $tp, $tp)>(0);
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 4, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 5, $num_samples, $sample_buffer, $conv);
                }
                7 => {
                    let samples = $self.frame.plane::<($tp, $tp, $tp, $tp, $tp, $tp, $tp)>(0);
                    output_tuple_to_buffer!(samples, 0, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 1, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 2, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 3, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 4, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 5, $num_samples, $sample_buffer, $conv);
                    output_tuple_to_buffer!(samples, 6, $num_samples, $sample_buffer, $conv);
                }
                _ => {
                    return Err(Error::ArgumentError(format!(
                        "unsupported number of channels `{}`",
                        $num_channels
                    )))
                }
            },
            ffmpeg_next::format::sample::Type::Planar => {
                output_to_buffer!(
                    $self,
                    $num_channels,
                    $num_samples,
                    $sample_buffer,
                    $tp,
                    $conv
                );
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
        sample_buffer.truncate(num_channels);

        match self.frame.format() {
            ffmpeg_next::format::Sample::U8(tp) => {
                process_samples!(
                    tp,
                    self,
                    num_channels,
                    num_samples,
                    sample_buffer,
                    u8,
                    |s: u8| { (s as f32 - 128.) / 128. }
                );
            }
            ffmpeg_next::format::Sample::I16(tp) => {
                process_samples!(
                    tp,
                    self,
                    num_channels,
                    num_samples,
                    sample_buffer,
                    i16,
                    |s: i16| { s as f32 / 32_768. }
                );
            }
            ffmpeg_next::format::Sample::I32(tp) => {
                process_samples!(
                    tp,
                    self,
                    num_channels,
                    num_samples,
                    sample_buffer,
                    i32,
                    |s: i32| { (s as f64 / 2_147_483_648.) as f32 }
                );
            }
            ffmpeg_next::format::Sample::F32(tp) => {
                process_samples!(
                    tp,
                    self,
                    num_channels,
                    num_samples,
                    sample_buffer,
                    f32,
                    |s: f32| s
                );
            }
            ffmpeg_next::format::Sample::F64(tp) => {
                process_samples!(
                    tp,
                    self,
                    num_channels,
                    num_samples,
                    sample_buffer,
                    f64,
                    |s: f64| { s as f32 }
                );
            }
            format @ (ffmpeg_next::format::Sample::I64(_) | ffmpeg_next::format::Sample::None) => {
                return Err(Error::ArgumentError(format!(
                    "Unsupported ffmpeg sample format `{:?}`",
                    format
                )));
            }
        }
        return Ok(Some((sample_rate, num_samples)));
    }
}
