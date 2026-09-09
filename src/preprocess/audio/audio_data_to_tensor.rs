use super::*;
use std::collections::VecDeque;

impl<'a, Source: AudioData> AudioDataToTensorIter<'a, Source> {
    pub(crate) fn poll_next_tensors<T: AsMut<[E]>, E: AsMut<[u8]>>(
        &mut self,
        output_buffers: &mut T,
    ) -> Result<Option<u64>, Error> {
        // todo: num_overlapping_samples, fft if need
        let timestamp_ms = self.processed_timestamp_ms;
        while self.process_buffer.len() == 0
            || self.process_buffer[0].len() < self.audio_to_tensor_info.num_samples
        {
            if let Some((sample_rate, num_samples)) =
                self.source.next_frame(&mut self.input_buffer)?
            {
                self.input_sample_rate = sample_rate;
                let num_samples = self.preprocess_input_buffer(sample_rate, num_samples)?;
                for c in 0..self.audio_to_tensor_info.num_channels {
                    if self.process_buffer.len() <= c {
                        self.process_buffer.push(VecDeque::with_capacity(
                            self.audio_to_tensor_info.num_samples << 1,
                        ));
                    }
                    self.process_buffer[c].extend(&self.input_buffer[c][..num_samples]);
                }
            } else {
                break;
            }
        }

        // stream end
        if self.process_buffer.len() == 0 || self.process_buffer[0].len() == 0 {
            return Ok(None);
        }

        self.output_to_tensor(output_buffers.as_mut()[0].as_mut())?;
        Ok(Some(timestamp_ms))
    }

    pub(crate) fn new(
        audio_to_tensor_info: &'a AudioToTensorInfo,
        source: Source,
    ) -> Result<Self, Error> {
        // reference: https://github.com/google/mediapipe/blob/master/mediapipe/tasks/cc/audio/utils/audio_tensor_specs.cc
        if audio_to_tensor_info.tensor_type != TensorType::F32 {
            return Err(Error::ModelInconsistentError(format!(
                "Audio model input must be F32, but got `{:?}`",
                audio_to_tensor_info.tensor_type
            )));
        }

        Ok(Self {
            audio_to_tensor_info,
            source,
            input_buffer: Vec::new(),
            process_buffer: Vec::new(),
            input_num_channels: 0,
            processed_timestamp_ms: 0,
            input_sample_rate: 0,
        })
    }

    // return the num_samples
    fn preprocess_input_buffer(
        &mut self,
        sample_rate: usize,
        num_samples: usize,
    ) -> Result<usize, Error> {
        let num_samples = num_samples as usize;
        let num_channels = self.input_buffer.len();
        if num_channels == 0 {
            return Err(Error::ArgumentError("Num channels cannot be `0`".into()));
        }
        if self.input_num_channels == 0 {
            self.input_num_channels = num_channels;
        } else {
            if self.input_num_channels != num_channels {
                return Err(Error::ArgumentError(format!(
                    "Audio Channels are not match with last package, expect `{}`, but got `{}`",
                    self.input_num_channels, num_channels
                )));
            }
        }

        for i in 0..num_channels {
            if self.input_buffer[i].len() < num_samples {
                return Err(Error::ArgumentError(format!(
                    "Audio input channel `{}` expect `{}` samples, but got `{}`",
                    i,
                    self.input_buffer[i].len(),
                    num_samples
                )));
            }
        }

        let mono_output = self.audio_to_tensor_info.num_channels == 1;
        let channels_mismatch = num_channels != self.audio_to_tensor_info.num_channels;
        if channels_mismatch && !mono_output {
            return Err(Error::ArgumentError(format!(
                "Audio input has `{}` channel(s) but the model requires `{}` channel(s)",
                num_channels, self.audio_to_tensor_info.num_channels
            )));
        }

        if channels_mismatch {
            // downmix to mono: average all channels into channel 0
            let (mean, buffers) = self.input_buffer.as_mut_slice().split_at_mut(1);
            let mean = &mut mean[0][..num_samples];
            for samples in buffers {
                for (m, s) in mean.iter_mut().zip(&samples[..num_samples]) {
                    *m += *s;
                }
            }
            let div = num_channels as f32;
            mean.iter_mut().for_each(|c| *c /= div);
        };

        if sample_rate != self.audio_to_tensor_info.sample_rate {
            return Err(Error::ArgumentError(format!(
                "Audio sample rate `{}` does not match the model sample rate `{}`, resampling is not supported",
                sample_rate, self.audio_to_tensor_info.sample_rate
            )));
        }

        return Ok(num_samples);
    }

    fn output_to_tensor(&mut self, output_buffer: &mut [u8]) -> Result<(), Error> {
        let num_samples = self.audio_to_tensor_info.num_samples;
        let sample_bytes = std::mem::size_of::<f32>();
        let channel_bytes = num_samples * sample_bytes;
        let expected = self.audio_to_tensor_info.num_channels * channel_bytes;
        if output_buffer.len() < expected {
            return Err(Error::ArgumentError(format!(
                "Expect output buffer at least `{}` bytes, but got `{}`",
                expected,
                output_buffer.len()
            )));
        }

        for (c, out) in output_buffer[..expected]
            .chunks_exact_mut(channel_bytes)
            .enumerate()
        {
            let buffer = &mut self.process_buffer[c];
            let process_len = std::cmp::min(buffer.len(), num_samples);
            let padding = std::iter::repeat(0f32).take(num_samples - process_len);
            write_ne_bytes(out, buffer.drain(..process_len).chain(padding));
            if c == 0 {
                self.processed_timestamp_ms +=
                    (process_len as f64 / self.audio_to_tensor_info.sample_rate as f64 * 1000.)
                        .round() as u64;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn info(num_channels: usize) -> AudioToTensorInfo {
        AudioToTensorInfo {
            num_channels,
            num_samples: 4,
            sample_rate: 4,
            num_overlapping_samples: 0,
            tensor_type: TensorType::F32,
        }
    }

    fn poll(info: &AudioToTensorInfo, channels: Vec<Vec<f32>>) -> Result<Vec<f32>, Error> {
        let data = AudioRawData::new(channels, 4)?;
        let mut iter = AudioDataToTensorIter::new(info, data)?;
        let mut buffers = [vec![0u8; info.num_channels * info.num_samples * 4]];
        iter.poll_next_tensors(&mut buffers)?;
        Ok(buffers[0]
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }

    #[test]
    fn test_stereo_input_downmixes_to_mono_model() {
        let out = poll(&info(1), vec![vec![1., 1., 1., 1.], vec![3., 3., 3., 3.]]).unwrap();
        assert_eq!(out, [2., 2., 2., 2.]);
    }

    #[test]
    fn test_matching_channels_pass_through() {
        let out = poll(&info(2), vec![vec![1., 1., 1., 1.], vec![3., 3., 3., 3.]]).unwrap();
        assert_eq!(out, [1., 1., 1., 1., 3., 3., 3., 3.]);
    }

    #[test]
    fn test_channel_mismatch_for_multichannel_model_is_error() {
        assert!(poll(&info(2), vec![vec![1., 1., 1., 1.]]).is_err());
    }
}
