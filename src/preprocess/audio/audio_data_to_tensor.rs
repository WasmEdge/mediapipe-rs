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

        self.output_to_tensor(&mut output_buffers.as_mut()[0]);
        Ok(Some(timestamp_ms))
    }

    pub(crate) fn new(
        audio_to_tensor_info: &'a AudioToTensorInfo,
        source: Source,
    ) -> Result<Self, Error> {
        match audio_to_tensor_info.tensor_type {
            // reference: https://github.com/google/mediapipe/blob/master/mediapipe/tasks/cc/audio/utils/audio_tensor_specs.cc
            TensorType::F16 | TensorType::F32 => {}
            _ => {
                return Err(Error::ModelInconsistentError(
                    "Model only support F32 or F16 input now.".into(),
                ));
            }
        };

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
        let channels_match = num_channels != self.audio_to_tensor_info.num_channels as usize;
        if !mono_output && !channels_match {
            return Err(Error::ArgumentError(format!(
                "Audio input has `{}` channel(s) but the model requires `{}` channel(s)",
                num_channels, self.audio_to_tensor_info.num_channels
            )));
        }

        if !channels_match {
            // cal the mean
            let (mean, buffers) = self.input_buffer.as_mut_slice().split_at_mut(1);
            let mean = mean[0].as_mut_slice();
            for samples in buffers {
                for j in 0..samples.len() {
                    mean[j] += samples[j];
                }
            }
            let div = num_channels as f32;
            mean.iter_mut().for_each(|c| *c /= div);
        };

        if sample_rate != self.audio_to_tensor_info.sample_rate {
            todo!("resample");
        }

        return Ok(num_samples);
    }

    fn output_to_tensor(&mut self, output_buffer: &mut impl AsMut<[u8]>) {
        match self.audio_to_tensor_info.tensor_type {
            TensorType::F16 => {
                todo!("fp16")
            }
            TensorType::F32 => {
                let output_buffer = unsafe {
                    core::slice::from_raw_parts_mut(
                        output_buffer.as_mut().as_mut_ptr() as *mut f32,
                        output_buffer.as_mut().len() / std::mem::size_of::<f32>(),
                    )
                };

                let mut index = 0;
                for c in 0..self.audio_to_tensor_info.num_channels {
                    let mut rest_need = self.audio_to_tensor_info.num_samples;
                    let (s1, s2) = self.process_buffer[c].as_slices();

                    let mut copy_len = std::cmp::min(s1.len(), rest_need);
                    let mut next_index = index + copy_len;
                    output_buffer[index..next_index].copy_from_slice(&s1[..copy_len]);
                    index = next_index;
                    rest_need -= copy_len;

                    if rest_need != 0 {
                        copy_len = std::cmp::min(s2.len(), rest_need);
                        next_index = index + copy_len;
                        output_buffer[index..next_index].copy_from_slice(&s2[..copy_len]);
                        index = next_index;
                        rest_need -= copy_len;
                    }

                    let process_len = if rest_need != 0 {
                        next_index = index + rest_need;
                        output_buffer[index..next_index].fill(0.);
                        index = next_index;
                        self.audio_to_tensor_info.num_samples - rest_need
                    } else {
                        self.audio_to_tensor_info.num_samples
                    };

                    self.process_buffer[c].drain(..process_len);
                    if c == 0 {
                        self.processed_timestamp_ms += (process_len as f64
                            / self.audio_to_tensor_info.sample_rate as f64
                            * 1000.)
                            .round() as u64;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}
