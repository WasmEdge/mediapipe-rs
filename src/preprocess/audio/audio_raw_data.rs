use super::*;

enum Matrix<'a, T = Vec<Vec<f32>>, E = Vec<f32>>
where
    T: AsRef<[E]> + 'a,
    E: AsRef<[f32]>,
{
    Owned {
        data: T,
        _marker: std::marker::PhantomData<E>,
    },
    Borrowed(&'a T),
    Take(Vec<Vec<f32>>),
}

/// Raw Audio matrix data.
pub struct AudioRawData<'a, T = Vec<Vec<f32>>, E = Vec<f32>>
where
    T: AsRef<[E]> + 'a,
    E: AsRef<[f32]>,
{
    matrix: Matrix<'a, T, E>,
    now_index: usize,
    sample_rate: usize,
    _marker: std::marker::PhantomData<E>,
}

impl<'a, T, E> AudioRawData<'a, T, E>
where
    T: AsRef<[E]> + 'a,
    E: AsRef<[f32]>,
{
    /// Create a new reference for data matrix.
    pub fn new_ref(raw_major_matrix: &'a T, sample_rate: usize) -> Result<Self, Error> {
        Self::check(raw_major_matrix)?;
        Ok(Self {
            matrix: Matrix::Borrowed(raw_major_matrix),
            now_index: 0,
            sample_rate,
            _marker: Default::default(),
        })
    }

    /// Create a new owned data for matrix.
    pub fn new(raw_major_matrix: T, sample_rate: usize) -> Result<Self, Error> {
        Self::check(&raw_major_matrix)?;
        Ok(Self {
            matrix: Matrix::Owned {
                data: raw_major_matrix,
                _marker: Default::default(),
            },
            now_index: 0,
            sample_rate,
            _marker: Default::default(),
        })
    }

    /// Reset the state, will read from start.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.now_index = 0;
    }

    /// Get sample rate.
    #[inline(always)]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Get the number of channels.
    #[inline(always)]
    pub fn num_channels(&self) -> usize {
        match &self.matrix {
            Matrix::Owned { data, .. } => data.as_ref().len(),
            Matrix::Borrowed(b) => b.as_ref().len(),
            Matrix::Take(v) => v.len(),
        }
    }

    /// Check whether the raw major matrix can be used as Audio Raw Data.
    #[inline(always)]
    pub fn check(raw_major_matrix: &T) -> Result<(), Error> {
        let channels = raw_major_matrix.as_ref();
        let num_channels = channels.len();
        if num_channels == 0 {
            return Err(Error::ArgumentError("Num channels cannot be `0`".into()));
        }
        let len = channels[0].as_ref().len();
        for i in 1..num_channels {
            if len != channels[i].as_ref().len() {
                return Err(Error::ArgumentError(format!(
                    "Data is not a matrix, expect channel[`{}`] len is `{}`, but got `{}",
                    i,
                    len,
                    channels[i].as_ref().len()
                )));
            }
        }
        Ok(())
    }
}

impl<'a> AudioRawData<'a, Vec<Vec<f32>>, Vec<f32>> {
    pub fn new_into(raw_major_matrix: Vec<Vec<f32>>, sample_rate: usize) -> Result<Self, Error> {
        Self::check(&raw_major_matrix)?;
        Ok(Self {
            matrix: Matrix::Take(raw_major_matrix),
            sample_rate,
            now_index: 0,
            _marker: Default::default(),
        })
    }
}

impl<'a, T, E> AudioData for AudioRawData<'a, T, E>
where
    T: AsRef<[E]> + 'a,
    E: AsRef<[f32]>,
{
    fn next_frame(
        &mut self,
        sample_buffer: &mut Vec<Vec<f32>>,
    ) -> Result<Option<(usize, usize)>, Error> {
        let buf_t = match &mut self.matrix {
            Matrix::Owned { data, .. } => data,
            Matrix::Borrowed(b) => *b,
            Matrix::Take(v) => {
                // just move
                let num_samples = v[0].len();
                if num_samples == 0 {
                    return Ok(None);
                }

                let num_channels = v.len();
                for c in 0..num_channels {
                    let take = std::mem::take(v.get_mut(c).unwrap());
                    if sample_buffer.len() <= c {
                        sample_buffer.push(take);
                    } else {
                        sample_buffer[c] = take;
                    }
                }
                return Ok(Some((self.sample_rate, num_samples)));
            }
        };
        let buf = buf_t.as_ref();

        let num_channels = buf.len();
        let max_samples = buf[0].as_ref().len();
        if self.now_index >= max_samples {
            return Ok(None);
        }

        let num_samples = std::cmp::min(self.sample_rate, max_samples - self.now_index);
        let data_end = self.now_index + num_samples;
        for c in 0..num_channels {
            if sample_buffer.len() <= c {
                sample_buffer.push(Vec::new());
            }
            let output_buffer = sample_buffer.get_mut(c).unwrap();
            if output_buffer.len() < num_samples {
                output_buffer.resize(num_samples, 0.);
            }
            output_buffer[..num_samples]
                .copy_from_slice(&buf[c].as_ref()[self.now_index..data_end]);
        }
        self.now_index = data_end;

        Ok(Some((self.sample_rate, num_samples)))
    }
}
