mod audio_classification;

pub use audio_classification::{AudioClassifier, AudioClassifierBuilder, AudioClassifierSession};

/// Task session trait to process the audio stream data
pub trait TaskSession {
    type Result: 'static;

    /// Process the next tensors from input stream
    fn process_next<Source: crate::preprocess::audio::AudioData>(
        &mut self,
        audio_stream_data: &mut crate::preprocess::audio::AudioDataToTensorIter<Source>,
    ) -> Result<Option<Self::Result>, crate::Error>;
}
