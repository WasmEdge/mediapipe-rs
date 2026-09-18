use crate::preprocess::audio::{AudioData, AudioDataToTensorIter};
use crate::tasks::audio::TaskSession;
use crate::Error;

/// Iterator over the results of an audio stream. Each chunk is processed when the next item
/// is polled, so the iterator yields `Result` items.
pub struct AudioResultsIter<'session, 'tensor, Session, Source>
where
    Session: TaskSession + 'session,
    Source: AudioData + 'tensor,
{
    audio_data: AudioDataToTensorIter<'tensor, Source>,
    session: &'session mut Session,
}

impl<'session, 'tensor, Session, Source> AudioResultsIter<'session, 'tensor, Session, Source>
where
    Session: TaskSession + 'session,
    Source: AudioData + 'tensor,
{
    pub(crate) fn new(
        session: &'session mut Session,
        audio_data: AudioDataToTensorIter<'tensor, Source>,
    ) -> Self {
        Self {
            audio_data,
            session,
        }
    }

    /// Process all remaining audio and collect the results.
    pub fn to_vec(self) -> Result<Vec<Session::Result>, Error> {
        self.collect()
    }
}

impl<'session, 'tensor, Session, Source> Iterator
    for AudioResultsIter<'session, 'tensor, Session, Source>
where
    Session: TaskSession + 'session,
    Source: AudioData + 'tensor,
{
    type Item = Result<Session::Result, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.session.process_next(&mut self.audio_data).transpose()
    }
}
