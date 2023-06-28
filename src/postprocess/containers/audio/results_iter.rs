/// Used for audio task results
pub struct AudioResultsIter<'session, 'tensor, TaskSession, AudioDataSource>
where
    TaskSession: crate::tasks::audio::TaskSession + 'session,
    AudioDataSource: crate::preprocess::audio::AudioData + 'tensor,
{
    audio_data: crate::preprocess::audio::AudioDataToTensorIter<'tensor, AudioDataSource>,
    session: &'session mut TaskSession,
}

impl<'session, 'tensor, TaskSession, AudioDataSource>
    AudioResultsIter<'session, 'tensor, TaskSession, AudioDataSource>
where
    TaskSession: crate::tasks::audio::TaskSession + 'session,
    AudioDataSource: crate::preprocess::audio::AudioData + 'tensor,
{
    #[inline(always)]
    pub(crate) fn new(
        session: &'session mut TaskSession,
        audio_data: crate::preprocess::audio::AudioDataToTensorIter<'tensor, AudioDataSource>,
    ) -> Self {
        Self {
            audio_data,
            session,
        }
    }

    /// poll next result
    #[inline(always)]
    pub fn next(&mut self) -> Result<Option<TaskSession::Result>, crate::Error> {
        self.session.process_next(&mut self.audio_data)
    }

    results_iter_impl!();
}
