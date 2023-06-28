/// Used for stream data results, such video, audio.
pub struct VideoResultsIter<'session, 'tensor, TaskSession, VideoData>
where
    TaskSession: crate::tasks::vision::TaskSession + 'session,
    VideoData: crate::preprocess::vision::VideoData,
{
    video_data: VideoData,
    session: &'session mut TaskSession,
    _marker: std::marker::PhantomData<&'tensor ()>,
}

impl<'session, 'tensor, TaskSession, VideoData>
    VideoResultsIter<'session, 'tensor, TaskSession, VideoData>
where
    TaskSession: crate::tasks::vision::TaskSession + 'session,
    VideoData: crate::preprocess::vision::VideoData,
{
    #[inline(always)]
    pub(crate) fn new(session: &'session mut TaskSession, video_data: VideoData) -> Self {
        Self {
            video_data,
            session,
            _marker: Default::default(),
        }
    }

    /// poll next result
    #[inline(always)]
    pub fn next(&mut self) -> Result<Option<TaskSession::Result>, crate::Error> {
        self.session
            .process_next(&Default::default(), &mut self.video_data)
    }

    /// poll next result
    #[inline(always)]
    pub fn next_with_options(
        &mut self,
        process_options: &crate::tasks::vision::ImageProcessingOptions,
    ) -> Result<Option<TaskSession::Result>, crate::Error> {
        self.session
            .process_next(process_options, &mut self.video_data)
    }

    results_iter_impl!();
}
