use crate::tasks::vision::{ImageProcessingOptions, TaskSession};
use crate::Error;

/// Iterator over the results of a video stream. Each frame is processed when the next item
/// is polled, so the iterator yields `Result` items.
pub struct VideoResultsIter<'session, Session, VideoData>
where
    Session: TaskSession + 'session,
    VideoData: crate::preprocess::vision::VideoData,
{
    video_data: VideoData,
    session: &'session mut Session,
    process_options: ImageProcessingOptions,
}

impl<'session, Session, VideoData> VideoResultsIter<'session, Session, VideoData>
where
    Session: TaskSession + 'session,
    VideoData: crate::preprocess::vision::VideoData,
{
    pub(crate) fn new(session: &'session mut Session, video_data: VideoData) -> Self {
        Self {
            video_data,
            session,
            process_options: ImageProcessingOptions::default(),
        }
    }

    /// Set the image processing options used for every following frame.
    pub fn with_options(mut self, process_options: ImageProcessingOptions) -> Self {
        self.process_options = process_options;
        self
    }

    /// Process the next frame with the given options.
    pub fn next_with_options(
        &mut self,
        process_options: &ImageProcessingOptions,
    ) -> Result<Option<Session::Result>, Error> {
        self.session
            .process_next(process_options, &mut self.video_data)
    }

    /// Process all remaining frames and collect the results.
    pub fn to_vec(self) -> Result<Vec<Session::Result>, Error> {
        self.collect()
    }
}

impl<'session, Session, VideoData> Iterator for VideoResultsIter<'session, Session, VideoData>
where
    Session: TaskSession + 'session,
    VideoData: crate::preprocess::vision::VideoData,
{
    type Item = Result<Session::Result, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.session
            .process_next(&self.process_options, &mut self.video_data)
            .transpose()
    }
}
