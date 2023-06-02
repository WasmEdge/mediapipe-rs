macro_rules! detector_impl {
    ( $DetectorSessionName:ident, $Result:ident ) => {
        base_task_options_get_impl!();

        /// Detect one image using a new session.
        #[inline(always)]
        pub fn detect(
            &self,
            input: &impl crate::preprocess::vision::ImageToTensor,
        ) -> Result<$Result, crate::Error> {
            self.new_session()?.detect(input)
        }

        /// Detect input video stream in a new session, and collect all results to [`Vec`].
        #[inline(always)]
        pub fn detect_for_video(
            &self,
            video_data: impl crate::preprocess::vision::VideoData,
        ) -> Result<Vec<$Result>, crate::Error> {
            self.new_session()?.detect_for_video(video_data)?.to_vec()
        }
    };
}

macro_rules! detector_session_impl {
    ( $Result:ident ) => {
        /// Detect one image using this session.
        #[inline(always)]
        pub fn detect(
            &mut self,
            input: &impl crate::preprocess::vision::ImageToTensor,
        ) -> Result<$Result, crate::Error> {
            input.to_tensor(
                self.image_to_tensor_info,
                &Default::default(),
                &mut self.input_buffer,
            )?;
            self.compute(input.timestamp_ms())
        }

        /// Detect input video stream use this session.
        /// Return a iterator for results, process input stream when poll next result.
        #[inline(always)]
        pub fn detect_for_video<InputVideoData: crate::preprocess::vision::VideoData>(
            &mut self,
            video_data: InputVideoData,
        ) -> Result<crate::postprocess::VideoResultsIter<Self, InputVideoData>, crate::Error> {
            Ok(crate::postprocess::VideoResultsIter::new(self, video_data))
        }
    };
}

macro_rules! detection_task_session_impl {
    ( $SessionName:ident, $Result:ident ) => {
        use crate::preprocess::vision::ImageToTensor;

        impl<'model> super::TaskSession for $SessionName<'model> {
            type Result = $Result;

            #[inline]
            fn process_next(
                &mut self,
                process_options: &super::ImageProcessingOptions,
                video_data: &mut impl crate::preprocess::vision::VideoData,
            ) -> Result<Option<Self::Result>, crate::Error> {
                if process_options.region_of_interest.is_some() {
                    return Err(crate::Error::ArgumentError(format!(
                        "{} does not support region of interest.",
                        stringify!($SessionName)
                    )));
                }

                // todo: support rotation
                assert_eq!(process_options.rotation, 0.);

                if let Some(frame) = video_data.next_frame()? {
                    frame.to_tensor(
                        self.image_to_tensor_info,
                        process_options,
                        &mut self.input_buffer,
                    )?;
                    return Ok(Some(self.compute(frame.timestamp_ms())?));
                }
                Ok(None)
            }
        }
    };
}
