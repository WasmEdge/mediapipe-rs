#[cfg(feature = "ffmpeg")]
mod ffmpeg {
    use mediapipe_rs::preprocess::vision::FFMpegVideoData;
    use mediapipe_rs::tasks::vision::{
        ImageClassifierBuilder, ImageProcessingOptions, ObjectDetectorBuilder,
    };

    const IMAGE_CLASSIFICATION_MODEL: &'static str =
        "assets/models/image_classification/efficientnet_lite0_fp32.tflite";
    const OBJECT_DETECTION_MODEL: &'static str =
        "assets/models/object_detection/efficientdet_lite0_fp32.tflite";

    const VIDEO_1: &'static str = "assets/testdata/video/bird_burger_tabby.mp4";

    const FRAME_CLASSIFY_CATEGORIES: &[&'static str] = &["junco", "cheeseburger", "tabby"];
    const FRAME_DETECTION_CATEGORIES: &[&'static str] = &["bird", "sandwich", "cat"];

    #[test]
    fn test_image_classification() {
        ffmpeg_next::init().unwrap();
        let input = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();

        let classification_results = ImageClassifierBuilder::new()
            .max_results(1)
            .build_from_file(IMAGE_CLASSIFICATION_MODEL)
            .unwrap()
            .classify_for_video(input)
            .unwrap();
        assert_eq!(classification_results.len(), 3);
        for i in 0..3 {
            assert_eq!(
                classification_results[i].classifications[0].categories[0].category_name,
                Some(FRAME_CLASSIFY_CATEGORIES[i].into())
            );
        }
    }

    #[test]
    fn test_object_detection() {
        ffmpeg_next::init().unwrap();
        let input = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();

        let detection_results = ObjectDetectorBuilder::new()
            .max_results(1)
            .build_from_file(OBJECT_DETECTION_MODEL)
            .unwrap()
            .detect_for_video(input)
            .unwrap();
        assert_eq!(detection_results.len(), 3);
        for i in 0..3 {
            assert_eq!(
                detection_results[i].detections[0].categories[0].category_name,
                Some(FRAME_DETECTION_CATEGORIES[i].into())
            );
        }
        for result in detection_results {
            eprintln!("{}", result);
        }
    }

    #[test]
    fn test_results_iter() {
        ffmpeg_next::init().unwrap();
        let input_1 = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();
        let input_2 = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();

        let classifier = ImageClassifierBuilder::new()
            .max_results(1)
            .build_from_file(IMAGE_CLASSIFICATION_MODEL)
            .unwrap();
        let mut session = classifier.new_session().unwrap();

        let mut results_iter = session.classify_for_video(input_1).unwrap();
        let mut num_frame = 0;
        while let Some(result) = results_iter.next().unwrap() {
            eprintln!("Frame {}: {}", num_frame, result);
            num_frame += 1;
        }

        // test with region of interest options
        let mut results_iter = session.classify_for_video(input_2).unwrap();
        num_frame = 0;
        let mut img_options = ImageProcessingOptions::new()
            .region_of_interest(0.1, 0.1, 0.9, 0.9)
            .unwrap();
        while let Some(result) = results_iter.next_with_options(&img_options).unwrap() {
            eprintln!("Frame {}: {}", num_frame, result);
            num_frame += 1;
            img_options = img_options.rotation_degrees(num_frame * 90).unwrap();
        }
    }
}
