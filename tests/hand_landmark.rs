use mediapipe_rs::tasks::vision::HandLandmarkerBuilder;

const MODEL_PATH: &'static str = "assets/models/hand_landmark_detection/hand_landmarker.task";
const HANDS_1: &'static str = "assets/testdata/img/woman_hands.jpg";

#[test]
fn test_hand_detection() {
    let hand_landmarker = HandLandmarkerBuilder::new()
        .num_hands(5)
        .build_from_file(MODEL_PATH)
        .unwrap();
    let hand_detector = hand_landmarker.subtask_hand_detector();
    let img = image::open(HANDS_1).unwrap();
    let detection_result = hand_detector.detect(&img).unwrap();
    assert_eq!(detection_result.detections.len(), 2);
    eprintln!("{}", detection_result);
}

#[test]
fn test_hand_landmark() {
    let img = image::open(HANDS_1).unwrap();
    let hand_landmark_results = HandLandmarkerBuilder::new()
        .cpu()
        .num_hands(10)
        .build_from_file(MODEL_PATH)
        .unwrap()
        .detect(&img)
        .unwrap();
    assert_eq!(hand_landmark_results.len(), 2);
    assert_eq!(
        hand_landmark_results[0]
            .handedness
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "Right"
    );
    assert_eq!(
        hand_landmark_results[1]
            .handedness
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "Left"
    );
    eprintln!("{}", hand_landmark_results);

    let draw = false;
    if draw {
        draw_hand_landmarks(
            img,
            hand_landmark_results,
            "./target/hand_landmark_test.jpg",
        );
    }
}

#[allow(unused)]
fn draw_hand_landmarks(
    mut img: image::DynamicImage,
    hand_landmark_results: mediapipe_rs::tasks::vision::results::HandLandmarkResults,
    path: &str,
) {
    let options = mediapipe_rs::postprocess::utils::DrawLandmarksOptions::default()
        .connections(mediapipe_rs::tasks::vision::HandLandmark::CONNECTIONS);
    for r in hand_landmark_results.iter() {
        mediapipe_rs::postprocess::utils::draw_landmarks_with_options(
            &mut img,
            &r.hand_landmarks,
            &options,
        );
    }
    img.save(path).unwrap();
}
