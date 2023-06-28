use mediapipe_rs::tasks::vision::GestureRecognizerBuilder;

const MODEL_ASSET: &'static str = "assets/models/gesture_recognition/gesture_recognizer.task";

const THUMBS_UP_IMG: &'static str = "assets/testdata/img/thumbs_up.jpg";
const THUMBS_DOWN_IMG: &'static str = "assets/testdata/img/thumbs_down.jpg";
const VICTORY_IMG: &'static str = "assets/testdata/img/victory.jpg";
const POINTING_UP_IMG: &'static str = "assets/testdata/img/pointing_up.jpg";

#[test]
fn test_gesture_recognition() {
    let gesture_recognizer = GestureRecognizerBuilder::new()
        .num_hands(1)
        .max_results(1)
        .build_from_file(MODEL_ASSET)
        .unwrap();
    let mut gesture_recognizer_session = gesture_recognizer.new_session().unwrap();

    let thumbs_up_img = image::open(THUMBS_UP_IMG).unwrap();
    let thumbs_up_res = gesture_recognizer_session
        .recognize(&thumbs_up_img)
        .unwrap();
    assert_eq!(thumbs_up_res.len(), 1);
    let gestures = &thumbs_up_res.get(0).unwrap().gestures;
    eprintln!("{}", gestures);

    let thumbs_down_img = image::open(THUMBS_DOWN_IMG).unwrap();
    let thumbs_down_res = gesture_recognizer_session
        .recognize(&thumbs_down_img)
        .unwrap();
    assert_eq!(thumbs_down_res.len(), 1);
    let gestures = &thumbs_down_res.get(0).unwrap().gestures;
    eprintln!("{}", gestures);

    let victory_img = image::open(VICTORY_IMG).unwrap();
    let victory_res = gesture_recognizer_session.recognize(&victory_img).unwrap();
    assert_eq!(victory_res.len(), 1);
    let gestures = &victory_res.get(0).unwrap().gestures;
    eprintln!("{}", gestures);

    let pointing_up_img = image::open(POINTING_UP_IMG).unwrap();
    let pointing_up_res = gesture_recognizer_session
        .recognize(&pointing_up_img)
        .unwrap();
    assert_eq!(pointing_up_res.len(), 1);
    let gestures = &pointing_up_res.get(0).unwrap().gestures;
    eprintln!("{}", gestures);

    let draw = false;
    if draw {
        draw_hand_landmarks(thumbs_up_img, thumbs_up_res, "./target/thumbs_up.jpg");
        draw_hand_landmarks(thumbs_down_img, thumbs_down_res, "./target/thumbs_down.jpg");
        draw_hand_landmarks(victory_img, victory_res, "./target/victory.jpg");
        draw_hand_landmarks(pointing_up_img, pointing_up_res, "./target/pointing_up.jpg");
    }
}

#[allow(unused)]
fn draw_hand_landmarks(
    mut img: image::DynamicImage,
    res: mediapipe_rs::tasks::vision::results::GestureRecognizerResults,
    out: &str,
) {
    let options = mediapipe_rs::postprocess::utils::DrawLandmarksOptions::default()
        .connections(mediapipe_rs::tasks::vision::HandLandmark::CONNECTIONS);
    mediapipe_rs::postprocess::utils::draw_landmarks_with_options(
        &mut img,
        &res.get(0).unwrap().hand_landmark.hand_landmarks,
        &options,
    );
    img.save(out).unwrap();
}
