use mediapipe_rs::postprocess::utils::DrawLandmarksOptions;
use mediapipe_rs::tasks::vision::results::FaceLandmarkResults;
use mediapipe_rs::tasks::vision::FaceLandmarkConnections;
use mediapipe_rs::tasks::vision::FaceLandmarkerBuilder;

const MODEL_PATH: &'static str = "assets/models/face_landmark/face_landmarker.task";
const FACE_1: &'static str = "assets/testdata/img/face.jpg";

#[test]
fn test_face_detection() {
    let face_landmarker = FaceLandmarkerBuilder::new()
        .num_faces(5)
        .build_from_file(MODEL_PATH)
        .unwrap();
    let face_detector = face_landmarker.subtask_face_detector();
    let img = image::open(FACE_1).unwrap();
    let detection_result = face_detector.detect(&img).unwrap();
    eprintln!("{}", detection_result);
}

#[test]
fn test_face_landmark() {
    let img = image::open(FACE_1).unwrap();
    let face_landmark_results = FaceLandmarkerBuilder::new()
        .cpu()
        .num_faces(1)
        .build_from_file(MODEL_PATH)
        .unwrap()
        .detect(&img)
        .unwrap();

    eprintln!("{}", face_landmark_results);

    let draw = false;
    if draw {
        draw_face_landmarks(
            img,
            face_landmark_results,
            "./target/face_landmark_test.jpg",
        );
    }
}

fn draw_face_landmarks(
    mut img: image::DynamicImage,
    face_landmark_results: FaceLandmarkResults,
    path: &str,
) {
    let options = DrawLandmarksOptions::default()
        .connections(FaceLandmarkConnections::get_connections(
            &FaceLandmarkConnections::FacemeshTesselation,
        ))
        .landmark_radius_percent(0.003);
    for r in face_landmark_results.iter() {
        mediapipe_rs::postprocess::utils::draw_landmarks_with_options(
            &mut img,
            &r.face_landmarks,
            &options,
        );
    }
    img.save(path).unwrap();
}
