use mediapipe_rs::tasks::vision::FaceDetectorBuilder;

const MODEL_1: &'static str = "assets/models/face_detection/face_detection_short_range.tflite";

const FACE_IMG_1: &'static str = "assets/testdata/img/face.jpg";

#[test]
fn test_face_detection_model_1() {
    face_detection_task_run(MODEL_1)
}

fn face_detection_task_run(model_path: &str) {
    let img = image::open(FACE_IMG_1).unwrap();
    let face_detection_result = FaceDetectorBuilder::new()
        .build_from_file(model_path)
        .unwrap()
        .detect(&img)
        .unwrap();
    eprintln!("{}", face_detection_result);

    let draw = false;
    if draw {
        let path = std::path::PathBuf::from(model_path);
        let out_file = path.file_stem().to_owned().unwrap().to_str().unwrap();
        draw_face_detection(
            img,
            face_detection_result,
            format!("./target/{}.jpg", out_file).as_str(),
        );
    }
}

fn draw_face_detection(
    mut img: image::DynamicImage,
    det: mediapipe_rs::postprocess::DetectionResult,
    save_path: &str,
) {
    mediapipe_rs::postprocess::utils::draw_detection(&mut img, &det);
    img.save(save_path).unwrap();
}
