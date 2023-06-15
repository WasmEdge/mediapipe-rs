use mediapipe_rs::tasks::vision::ObjectDetectorBuilder;

const MODEL_1: &'static str = "assets/models/object_detection/efficientdet_lite0_fp32.tflite";
const MODEL_2: &'static str = "assets/models/object_detection/efficientdet_lite0_uint8.tflite";
const MODEL_3: &'static str = "assets/models/object_detection/efficientdet_lite2_fp32.tflite";
const MODEL_4: &'static str = "assets/models/object_detection/efficientdet_lite2_uint8.tflite";
const MODEL_5: &'static str = "assets/models/object_detection/mobilenetv2_ssd_256_fp32.tflite";
const MODEL_6: &'static str = "assets/models/object_detection/mobilenetv2_ssd_256_uint8.tflite";
const IMG: &'static str = "assets/testdata/img/cat_and_dog.jpg";

#[test]
fn test_object_detection_model_1() {
    object_detection_task_run(MODEL_1);
}

#[test]
fn test_object_detection_model_2() {
    object_detection_task_run(MODEL_2);
}

#[test]
fn test_object_detection_model_3() {
    object_detection_task_run(MODEL_3);
}

#[test]
fn test_object_detection_model_4() {
    object_detection_task_run(MODEL_4);
}

#[test]
fn test_object_detection_model_5() {
    object_detection_task_run(MODEL_5);
}

#[test]
fn test_object_detection_model_6() {
    object_detection_task_run(MODEL_6);
}

fn object_detection_task_run(model_asset_path: &str) {
    let object_detector = ObjectDetectorBuilder::new()
        .cpu()
        .max_results(5)
        .build_from_file(model_asset_path)
        .unwrap();

    let img = image::open(IMG).unwrap();
    let mut session = object_detector.new_session().unwrap();
    let res = session.detect(&img).unwrap();
    eprintln!("{}", res);
}

#[test]
fn test_allow_deny_list() {
    let res = ObjectDetectorBuilder::new()
        .cpu()
        .max_results(1)
        .category_deny_list(vec!["dog".into()])
        .build_from_file(MODEL_1)
        .unwrap()
        .detect(&image::open(IMG).unwrap())
        .unwrap();
    assert_eq!(
        res.detections[0].categories[0]
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "cat"
    );

    let res = ObjectDetectorBuilder::new()
        .cpu()
        .max_results(1)
        .category_allow_list(vec!["dog".into()])
        .build_from_file(MODEL_1)
        .unwrap()
        .detect(&image::open(IMG).unwrap())
        .unwrap();
    assert_eq!(
        res.detections[0].categories[0]
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "dog"
    );
}
