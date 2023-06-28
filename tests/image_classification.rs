use mediapipe_rs::tasks::vision::{ImageClassifierBuilder, ImageProcessingOptions};

const MODEL_1: &'static str = "assets/models/image_classification/efficientnet_lite0_fp32.tflite";
const MODEL_2: &'static str = "assets/models/image_classification/efficientnet_lite0_uint8.tflite";
const MODEL_3: &'static str = "assets/models/image_classification/efficientnet_lite2_fp32.tflite";
const MODEL_4: &'static str = "assets/models/image_classification/efficientnet_lite2_uint8.tflite";
const IMG: &'static str = "assets/testdata/img/burger.jpg";
const CAT_AND_DOG_IMG: &'static str = "assets/testdata/img/cat_and_dog.jpg";

#[test]
fn test_image_classification_model_1() {
    image_classification_task_run(MODEL_1);
}

#[test]
fn test_image_classification_model_2() {
    image_classification_task_run(MODEL_2);
}

#[test]
fn test_image_classification_model_3() {
    image_classification_task_run(MODEL_3);
}

#[test]
fn test_image_classification_model_4() {
    image_classification_task_run(MODEL_4);
}

fn image_classification_task_run(model_asset_path: &str) {
    let image_classifier = ImageClassifierBuilder::new()
        .cpu()
        .max_results(2)
        .build_from_file(model_asset_path)
        .unwrap();

    let res = image_classifier
        .classify(&image::open(IMG).unwrap())
        .unwrap();
    eprintln!("{}", res);
    // cheeseburger: 933
    let top = res
        .classifications
        .get(0)
        .unwrap()
        .categories
        .get(0)
        .unwrap();
    assert_eq!(top.index, 933);
    assert_eq!(top.category_name.as_ref().unwrap().as_str(), "cheeseburger");
}

#[test]
fn test_bird_from_tf_hub() {
    const MODEL: &'static str =
        "assets/models/image_classification/lite-model_aiy_vision_classifier_birds_V1_3.tflite";
    const IMAGE: &'static str = "assets/testdata/img/bird.jpg";

    let res = ImageClassifierBuilder::new()
        .max_results(2)
        .build_from_file(MODEL)
        .unwrap()
        .classify(&image::open(IMAGE).unwrap())
        .unwrap();
    assert_eq!(
        res.classifications[0].categories[0]
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "/m/01bwb9"
    );
    assert_eq!(
        res.classifications[0].categories[0]
            .display_name
            .as_ref()
            .unwrap()
            .as_str(),
        "Passer domesticus"
    );
    eprintln!("{}", res);
}

#[test]
fn test_classify_with_options() {
    let classifier = ImageClassifierBuilder::new()
        .max_results(1)
        .build_from_file(MODEL_1)
        .unwrap();
    let mut session = classifier.new_session().unwrap();
    let img = image::open(CAT_AND_DOG_IMG).unwrap();

    // region of interest is left (cat)
    let cat_res = session
        .classify_with_options(
            &img,
            &ImageProcessingOptions::new()
                .region_of_interest(0.1, 0.35, 0.55, 0.9)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        cat_res.classifications[0].categories[0]
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "Egyptian cat"
    );

    // region of interest is right (dog)
    let dog_res = session
        .classify_with_options(
            &img,
            &ImageProcessingOptions::new()
                .region_of_interest(0.45, 0., 1., 0.9)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        dog_res.classifications[0].categories[0]
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "bull mastiff"
    );
}
