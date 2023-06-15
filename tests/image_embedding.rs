use mediapipe_rs::tasks::vision::ImageEmbedderBuilder;

const MODEL_1: &'static str =
    "assets/models/image_embedding/mobilenet_v3_large_075_224_embedder.tflite";
const MODEL_2: &'static str =
    "assets/models/image_embedding/mobilenet_v3_small_075_224_embedder.tflite";

const IMG_1: &'static str = "assets/testdata/img/burger.jpg";
const IMG_2: &'static str = "assets/testdata/img/burger_crop.jpg";

#[test]
fn test_image_embedding_model_1() {
    image_embedding_tasks_run(MODEL_1)
}

#[test]
fn test_image_embedding_model_2() {
    image_embedding_tasks_run(MODEL_2)
}

fn image_embedding_tasks_run(model_asset: &str) {
    let image_embedder = ImageEmbedderBuilder::new()
        .l2_normalize(true)
        .quantize(true)
        .build_from_file(model_asset)
        .unwrap();
    let mut session = image_embedder.new_session().unwrap();

    let embedding_1 = session.embed(&image::open(IMG_1).unwrap()).unwrap();
    let embedding_2 = session.embed(&image::open(IMG_2).unwrap()).unwrap();
    assert_eq!(embedding_1.embeddings.len(), 1);
    assert_eq!(embedding_2.embeddings.len(), 1);
    let e_1 = embedding_1.embeddings.get(0).unwrap();
    let e_2 = embedding_2.embeddings.get(0).unwrap();
    assert_eq!(e_1.float_embedding.len(), 0);
    assert_eq!(e_2.float_embedding.len(), 0);
    assert_eq!(e_1.quantized_embedding.len(), e_2.quantized_embedding.len());
    assert_ne!(e_1.quantized_embedding.len(), 0);

    let similarity = e_1.cosine_similarity(e_2).unwrap();
    eprintln!("similarity = {}", similarity);
}
