use mediapipe_rs::tasks::text::TextEmbedderBuilder;

const MODEL_1: &'static str = "assets/models/text_embedding/bert_embedder.tflite";
const MODEL_2: &'static str = "assets/models/text_embedding/universal_sentence_encoder.tflite";

#[test]
fn test_text_embedding_model_1() {
    text_embedding_tasks_run(MODEL_1)
}

#[test]
fn test_text_embedding_model_2() {
    // todo: add universal_sentence_encoder input support.
    // text_embedding_tasks_run(MODEL_2)
}

fn text_embedding_tasks_run(model_asset: &str) {
    let text_embedder = TextEmbedderBuilder::new()
        .l2_normalize(false)
        .quantize(false)
        .build_from_file(model_asset)
        .unwrap();
    let mut session = text_embedder.new_session().unwrap();

    let text_1 = "I'm feeling so good";
    let text_2 = "I'm okay I guess";

    let embedding_1 = session.embed(&text_1).unwrap();
    let embedding_2 = session.embed(&text_2).unwrap();
    assert_eq!(embedding_1.embeddings.len(), 1);
    assert_eq!(embedding_2.embeddings.len(), 1);
    let e_1 = embedding_1.embeddings.get(0).unwrap();
    let e_2 = embedding_2.embeddings.get(0).unwrap();
    assert_eq!(e_1.quantized_embedding.len(), 0);
    assert_eq!(e_2.quantized_embedding.len(), 0);
    assert_eq!(e_1.float_embedding.len(), e_2.float_embedding.len());
    assert_ne!(e_1.float_embedding.len(), 0);

    let similarity = e_1.cosine_similarity(e_2).unwrap();
    eprintln!("'{}', '{}' similarity = {}", text_1, text_2, similarity);
}
