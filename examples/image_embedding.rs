fn parse_args() -> Result<(String, String, String), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err(format!("Usage {} model_path image_1_path image_2_path", args[0]).into());
    }
    Ok((args[1].clone(), args[2].clone(), args[3].clone()))
}

use mediapipe_rs::tasks::vision::ImageEmbedderBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, img_1, img_2) = parse_args()?;

    let task = ImageEmbedderBuilder::new()
        .l2_normalize(true)
        .quantize(true)
        .build_from_file(model_path)?; // create a task instance
    let mut session = task.new_session()?; // create a new session to perform task

    // do inference and generate results
    let res_1 = session.embed(&image::open(img_1)?)?;
    let res_2 = session.embed(&image::open(img_2)?)?;
    eprintln!(
        "Cosine Similarity = {}",
        res_1.embeddings[0].cosine_similarity(&res_2.embeddings[0])?
    );
    Ok(())
}
