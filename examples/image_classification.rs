fn parse_args() -> Result<(String, String), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!("Usage {} model_path image_path", args[0]).into());
    }
    Ok((args[1].clone(), args[2].clone()))
}

use mediapipe_rs::tasks::vision::ImageClassifierBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, img_path) = parse_args()?;

    let classification_result = ImageClassifierBuilder::new()
        .model_asset_path(model_path) // set model path
        .max_results(3) // set max result
        .finalize()? // create a image classifier
        .classify(&image::open(img_path)?)?; // do inference and generate results

    // show formatted result message
    println!("{}", classification_result);

    Ok(())
}
