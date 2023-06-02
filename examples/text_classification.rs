use mediapipe_rs::tasks::text::TextClassifierBuilder;

fn parse_args() -> Result<String, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err(format!("Usage {} model_path", args[0]).into());
    }
    Ok(args[1].clone())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = parse_args()?;

    let text_classifier = TextClassifierBuilder::new()
        .model_asset_path(model_path) // set model path
        .max_results(1) // set max result
        .finalize()?; // create a text classifier

    let positive_str = "I love coding so much!";
    let negative_str = "I don't like raining.";

    // classify show formatted result message
    let result = text_classifier.classify(&positive_str)?;
    println!("`{}` -- {}", positive_str, result);

    let result = text_classifier.classify(&negative_str)?;
    println!("`{}` -- {}", negative_str, result);

    Ok(())
}
