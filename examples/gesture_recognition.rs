fn parse_args() -> Result<(String, String), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!("Usage {} model_path image_path", args[0]).into());
    }
    Ok((args[1].clone(), args[2].clone()))
}

use mediapipe_rs::tasks::vision::GestureRecognizerBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, img_path) = parse_args()?;

    let gesture_recognition_results = GestureRecognizerBuilder::new()
        .num_hands(1) // set only recognition one hand
        .max_results(1) // set max result
        .build_from_file(model_path)? // create a task instance
        .recognize(&image::open(img_path)?)?; // do inference and generate results

    for g in gesture_recognition_results {
        println!("{}", g.gestures.classifications[0].categories[0]);
    }

    Ok(())
}
