fn parse_args() -> Result<(String, String, Option<String>), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        return Err(format!(
            "Usage {} model_path image_path [output image path]",
            args[0]
        )
        .into());
    }
    Ok((args[1].clone(), args[2].clone(), args.get(3).cloned()))
}

use mediapipe_rs::tasks::vision::FaceLandmarkerBuilder;
use mediapipe_rs::postprocess::utils::DrawLandmarksOptions;
use mediapipe_rs::tasks::vision::FaceLandmarkConnections;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, img_path, output_path) = parse_args()?;

    let mut input_img = image::open(img_path)?;
    let face_landmark_results = FaceLandmarkerBuilder::new()
        .num_faces(1) // set max number of faces to detect
        .min_face_detection_confidence(0.5)
        .min_face_presence_confidence(0.5)
        .min_tracking_confidence(0.5)
        .output_face_blendshapes(true)
        .build_from_file(model_path)? // create a face landmarker
        .detect(&input_img)?; // do inference and generate results

    // show formatted result message
    println!("{}", face_landmark_results);

    if let Some(output_path) = output_path {
        // draw face landmarks result to image
        let options = DrawLandmarksOptions::default()
            .connections(FaceLandmarkConnections::get_connections(
                &FaceLandmarkConnections::FacemeshTesselation,
            ))
            .landmark_radius_percent(0.003);

        for result in face_landmark_results.iter() {
            result.draw_with_options(&mut input_img, &options);
        }
        // save output image
        input_img.save(output_path)?;
    }

    Ok(())
}