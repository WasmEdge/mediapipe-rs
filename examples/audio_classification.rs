fn parse_args() -> Result<(String, String), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!("Usage {} model_path audio_path", args[0]).into());
    }
    Ok((args[1].clone(), args[2].clone()))
}

use mediapipe_rs::tasks::audio::AudioClassifierBuilder;

#[cfg(feature = "ffmpeg")]
use mediapipe_rs::preprocess::audio::FFMpegAudioData;
#[cfg(not(feature = "ffmpeg"))]
use mediapipe_rs::preprocess::audio::SymphoniaAudioData;

#[cfg(not(feature = "ffmpeg"))]
fn read_audio_using_symphonia(audio_path: String) -> SymphoniaAudioData {
    let file = std::fs::File::open(audio_path).unwrap();
    let probed = symphonia::default::get_probe()
        .format(
            &Default::default(),
            symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default()),
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
    let codec_params = &probed.format.default_track().unwrap().codec_params;
    let decoder = symphonia::default::get_codecs()
        .make(codec_params, &Default::default())
        .unwrap();
    SymphoniaAudioData::new(probed.format, decoder)
}

#[cfg(feature = "ffmpeg")]
fn read_video_using_ffmpeg(audio_path: String) -> FFMpegAudioData {
    ffmpeg_next::init().unwrap();
    FFMpegAudioData::new(ffmpeg_next::format::input(&audio_path.as_str()).unwrap()).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, audio_path) = parse_args()?;

    #[cfg(not(feature = "ffmpeg"))]
    let audio = read_audio_using_symphonia(audio_path);
    #[cfg(feature = "ffmpeg")]
    let audio = read_video_using_ffmpeg(audio_path);

    let classification_results = AudioClassifierBuilder::new()
        .max_results(3) // set max result
        .build_from_file(model_path)? // use model file to create task instance
        .classify(audio)?; // do inference and generate results

    // show formatted result message
    for c in classification_results {
        println!("{}", c);
    }

    Ok(())
}
