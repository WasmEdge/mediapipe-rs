use mediapipe_rs::preprocess::audio::{AudioData, SymphoniaAudioData};
use mediapipe_rs::tasks::audio::AudioClassifierBuilder;
use symphonia::core::io::MediaSourceStream;

const MODEL_1: &'static str =
    "assets/models/audio_classification/yamnet_audio_classifier_with_metadata.tflite";

const AUDIO_PATH: &'static str = "assets/testdata/audio/speech_16000_hz_mono.wav";

#[test]
fn test_audio_classification() {
    // read the audio using symphonia
    let file = std::fs::File::open(AUDIO_PATH).unwrap();
    let probed = symphonia::default::get_probe()
        .format(
            &Default::default(),
            MediaSourceStream::new(Box::new(file), Default::default()),
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
    let codec_params = &probed.format.default_track().unwrap().codec_params;
    let decoder = symphonia::default::get_codecs()
        .make(codec_params, &Default::default())
        .unwrap();
    let input = SymphoniaAudioData::new(probed.format, decoder);

    audio_classification_task_run(MODEL_1, input);
}

#[cfg(feature = "ffmpeg")]
#[test]
fn test_ffmpeg() {
    ffmpeg_next::init().unwrap();
    // read the audio using ffmpeg
    let input = mediapipe_rs::preprocess::audio::FFMpegAudioData::new(
        ffmpeg_next::format::input(&AUDIO_PATH).unwrap(),
    )
    .unwrap();

    audio_classification_task_run(MODEL_1, input);
}

fn audio_classification_task_run(model_asset_path: &str, input: impl AudioData) {
    let classification_list = AudioClassifierBuilder::new()
        .cpu()
        .max_results(1)
        .build_from_file(model_asset_path)
        .unwrap()
        .classify(input)
        .unwrap();
    for classification in &classification_list {
        eprintln!("{}", classification);
    }

    assert_eq!(classification_list.len(), 5);
    assert_eq!(
        classification_list[0].classifications[0].categories[0].index,
        0
    );
}
