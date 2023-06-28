//!
//! # A Rust library for MediaPipe tasks for WasmEdge WASI-NN
//!
//! ## Introduction
//!
//! * **Easy to use**: low-code APIs such as mediapipe-python.
//! * **Low overhead**: No unnecessary data copy, allocation, and free during the processing.
//! * **Flexible**: Users can use custom media bytes as input.
//! * For TfLite models, the library not only supports all models downloaded from [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/)
//!   but also supports **[TF Hub](https://tfhub.dev/)** models and **custom models** with essential information.
//!
//! ## Task APIs
//!
//! Every task has three types: ```XxxBuilder``` -> ```Xxx``` -> ```XxxSession```. (``Xxx`` is the task name)
//!
//! * ```XxxBuilder``` is used to create a task instance ```Xxx```, which has many options to set.
//!
//!   example: use ```ImageClassifierBuilder``` to build a ```ImageClassifier``` task instance.
//!   ```
//!   use mediapipe_rs::tasks::vision::ImageClassifierBuilder;
//!
//!   let classifier = ImageClassifierBuilder::new()
//!         .max_results(3) // set max result
//!         .category_deny_list(vec!["denied label".into()]) // set deny list
//!         .gpu() // set running device
//!         .build_from_file(model_path)?; // create a image classifier
//!   ```
//! * ```Xxx``` is a task instance, which contains task information and model information.
//!
//!   example: use ```ImageClassifier``` to create a new ```ImageClassifierSession```
//!   ```
//!   let classifier_session = classifier.new_session()?;
//!   ```
//! * ```XxxSession``` is a running session to perform pre-process, inference, and post-process, which has buffers to store
//!   mid-results.
//!
//!   example: use ```ImageClassifierSession``` to run the image classification task and return classification results:
//!   ```
//!   let classification_result = classifier_session.classify(&img)?;
//!   ```
//!   **Note**: the session can be reused to speed up, if the code just uses the session once, it can use the task's wrapper
//!   function to simplify.
//!   ```
//!   // let classifier_session = classifier.new_session()?;
//!   // let classification_result = classifier_session.classify(&img)?;
//!   // The above 2-line code is equal to:
//!   let classification_result = classifier.classify(&img)?;
//!   ```
//!
//! ## Available tasks
//! * vision:
//!   * gesture recognition: [`GestureRecognizerBuilder`] -> [`GestureRecognizer`] -> [`GestureRecognizerSession`]
//!   * hand detection: [`HandDetectorBuilder`] -> [`HandDetector`] -> [`HandDetectorSession`]
//!   * image classification: [`ImageClassifierBuilder`] -> [`ImageClassifier`] -> [`ImageClassifierSession`]
//!   * image embedding: [`ImageEmbedderBuilder`] -> [`ImageEmbedder`] -> [`ImageEmbedderSession`]
//!   * image segmentation: [`ImageSegmenterBuilder`] -> [`ImageSegmenter`] -> [`ImageSegmenterSession`]
//!   * object detection: [`ObjectDetectorBuilder`] -> [`ObjectDetector`] -> [`ObjectDetectorSession`]
//! * audio:
//!   * audio classification: [`AudioClassifierBuilder`] -> [`AudioClassifier`] -> [`AudioClassifierSession`]
//! * text:
//!   * text classification: [`TextClassifierBuilder`] -> [`TextClassifier`] -> [`TextClassifierSession`]
//!
//!
//! ## Examples
//!
//! ### Image classification
//!
//! ```rust
//! use mediapipe_rs::tasks::vision::ImageClassifierBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let (model_path, img_path) = parse_args()?;
//!
//!     let classification_result = ImageClassifierBuilder::new()
//!         .max_results(4) // set max result
//!         .build_from_file(model_path)? // create a image classifier
//!         .classify(&image::open(img_path)?)?; // do inference and generate results
//!
//!     //! show formatted result message
//!     println!("{}", classification_result);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Object Detection
//!
//! ```rust
//! use mediapipe_rs::postprocess::utils::draw_detection;
//! use mediapipe_rs::tasks::vision::ObjectDetectorBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let (model_path, img_path, output_path) = parse_args()?;
//!
//!     let mut input_img = image::open(img_path)?;
//!     let detection_result = ObjectDetectorBuilder::new()
//!         .max_results(2) // set max result
//!         .build_from_file(model_path)? // create a object detector
//!         .detect(&input_img)?; // do inference and generate results
//!
//!     // show formatted result message
//!     println!("{}", detection_result);
//!
//!     if let Some(output_path) = output_path {
//!         // draw detection result to image
//!         draw_detection(&mut input_img, &detection_result);
//!         // save output image
//!         input_img.save(output_path)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Text Classification
//! ```rust
//! use mediapipe_rs::tasks::text::TextClassifierBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model_path = parse_args()?;
//!
//!     let text_classifier = TextClassifierBuilder::new()
//!         .max_results(1) // set max result
//!         .build_from_file(model_path)?; // create a text classifier
//!
//!     let positive_str = "I love coding so much!";
//!     let negative_str = "I don't like raining.";
//!
//!     // classify show formatted result message
//!     let result = text_classifier.classify(&positive_str)?;
//!     println!("`{}` -- {}", positive_str, result);
//!
//!     let result = text_classifier.classify(&negative_str)?;
//!     println!("`{}` -- {}", negative_str, result);
//!
//!     Ok(())
//! }
//! ```
//!
//!
//! ## Gesture Recognition
//!
//! ```rust
//! use mediapipe_rs::tasks::vision::GestureRecognizerBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let (model_path, img_path) = parse_args()?;
//!
//!     let gesture_recognition_results = GestureRecognizerBuilder::new()
//!         .num_hands(1) // set only recognition one hand
//!         .max_results(1) // set max result
//!         .build_from_file(model_path)? // create a task instance
//!         .recognize(&image::open(img_path)?)?; // do inference and generate results
//!
//! for g in gesture_recognition_results {
//!         println!("{}", g.gestures.classifications[0].categories[0]);
//!     }
//!
//! Ok(())
//! }
//! ```
//!
//! ### Audio Input
//!
//! Every audio media which implements the trait [`preprocess::audio::AudioData`] can be used as audio tasks input.
//! Now the library has builtin implementation to support ```symphonia```, ```ffmpeg```, and raw audio data as input.
//!
//! Examples for Audio Classification:
//!
//! ```rust
//! use mediapipe_rs::tasks::audio::AudioClassifierBuilder;
//!
//! #[cfg(feature = "ffmpeg")]
//! use mediapipe_rs::preprocess::audio::FFMpegAudioData;
//! #[cfg(not(feature = "ffmpeg"))]
//! use mediapipe_rs::preprocess::audio::SymphoniaAudioData;
//!
//! #[cfg(not(feature = "ffmpeg"))]
//! fn read_audio_using_symphonia(audio_path: String) -> SymphoniaAudioData {
//!     let file = std::fs::File::open(audio_path).unwrap();
//!     let probed = symphonia::default::get_probe()
//!         .format(
//!             &Default::default(),
//!             symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default()),
//!             &Default::default(),
//!             &Default::default(),
//!         )
//!         .unwrap();
//!     let codec_params = &probed.format.default_track().unwrap().codec_params;
//!     let decoder = symphonia::default::get_codecs()
//!         .make(codec_params, &Default::default())
//!         .unwrap();
//!     SymphoniaAudioData::new(probed.format, decoder)
//! }
//!
//! #[cfg(feature = "ffmpeg")]
//! fn read_video_using_ffmpeg(audio_path: String) -> FFMpegAudioData {
//!     ffmpeg_next::init().unwrap();
//!     FFMpegAudioData::new(ffmpeg_next::format::input(&audio_path.as_str()).unwrap()).unwrap()
//! }
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let (model_path, audio_path) = parse_args()?;
//!
//!     #[cfg(not(feature = "ffmpeg"))]
//!     let audio = read_audio_using_symphonia(audio_path);
//!     #[cfg(feature = "ffmpeg")]
//!     let audio = read_video_using_ffmpeg(audio_path);
//!
//!     let classification_results = AudioClassifierBuilder::new()
//!         .max_results(3) // set max result
//!         .build_from_file(model_path)? // create a task instance
//!         .classify(audio)?; // do inference and generate results
//!
//!     // show formatted result message
//!     for c in classification_results {
//!         println!("{}", c);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Use the Session to speed up
//!
//! The session includes inference sessions (such as TfLite interpreter), input and output buffers, etc.
//! Explicitly using the session can reuse these resources to speed up.
//!
//! ### Example: Text Classification
//!
//! Origin:
//! ```rust
//! use mediapipe_rs::tasks::text::TextClassifier;
//! use mediapipe_rs::postprocess::ClassificationResult;
//! use mediapipe_rs::Error;
//!
//! fn inference(
//!     text_classifier: &TextClassifier,
//!     inputs: &Vec<String>
//! ) -> Result<Vec<ClassificationResult>, Error> {
//!     let mut res = Vec::with_capacity(inputs.len());
//!     for input in inputs {
//!         // text_classifier will create new session every time
//!         res.push(text_classifier.classify(input.as_str())?);
//!     }
//!     Ok(res)
//! }
//! ```
//!
//! Use the session to speed up:
//! ```rust
//! use mediapipe_rs::tasks::text::TextClassifier;
//! use mediapipe_rs::postprocess::ClassificationResult;
//! use mediapipe_rs::Error;
//!
//! fn inference(
//!     text_classifier: &TextClassifier,
//!     inputs: &Vec<String>
//! ) -> Result<Vec<ClassificationResult>, Error> {
//!     let mut res = Vec::with_capacity(inputs.len());
//!     // only create one session and reuse the resources in session.
//!     let mut session = text_classifier.new_session()?;
//!     for input in inputs {
//!         res.push(session.classify(input.as_str())?);
//!     }
//!     Ok(res)
//! }
//! ```
//!
//! ## Use the FFMPEG feature to process video and audio.
//!
//! When building the library with ```ffmpeg``` feature using cargo, users must set the following environment variables:
//!
//! * ```FFMPEG_DIR```: the pre-built FFmpeg library path. You can download it from
//!   https://github.com/yanghaku/ffmpeg-wasm32-wasi/releases.
//! * ```WASI_SDK``` or (```WASI_SYSROOT``` and ```CLANG_RT```), You can download it from
//!   https://github.com/WebAssembly/wasi-sdk/releases
//! * ```BINDGEN_EXTRA_CLANG_ARGS```: set **sysroot** and **target** and **function visibility** for libclang.
//!   (The sysroot must be **absolute path**).
//!
//! Example:
//! ```shell
//! export FFMPEG_DIR=/path/to/ffmpeg/library
//! export WASI_SDK=/opt/wasi-sdk
//! export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=/opt/wasi-sdk/share/wasi-sysroot --target=wasm32-wasi -fvisibility=default"
//!
//! # Then run cargo
//! ```
//!
//! ## GPU and TPU support
//!
//! The default device is CPU, and user can use APIs to choose device to use:
//! ```rust
//! use mediapipe_rs::tasks::vision::ObjectDetectorBuilder;
//!
//! fn create_gpu(model_blob: Vec<u8>) {
//!     let detector_gpu = ObjectDetectorBuilder::new()
//!         .gpu()
//!         .build_from_buffer(model_blob)
//!         .unwrap();
//! }
//!
//! fn create_tpu(model_blob: Vec<u8>) {
//!     let detector_tpu = ObjectDetectorBuilder::new()
//!         .tpu()
//!         .build_from_buffer(model_blob)
//!         .unwrap();
//! }
//! ```
//!
//! ## Notice
//! This work is made possible by **Google's work on [Mediapipe](https://github.com/google/mediapipe)**.
//!
//! ## License
//! This project is licensed under the Apache 2.0 license.
//!

#[cfg(not(any(feature = "vision", feature = "audio", feature = "text")))]
compile_error!("Must select at least one task type: `vision`, `audio`, `text`");

mod error;
#[macro_use]
mod model;

/// MediaPipe-rs postprocess api, which define the tasks results and implement tensors results to task results.
/// The module also has utils to make use of results, such as drawing utils.
pub mod postprocess;
/// MediaPipe-rs preprocess api, which define the tasks input interface (convert media input to tensors) and implement some builtin pre-process function for types.
pub mod preprocess;
/// MediaPipe-rs tasks api, contain audio, vision and text tasks.
pub mod tasks;

pub use error::Error;
pub use wasi_nn::ExecutionTarget as Device;
use wasi_nn::{Graph, GraphBuilder, GraphEncoding, GraphExecutionContext, TensorType};

#[cfg(doc)]
use tasks::{audio::*, text::*, vision::*};

/// Re-export the ffmpeg-next crate
#[cfg(feature = "ffmpeg")]
pub use ffmpeg_next;
