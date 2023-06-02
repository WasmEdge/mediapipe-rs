#![allow(non_snake_case)]
#![allow(unused_imports)]

mod metadata_schema_generated;
mod schema_generated;

pub(super) use metadata_schema_generated::tflite as tflite_metadata;
pub(super) use schema_generated::tflite;

#[cfg(feature = "vision")]
mod image_segmenter_metadata_schema_generated;
#[cfg(feature = "vision")]
pub(super) use image_segmenter_metadata_schema_generated::mediapipe::tasks as custom_img_segmentation;
#[cfg(feature = "vision")]
pub(super) const CUSTOM_SEGMENTATION_METADATA_NAME: &'static str = "SEGMENTER_METADATA";
