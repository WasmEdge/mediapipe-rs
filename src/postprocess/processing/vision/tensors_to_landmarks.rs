// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.cc

use crate::postprocess::{Landmark, Landmarks};

use super::*;

struct ToLandmarksOptions {
    img_size: Option<(f32, f32)>,   // [default = None];
    normalize_z: f32,               // [default = 1.0];
    flip_vertically: bool,          // [default = false];
    flip_horizontally: bool,        // [default = false];
    visibility_score_sigmoid: bool, // [default = false];
    presence_score_sigmoid: bool,   // [default = false];
}

impl Default for ToLandmarksOptions {
    #[inline(always)]
    fn default() -> Self {
        Self {
            img_size: None,
            normalize_z: 1.0,
            flip_vertically: false,
            flip_horizontally: false,
            visibility_score_sigmoid: false,
            presence_score_sigmoid: false,
        }
    }
}

pub(crate) struct TensorsToLandmarks {
    num_landmarks: usize,
    num_dimensions: usize,
    landmark_buffer: OutputBuffer,
    options: ToLandmarksOptions,
}

impl TensorsToLandmarks {
    pub fn new(
        num_landmarks: usize,
        landmark_buf: (TensorType, Option<QuantizationParameters>),
        landmark_shape: &[usize],
    ) -> Result<Self, crate::Error> {
        let mut elem_size = 1; // tensor size, not byte size
        for s in landmark_shape {
            elem_size *= s;
        }
        let num_dimensions = elem_size / num_landmarks;
        if num_dimensions == 0 {
            return Err(crate::Error::ModelInconsistentError(
                format!("Expect tensor element size > num landmarks, but got tensor shape: `{:?}`, num landmarks: `{}",
                        landmark_shape,num_landmarks)
            ));
        }

        Ok(Self {
            num_landmarks,
            num_dimensions,
            landmark_buffer: empty_output_buffer!(landmark_buf, elem_size),
            options: Default::default(),
        })
    }

    #[inline(always)]
    pub fn set_normalize_z(&mut self, normalize_z: f32) {
        self.options.normalize_z = normalize_z;
    }

    #[inline(always)]
    pub fn set_image_size(&mut self, w: u32, h: u32) {
        self.options.img_size = Some((w as f32, h as f32))
    }

    #[inline(always)]
    pub(crate) fn set_flip_vertically(&mut self, flip_vertically: bool) {
        self.options.flip_vertically = flip_vertically;
        assert!(self.options.img_size.is_some());
    }

    #[inline(always)]
    pub(crate) fn set_flip_horizontally(&mut self, flip_horizontally: bool) {
        self.options.flip_horizontally = flip_horizontally;
        assert!(self.options.img_size.is_some());
    }

    #[inline(always)]
    pub(crate) fn landmark_buffer(&mut self) -> &mut [u8] {
        self.landmark_buffer.data_buffer.as_mut_slice()
    }

    pub fn result(&mut self, normalized: bool) -> Landmarks {
        let mut landmark_buf = output_buffer_mut_slice!(self.landmark_buffer);
        let mut landmarks = Vec::with_capacity(self.num_landmarks);

        let mut index = 0;
        let num_dimensions = self.num_dimensions as usize;
        for _ in 0..self.num_landmarks {
            let mut landmark = Landmark {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                visibility: None,
                presence: None,
                name: None,
            };

            if self.options.flip_horizontally {
                landmark.x = self.options.img_size.unwrap().0 - landmark_buf[index];
            } else {
                landmark.x = landmark_buf[index];
            }
            index += 1;

            if num_dimensions > 1 {
                if self.options.flip_vertically {
                    landmark.y = self.options.img_size.unwrap().1 - landmark_buf[index];
                } else {
                    landmark.y = landmark_buf[index];
                }
                index += 1;
            }

            if num_dimensions > 2 {
                landmark.z = landmark_buf[index];
                index += 1;
            }

            if num_dimensions > 3 {
                landmark.visibility = if self.options.visibility_score_sigmoid {
                    let mut v = landmark_buf[index];
                    v.sigmoid_inplace();
                    Some(v)
                } else {
                    Some(landmark_buf[index])
                };
                index += 1;
            }

            if num_dimensions > 4 {
                landmark.presence = if self.options.presence_score_sigmoid {
                    let mut v = landmark_buf[index];
                    v.sigmoid_inplace();
                    Some(v)
                } else {
                    Some(landmark_buf[index])
                };
                index += 1;
            }

            landmarks.push(landmark);
        }

        if normalized {
            let (img_w, img_h) = self.options.img_size.unwrap();
            let normalize_z = self.options.normalize_z;
            for landmark in landmarks.iter_mut() {
                landmark.x /= img_w;
                landmark.y /= img_h;
                landmark.z = landmark.z / normalize_z / img_w;
            }
        }

        Landmarks(landmarks)
    }
}
