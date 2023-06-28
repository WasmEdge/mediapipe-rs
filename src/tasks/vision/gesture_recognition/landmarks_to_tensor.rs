// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.cc

use super::*;

pub fn landmarks_to_tensor(
    landmarks: &Landmarks,
    tensor_buffer: &mut impl AsMut<[f32]>,
    img_size: (u32, u32),
    object_normalization_origin_offset: usize,
) {
    let max = std::cmp::max(img_size.0, img_size.1) as f32;
    let width_scale_factor = img_size.0 as f32 / max;
    let height_scale_factor = img_size.1 as f32 / max;

    let origin = &landmarks[object_normalization_origin_offset];
    let origin_x = origin.x;
    let origin_y = origin.y;
    let origin_z = origin.z;
    let scale = get_normalized_object_scale(landmarks);

    let buf = tensor_buffer.as_mut();
    let mut index = 0;

    for l in landmarks.iter() {
        let x = (l.x - 0.5) * width_scale_factor + 0.5;
        let y = (l.y - 0.5) * height_scale_factor + 0.5;
        let z = l.z;

        buf[index] = (x - origin_x) / scale;
        buf[index + 1] = (y - origin_y) / scale;
        buf[index + 2] = (z - origin_z) / scale;
        index += 3;
    }
}

fn get_normalized_object_scale(landmarks: &Landmarks) -> f32 {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for l in landmarks.iter() {
        if l.x < min_x {
            min_x = l.x;
        }
        if l.x > max_x {
            max_x = l.x;
        }
        if l.y < min_y {
            min_y = l.y;
        }
        if l.y > max_y {
            max_y = l.y;
        }
    }
    const EPSILON: f32 = 1e-5;
    let x = max_x - min_x;
    let y = max_y - min_y;
    if x > y {
        x + EPSILON
    } else {
        y + EPSILON
    }
}

pub fn world_landmarks_to_tensor(landmarks: &Landmarks, tensor_buffer: &mut impl AsMut<[f32]>) {
    let buf = tensor_buffer.as_mut();
    let mut index = 0;
    for l in landmarks.iter() {
        buf[index] = l.x;
        buf[index + 1] = l.y;
        buf[index + 2] = l.z;
        index += 3;
    }
}
