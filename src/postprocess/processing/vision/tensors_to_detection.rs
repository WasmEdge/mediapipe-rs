// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc

use super::*;
use crate::postprocess::{
    CategoriesFilter, Category, Detection, DetectionResult, NormalizedKeypoint, Rect,
};

/// Tells the calculator how to convert the detector output to bounding boxes.
#[derive(Debug, Clone, Copy)]
pub enum DetectionBoxFormat {
    /// bbox [y_center, x_center, height, width], keypoint [y, x]
    YXHW,
    /// bbox [x_center, y_center, width, height], keypoint [x, y]
    XYWH,
    /// bbox [xmin, ymin, xmax, ymax], keypoint [x, y]
    XYXY,
}

impl Default for DetectionBoxFormat {
    // if UNSPECIFIED, the calculator assumes YXHW
    fn default() -> Self {
        Self::YXHW
    }
}

struct ToDetectionOptions {
    /// The number of output classes predicted by the detection model.
    /// if categories buffer is not None, num_classes must be 1
    num_classes: usize,
    /// The number of output values per boxes predicted by the detection model.
    /// The values contain bounding boxes, key points, etc.
    num_coords: usize,
    /// The offset of keypoint coordinates in the location tensor.
    keypoint_coord_offset: usize,
    /// The number of predicted key points.
    num_key_points: usize, // [default = 0]
    /// The dimension of each keypoint, e.g. number of values predicted for each keypoint.
    num_values_per_key_point: usize, // [default = 2]
    /// The offset of box coordinates in the location tensor.
    box_coord_offset: usize, // [default = 0]
    /// Parameters for decoding SSD detection model.
    x_scale: f32, // [default = 0.0];
    y_scale: f32, // [default = 0.0];
    w_scale: f32, // [default = 0.0];
    h_scale: f32, // [default = 0.0];
    /// Represents the bounding box by using the combination of boundaries, {ymin, xmin, ymax, xmax}.
    /// The default order is {ymin, xmin, ymax, xmax}.
    box_indices: [usize; 4],
    box_format: DetectionBoxFormat,
    min_score_threshold: f32, // used if categories filter is none
    score_clipping_thresh: Option<f32>,
    apply_exponential_on_box_size: bool, // [default = false]
    sigmoid_score: bool,                 // [default = false];
    /// Whether the detection coordinates from the input tensors should be flipped vertically (along the y-direction).
    flip_vertically: bool, //[default = false];
}

macro_rules! box_y_min {
    ($options:ident, $v:ident) => {
        $v[$options.box_indices[0] + $options.box_coord_offset]
    };
}

macro_rules! box_x_min {
    ($options:ident, $v:ident) => {
        $v[$options.box_indices[1] + $options.box_coord_offset]
    };
}

macro_rules! box_y_max {
    ($options:ident, $v:ident) => {
        $v[$options.box_indices[2] + $options.box_coord_offset]
    };
}

macro_rules! box_x_max {
    ($options:ident, $v:ident) => {
        $v[$options.box_indices[3] + $options.box_coord_offset]
    };
}

macro_rules! check_options_valid {
    ( $self:expr ) => {
        assert!(
            $self.num_coords
                >= $self.box_coord_offset
                    + $self.keypoint_coord_offset
                    + $self.num_key_points * $self.num_values_per_key_point
        );
    };
}

macro_rules! process_scores {
    ( $self:ident, $score:expr ) => {
        if $self.options.sigmoid_score {
            if let Some(t) = $self.options.score_clipping_thresh {
                let mut s = $score;
                if s < -t {
                    s = -t;
                }
                if s > t {
                    s = t;
                }
                s.sigmoid_inplace();
                s
            } else {
                let mut s = $score;
                s.sigmoid_inplace();
                s
            }
        } else {
            $score
        }
    };
}

impl Default for ToDetectionOptions {
    fn default() -> Self {
        Self {
            box_format: Default::default(),
            box_indices: [0, 1, 2, 3],
            num_classes: 1,
            num_coords: 4,
            num_key_points: 0,
            num_values_per_key_point: 2,
            box_coord_offset: 0,
            keypoint_coord_offset: 4,
            x_scale: 0.0,
            y_scale: 0.0,
            w_scale: 0.0,
            h_scale: 0.0,
            min_score_threshold: 0., // default is 0
            score_clipping_thresh: None,
            apply_exponential_on_box_size: false,
            sigmoid_score: false,
            flip_vertically: false,
        }
    }
}

pub(crate) struct TensorsToDetection<'a> {
    anchors: Option<&'a Vec<Anchor>>,
    nms: NonMaxSuppression,
    categories_filter: CategoriesFilter<'a>,

    location_buf: OutputBuffer,
    score_buf: OutputBuffer,
    categories_buf: Option<OutputBuffer>,
    options: ToDetectionOptions,
}

impl<'a> TensorsToDetection<'a> {
    pub(crate) fn new_with_anchors(
        categories_filter: CategoriesFilter<'a>,
        anchors: &'a Vec<Anchor>,
        min_score_threshold: f32,
        max_results: i32,
        location_buf: (TensorType, Option<QuantizationParameters>),
        score_buf: (TensorType, Option<QuantizationParameters>),
    ) -> Self {
        let mut options = ToDetectionOptions::default();
        options.min_score_threshold = min_score_threshold;
        Self {
            nms: NonMaxSuppression::new(max_results),
            categories_filter,
            location_buf: empty_output_buffer!(location_buf),
            score_buf: empty_output_buffer!(score_buf),
            anchors: Some(anchors),
            categories_buf: None,
            options,
        }
    }

    #[inline]
    pub(crate) fn new(
        categories_filter: CategoriesFilter<'a>,
        max_results: i32,
        location_buf: (TensorType, Option<QuantizationParameters>),
        categories_buf: (TensorType, Option<QuantizationParameters>),
        score_buf: (TensorType, Option<QuantizationParameters>),
    ) -> Self {
        Self {
            anchors: None,
            nms: NonMaxSuppression::new(max_results),
            categories_filter,
            location_buf: empty_output_buffer!(location_buf),
            score_buf: empty_output_buffer!(score_buf),
            categories_buf: Some(empty_output_buffer!(categories_buf)),
            options: Default::default(),
        }
    }

    pub(crate) fn set_box_indices(&mut self, bound_box_properties: &[usize; 4]) {
        let box_indices = [
            bound_box_properties[1], // y_min
            bound_box_properties[0], // x_min
            bound_box_properties[3], // y_max
            bound_box_properties[2], // x_max
        ];
        for i in &box_indices {
            assert!(*i < 4);
        }
        self.options.box_indices = box_indices;
    }

    #[inline(always)]
    pub(crate) fn set_num_coords(&mut self, num_coords: usize) {
        self.options.num_coords = num_coords;
        check_options_valid!(self.options);
    }

    #[inline(always)]
    pub(crate) fn set_key_points(
        &mut self,
        num_key_points: usize,
        num_values_per_key_point: usize,
        keypoint_coord_offset: usize,
    ) {
        self.options.num_key_points = num_key_points;
        self.options.num_values_per_key_point = num_values_per_key_point;
        self.options.keypoint_coord_offset = keypoint_coord_offset;
        check_options_valid!(self.options);
    }

    #[inline(always)]
    pub(crate) fn set_anchors_scales(
        &mut self,
        x_scale: f32,
        y_scale: f32,
        w_scale: f32,
        h_scale: f32,
    ) {
        self.options.x_scale = x_scale;
        self.options.y_scale = y_scale;
        self.options.w_scale = w_scale;
        self.options.h_scale = h_scale;
    }

    #[inline(always)]
    pub(crate) fn set_sigmoid_score(&mut self, sigmoid_score: bool) {
        self.options.sigmoid_score = sigmoid_score;
    }

    #[inline(always)]
    pub(crate) fn set_box_format(&mut self, box_format: DetectionBoxFormat) {
        self.options.box_format = box_format;
    }

    #[inline(always)]
    pub(crate) fn set_score_clipping_thresh(&mut self, score_clipping_thresh: f32) {
        self.options.score_clipping_thresh = Some(score_clipping_thresh);
    }

    #[inline(always)]
    pub(crate) fn set_nms_overlap_type(&mut self, overlap_type: NonMaxSuppressionOverlapType) {
        self.nms.set_overlap_type(overlap_type);
    }

    #[inline(always)]
    pub(crate) fn set_nms_algorithm(&mut self, algorithm: NonMaxSuppressionAlgorithm) {
        self.nms.set_algorithm(algorithm);
    }

    #[inline(always)]
    pub(crate) fn set_nms_min_suppression_threshold(&mut self, min_suppression_threshold: f32) {
        self.nms
            .set_min_suppression_threshold(min_suppression_threshold);
    }

    #[inline(always)]
    pub(crate) fn location_buf(&mut self) -> &mut [u8] {
        self.location_buf.data_buffer.as_mut_slice()
    }

    #[inline(always)]
    pub(crate) fn categories_buf(&mut self) -> Option<&mut [u8]> {
        if let Some(ref mut c) = self.categories_buf {
            return Some(c.data_buffer.as_mut_slice());
        }
        None
    }

    #[inline(always)]
    pub(crate) fn score_buf(&mut self) -> &mut [u8] {
        self.score_buf.data_buffer.as_mut_slice()
    }

    #[inline(always)]
    pub(crate) fn realloc(&mut self, num_boxes: usize) {
        realloc_output_buffer!(self.score_buf, num_boxes * self.options.num_classes);
        if let Some(ref mut c) = self.categories_buf {
            realloc_output_buffer!(c, num_boxes);
        }
        realloc_output_buffer!(self.location_buf, num_boxes * num_boxes);
    }

    pub(crate) fn result(&mut self, num_boxes: usize) -> DetectionResult {
        let scores = output_buffer_mut_slice!(self.score_buf);
        let location = output_buffer_mut_slice!(self.location_buf);

        // check buf if is valid
        debug_assert!(location.len() >= num_boxes * self.options.num_coords);
        debug_assert!(scores.len() >= num_boxes * self.options.num_classes);
        if let Some(a) = self.anchors {
            debug_assert_eq!(a.len(), num_boxes);
        }

        let mut detections = Vec::with_capacity(num_boxes);
        if let Some(ref mut categories_buf) = self.categories_buf {
            assert_eq!(self.options.num_classes, 1);
            let categories_buf = output_buffer_mut_slice!(categories_buf);
            let mut index = 0;
            for i in 0..num_boxes {
                let next_index = index + self.options.num_coords;
                if let Some(category) = self
                    .categories_filter
                    .create_category(categories_buf[i] as usize, scores[i])
                {
                    if let Some(d) = Self::generate_detection(
                        &self.options,
                        category,
                        &location[index..next_index],
                    ) {
                        detections.push(d);
                    }
                }
                index = next_index;
            }
        } else {
            let anchors = self.anchors.unwrap();
            let mut index = 0;
            let mut score_index = 0;
            for i in 0..num_boxes {
                let mut max_score = process_scores!(self, scores[score_index]);
                let mut class_index = 0;
                let num_classes = self.options.num_classes;
                if num_classes != 1 {
                    for i in 1..num_classes {
                        let s = process_scores!(self, scores[score_index + i]);
                        if s > max_score {
                            max_score = s;
                            class_index = i;
                        }
                    }
                }
                score_index += num_classes;

                let next_index = index + self.options.num_coords;

                if max_score >= self.options.min_score_threshold {
                    if let Some(category) = self
                        .categories_filter
                        .create_category(class_index, max_score)
                    {
                        let raw_boxes = &mut location[index..next_index];

                        Self::decode_boxes(&self.options, raw_boxes, &anchors[i]);
                        if let Some(d) =
                            Self::generate_detection(&self.options, category, raw_boxes)
                        {
                            detections.push(d);
                        }
                    }
                }

                index = next_index;
            }
        }

        let mut result = DetectionResult { detections };
        self.nms.do_nms(&mut result);
        result
    }

    #[inline(always)]
    fn generate_detection(
        options: &ToDetectionOptions,
        category: Category,
        location: &[f32],
    ) -> Option<Detection> {
        let mut rect = Rect {
            left: box_x_min!(options, location),
            top: box_y_min!(options, location),
            right: box_x_max!(options, location),
            bottom: box_y_max!(options, location),
        };
        if options.flip_vertically {
            let bottom = 1. - rect.top;
            let top = 1. - rect.bottom;
            rect.bottom = bottom;
            rect.top = top;
        }
        if rect.left.is_nan()
            || rect.right.is_nan()
            || rect.top.is_nan()
            || rect.bottom.is_nan()
            || rect.left >= rect.right
            || rect.top >= rect.bottom
        {
            return None;
        }

        let key_points = if options.num_key_points != 0 {
            let mut key_points = Vec::with_capacity(options.num_key_points);
            let mut index = options.box_coord_offset + options.keypoint_coord_offset;
            for _ in 0..options.num_key_points {
                let y = if options.flip_vertically {
                    1. - location[index + 1]
                } else {
                    location[index + 1]
                };
                key_points.push(NormalizedKeypoint {
                    x: location[index],
                    y,
                    label: None,
                    score: None,
                });
                index += options.num_values_per_key_point;
            }

            Some(key_points)
        } else {
            None
        };

        Some(Detection {
            categories: vec![category],
            bounding_box: rect,
            key_points,
        })
    }

    fn decode_boxes(options: &ToDetectionOptions, raw_boxes: &mut [f32], anchor: &Anchor) {
        let mut x_center;
        let mut y_center;
        let mut h;
        let mut w;
        let box_offset = options.box_coord_offset;
        match options.box_format {
            DetectionBoxFormat::YXHW => {
                y_center = raw_boxes[box_offset];
                x_center = raw_boxes[box_offset + 1];
                h = raw_boxes[box_offset + 2];
                w = raw_boxes[box_offset + 3];
            }
            DetectionBoxFormat::XYWH => {
                x_center = raw_boxes[box_offset];
                y_center = raw_boxes[box_offset + 1];
                w = raw_boxes[box_offset + 2];
                h = raw_boxes[box_offset + 3];
            }
            DetectionBoxFormat::XYXY => {
                x_center = (-raw_boxes[box_offset] + raw_boxes[box_offset + 2]) / 2.;
                y_center = (-raw_boxes[box_offset + 1] + raw_boxes[box_offset + 3]) / 2.;
                w = raw_boxes[box_offset + 2] + raw_boxes[box_offset];
                h = raw_boxes[box_offset + 3] + raw_boxes[box_offset + 1];
            }
        }

        x_center = x_center / options.x_scale * anchor.w + anchor.x_center;
        y_center = y_center / options.y_scale * anchor.h + anchor.y_center;

        if options.apply_exponential_on_box_size {
            h = (h / options.h_scale).exp() * anchor.h;
            w = (w / options.w_scale).exp() * anchor.w;
        } else {
            h = h / options.h_scale * anchor.h;
            w = w / options.w_scale * anchor.w;
        }

        let h_div_2 = h / 2.;
        let w_div_2 = w / 2.;
        let ymin = y_center - h_div_2;
        let xmin = x_center - w_div_2;
        let ymax = y_center + h_div_2;
        let xmax = x_center + w_div_2;
        raw_boxes[box_offset] = ymin;
        raw_boxes[box_offset + 1] = xmin;
        raw_boxes[box_offset + 2] = ymax;
        raw_boxes[box_offset + 3] = xmax;

        if options.num_key_points != 0 {
            let mut index = box_offset + options.keypoint_coord_offset;
            for _ in 0..options.num_key_points {
                let keypoint_y;
                let keypoint_x;
                match options.box_format {
                    DetectionBoxFormat::YXHW => {
                        keypoint_y = raw_boxes[index];
                        keypoint_x = raw_boxes[index + 1];
                    }
                    _ => {
                        keypoint_x = raw_boxes[index];
                        keypoint_y = raw_boxes[index + 1];
                    }
                }
                raw_boxes[index] = keypoint_x / options.x_scale * anchor.w + anchor.x_center;
                raw_boxes[index + 1] = keypoint_y / options.y_scale * anchor.h + anchor.y_center;
                index += options.num_values_per_key_point;
            }
        }
    }
}
