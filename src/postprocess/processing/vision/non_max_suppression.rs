// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/calculators/util/non_max_suppression_calculator.cc

use crate::postprocess::{Detection, DetectionResult, Rect};

#[derive(Debug, Copy, Clone)]
pub enum NonMaxSuppressionOverlapType {
    Jaccard,
    ModifiedJaccard,
    IntersectionOverUnion,
}

#[derive(Debug, Copy, Clone)]
pub enum NonMaxSuppressionAlgorithm {
    DEFAULT,
    WEIGHTED,
}

pub struct NonMaxSuppression {
    overlap_type: NonMaxSuppressionOverlapType,
    algorithm: NonMaxSuppressionAlgorithm,
    max_results: usize,
    min_suppression_threshold: f32,
}

impl NonMaxSuppression {
    #[inline(always)]
    pub fn new(max_results: i32) -> Self {
        let max_results = if max_results < 0 {
            usize::MAX
        } else {
            max_results as usize
        };
        Self {
            overlap_type: NonMaxSuppressionOverlapType::Jaccard,
            algorithm: NonMaxSuppressionAlgorithm::DEFAULT,
            max_results,
            min_suppression_threshold: 1.0, // default
        }
    }

    #[inline(always)]
    pub fn set_overlap_type(&mut self, overlap_type: NonMaxSuppressionOverlapType) {
        self.overlap_type = overlap_type;
    }

    #[inline(always)]
    pub fn overlap_type(mut self, overlap_type: NonMaxSuppressionOverlapType) -> Self {
        self.overlap_type = overlap_type;
        self
    }

    #[inline(always)]
    pub fn set_algorithm(&mut self, algorithm: NonMaxSuppressionAlgorithm) {
        self.algorithm = algorithm;
    }

    #[inline(always)]
    pub fn algorithm(mut self, algorithm: NonMaxSuppressionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    #[inline(always)]
    pub fn set_min_suppression_threshold(&mut self, min_suppression_threshold: f32) {
        self.min_suppression_threshold = min_suppression_threshold;
    }

    #[inline(always)]
    pub fn max_results(mut self, max_results: i32) -> Self {
        self.max_results = if max_results < 0 {
            usize::MAX
        } else {
            max_results as usize
        };
        self
    }

    #[inline]
    pub fn do_nms(&self, detection_result: &mut DetectionResult) {
        // remove all but the maximum scoring label from each input detection.
        detection_result
            .detections
            .retain(|d| !d.categories.is_empty());
        for d in &mut detection_result.detections {
            if d.categories.len() > 1 {
                d.categories.sort();
                d.categories.drain(1..);
            }
        }

        let mut indexed_scores = Vec::with_capacity(detection_result.detections.len());
        for i in 0..detection_result.detections.len() {
            indexed_scores.push((i, detection_result.detections[i].categories[0].score));
        }
        indexed_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        match self.algorithm {
            NonMaxSuppressionAlgorithm::DEFAULT => {
                self.non_max_suppression(&mut detection_result.detections, indexed_scores);
            }
            NonMaxSuppressionAlgorithm::WEIGHTED => {
                self.non_max_suppression_weighted(&mut detection_result.detections, indexed_scores);
            }
        }
    }

    fn non_max_suppression(
        &self,
        detections: &mut Vec<Detection>,
        indexed_scores: Vec<(usize, f32)>,
    ) {
        let mut retains = vec![false; detections.len()];
        let mut retained_locations = Vec::new();
        for (index, score) in indexed_scores {
            let location = &detections[index].bounding_box;
            let mut suppressed = false;

            for retained_location in &retained_locations {
                let similarity = self.overlap_similarity(location, *retained_location);
                if similarity > self.min_suppression_threshold {
                    suppressed = true;
                    break;
                }
            }

            if !suppressed {
                retains[index] = true;
                retained_locations.push(location);
                if retained_locations.len() >= self.max_results {
                    break;
                }
            }
        }

        let mut iter = retains.iter();
        detections.retain(|_| *iter.next().unwrap());
    }

    fn non_max_suppression_weighted(
        &self,
        out_detections: &mut Vec<Detection>,
        mut indexed_scores: Vec<(usize, f32)>,
    ) {
        let mut remains = Vec::with_capacity(indexed_scores.len());
        let mut in_detections = std::mem::take(out_detections);
        while !indexed_scores.is_empty() {
            let detection = &mut in_detections[indexed_scores[0].0];
            let categories = std::mem::take(&mut detection.categories);
            let mut bounding_box = detection.bounding_box.clone();
            let mut key_points = detection.key_points.take();
            let mut total_score = indexed_scores[0].1;
            bounding_box.top *= total_score;
            bounding_box.bottom *= total_score;
            bounding_box.left *= total_score;
            bounding_box.right *= total_score;
            if let Some(ref mut ks) = key_points {
                for k in ks {
                    k.x *= total_score;
                    k.y *= total_score;
                }
            }

            let location = &in_detections[indexed_scores[0].0].bounding_box;
            for i in 1..indexed_scores.len() {
                let indexed_score = indexed_scores[i];
                let rest_detection = &in_detections[indexed_score.0];

                let similarity = self.overlap_similarity(location, &rest_detection.bounding_box);
                if similarity > self.min_suppression_threshold {
                    let score = indexed_score.1;
                    total_score += score;
                    bounding_box.top += rest_detection.bounding_box.top * score;
                    bounding_box.bottom += rest_detection.bounding_box.bottom * score;
                    bounding_box.left += rest_detection.bounding_box.left * score;
                    bounding_box.right += rest_detection.bounding_box.right * score;
                    if let Some(ref mut k) = key_points {
                        let add = rest_detection.key_points.as_ref().unwrap();
                        for id in 0..add.len() {
                            k[id].x += add[id].x * score;
                            k[id].y += add[id].y * score;
                        }
                    }
                } else {
                    remains.push(indexed_score);
                }
            }

            if total_score != 0. {
                bounding_box.top /= total_score;
                bounding_box.bottom /= total_score;
                bounding_box.left /= total_score;
                bounding_box.right /= total_score;
                if let Some(ref mut ks) = key_points {
                    for k in ks {
                        k.x /= total_score;
                        k.y /= total_score;
                    }
                }

                out_detections.push(Detection {
                    categories,
                    bounding_box,
                    key_points,
                });
                if out_detections.len() >= self.max_results {
                    break;
                }
            }

            if remains.is_empty() {
                break;
            } else {
                indexed_scores.clear();
                std::mem::swap(&mut remains, &mut indexed_scores);
            }
        }
    }

    #[inline]
    fn overlap_similarity(&self, rect_1: &Rect<f32>, rect_2: &Rect<f32>) -> f32 {
        if let Some(intersection) = rect_1.intersect(rect_2) {
            let intersection_area = intersection.area();
            let normalization = match self.overlap_type {
                NonMaxSuppressionOverlapType::Jaccard => rect_1.union(rect_2).area(),
                NonMaxSuppressionOverlapType::ModifiedJaccard => rect_2.area(),
                NonMaxSuppressionOverlapType::IntersectionOverUnion => {
                    rect_1.area() + rect_2.area() - intersection_area
                }
            };
            if normalization > 0. {
                intersection_area / normalization
            } else {
                0.
            }
        } else {
            0.
        }
    }
}
