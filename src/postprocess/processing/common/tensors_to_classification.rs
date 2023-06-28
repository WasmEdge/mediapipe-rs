use super::*;
use crate::postprocess::{ClassificationResult, Classifications};

pub(crate) struct TensorsToClassification<'a> {
    categories_filters: Vec<CategoriesFilter<'a>>,
    outputs: Vec<OutputBuffer>,
    max_results: Vec<usize>,
}

impl<'a> TensorsToClassification<'a> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            categories_filters: Vec::new(),
            outputs: Vec::new(),
            max_results: Vec::new(),
        }
    }

    pub(crate) fn add_classification_options(
        &mut self,
        categories_filter: CategoriesFilter<'a>,
        max_results: i32,
        buffer_config: (TensorType, Option<QuantizationParameters>),
        buffer_shape: &[usize],
    ) {
        let max_results = if max_results < 0 {
            usize::MAX
        } else {
            max_results as usize
        };
        self.categories_filters.push(categories_filter);
        self.max_results.push(max_results);

        let elem_size = buffer_shape.iter().fold(1, |a, b| a * b);
        self.outputs
            .push(empty_output_buffer!(buffer_config, elem_size));
    }

    /// index must be valid. or panic!
    #[inline(always)]
    pub(crate) fn output_buffer(&mut self, index: usize) -> &mut [u8] {
        self.outputs
            .get_mut(index)
            .unwrap()
            .data_buffer
            .as_mut_slice()
    }

    #[inline]
    pub(crate) fn result(&mut self, timestamp_ms: Option<u64>) -> ClassificationResult {
        let classifications_count = self.outputs.len();
        let mut res = ClassificationResult {
            classifications: Vec::with_capacity(classifications_count),
            timestamp_ms,
        };

        for id in 0..classifications_count {
            let max_results = self.max_results[id];
            let categories_filter = self.categories_filters.get(id).unwrap();

            let out = self.outputs.get_mut(id).unwrap();
            let scores = output_buffer_mut_slice!(out);
            let mut categories = Vec::new();
            for i in 0..scores.len() {
                if let Some(category) = categories_filter.create_category(i, scores[i]) {
                    categories.push(category);
                }
            }

            categories.sort();
            if max_results < categories.len() {
                categories.drain(max_results..);
            }
            res.classifications.push(Classifications {
                head_index: id,
                head_name: None,
                categories,
            })
        }

        res
    }
}
