use super::ObjectDetector;
use crate::postprocess::TensorsToDetection;
use crate::tasks::common::{BaseTaskOptions, ClassificationOptions};

/// Configure the build options of a new **Object Detection** task instance.
///
/// Methods can be chained on it in order to configure it.
#[derive(Default)]
pub struct ObjectDetectorBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) classification_options: ClassificationOptions,
}

impl ObjectDetectorBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    base_task_options_impl!(ObjectDetector);

    classification_options_impl!();

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<ObjectDetector, crate::Error> {
        classification_options_check!(self, classification_options);

        let buf = buffer.as_ref();
        // parse model and get model resources.
        let model_resource = crate::model::parse_model(buf)?;

        // check model
        model_resource.check_tensor_counts(Some(1), 4)?;
        model_resource.expect_to_tensor_info(0)?.try_to_image()?;

        let graph = crate::tasks::common::build_graph(
            model_resource.as_ref(),
            self.base_task_options.device,
            buf,
        )?;

        let input_tensor_type = model_resource.expect_input_tensor_type(0)?;
        let location_buf_index = model_resource.expect_output_tensor_index("location")?;
        let mut bound_box_properties = [0, 1, 2, 3];
        if model_resource
            .output_bounding_box_properties(location_buf_index, &mut bound_box_properties)
        {
            TensorsToDetection::check_box_indices(&bound_box_properties)?;
        }

        let categories_buf_index = model_resource.expect_output_tensor_index("category")?;
        let score_buf_index = model_resource.expect_output_tensor_index("score")?;
        let named = [location_buf_index, categories_buf_index, score_buf_index];
        let used = named
            .iter()
            .filter(|i| **i < 4)
            .fold(0u8, |mask, i| mask | (1 << i));
        let num_box_buf_index = (0..4)
            .find(|i| used & (1 << i) == 0)
            .filter(|_| used.count_ones() == 3)
            .ok_or_else(|| {
                crate::Error::ModelInconsistentError(format!(
                    "Output tensors `location`, `category`, `score` must be three distinct indices in `0..4`, but got `{:?}`",
                    named
                ))
            })?;
        model_resource.check_scalar_f32_output(num_box_buf_index)?;
        Ok(ObjectDetector {
            build_options: self,
            model_resource,
            graph,
            bound_box_properties,
            location_buf_index,
            categories_buf_index,
            score_buf_index,
            num_box_buf_index,
            input_tensor_type,
        })
    }
}
