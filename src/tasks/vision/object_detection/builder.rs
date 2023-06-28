use super::ObjectDetector;
use crate::tasks::common::{BaseTaskOptions, ClassificationOptions};

/// Configure the build options of a new **Object Detection** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct ObjectDetectorBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) classification_options: ClassificationOptions,
}

impl Default for ObjectDetectorBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            classification_options: Default::default(),
        }
    }
}

impl ObjectDetectorBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            classification_options: Default::default(),
        }
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
        model_base_check_impl!(model_resource, 1, 4);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.device,
        )
        .build_from_bytes([buf])?;

        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);
        let location_buf_index = model_resource_check_and_get_impl!(
            model_resource,
            output_tensor_name_to_index,
            "location"
        );
        let mut bound_box_properties = [0, 1, 2, 3];
        if model_resource
            .output_bounding_box_properties(location_buf_index, &mut bound_box_properties)
        {
            for i in 0..4 {
                if bound_box_properties[i] >= 4 {
                    return Err(crate::Error::ModelInconsistentError(format!(
                        "BoundingBoxProperties must contains `0,1,2,3`, but got `{}`",
                        bound_box_properties[i]
                    )));
                }
            }
        }

        let categories_buf_index = model_resource_check_and_get_impl!(
            model_resource,
            output_tensor_name_to_index,
            "category"
        );
        let score_buf_index = model_resource_check_and_get_impl!(
            model_resource,
            output_tensor_name_to_index,
            "score"
        );
        let num_box_buf_index = {
            let mut p = [true; 4];
            p[location_buf_index] = false;
            p[categories_buf_index] = false;
            p[score_buf_index] = false;
            let mut i = 0;
            while i < 4 {
                if p[i] {
                    break;
                }
                i += 1;
            }
            i
        };
        return Ok(ObjectDetector {
            build_options: self,
            model_resource,
            graph,
            bound_box_properties,
            location_buf_index,
            categories_buf_index,
            score_buf_index,
            num_box_buf_index,
            input_tensor_type,
        });
    }
}
