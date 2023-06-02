use super::ImageEmbedder;
use crate::model::ModelResourceTrait;
use crate::tasks::common::{BaseTaskOptions, EmbeddingOptions};

/// Configure the build options of a new **Image Embedding** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct ImageEmbedderBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) embedding_options: EmbeddingOptions,
}

impl Default for ImageEmbedderBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            embedding_options: Default::default(),
        }
    }
}

impl ImageEmbedderBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    base_task_options_impl!();

    embedding_options_impl!();

    /// Use the build options to create a new task instance.
    #[inline]
    pub fn finalize(mut self) -> Result<ImageEmbedder, crate::Error> {
        let buf = base_task_options_check_and_get_buf!(self);

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model::parse_model(buf.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

        // check model
        model_base_check_impl!(model_resource, 1, 1);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.execution_target,
        )
        .build_from_shared_slices([buf])?;

        return Ok(ImageEmbedder {
            build_options: self,
            model_resource,
            graph,
            input_tensor_type,
        });
    }
}
