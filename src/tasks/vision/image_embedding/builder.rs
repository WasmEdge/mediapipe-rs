use super::ImageEmbedder;
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

    base_task_options_impl!(ImageEmbedder);

    embedding_options_impl!();

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<ImageEmbedder, crate::Error> {
        let buf = buffer.as_ref();
        // parse model and get model resources.
        let model_resource = crate::model::parse_model(buf)?;

        // check model
        model_base_check_impl!(model_resource, 1, 1);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.device,
        )
        .build_from_bytes([buf])?;

        return Ok(ImageEmbedder {
            build_options: self,
            model_resource,
            graph,
            input_tensor_type,
        });
    }
}
