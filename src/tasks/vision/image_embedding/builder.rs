use super::ImageEmbedder;
use crate::tasks::common::{BaseTaskOptions, EmbeddingOptions};

/// Configure the build options of a new **Image Embedding** task instance.
///
/// Methods can be chained on it in order to configure it.
#[derive(Default)]
pub struct ImageEmbedderBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) embedding_options: EmbeddingOptions,
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
        model_resource.check_tensor_counts(Some(1), 1)?;
        model_resource.expect_to_tensor_info(0)?.try_to_image()?;
        let input_tensor_type = model_resource.expect_input_tensor_type(0)?;

        let graph = crate::tasks::common::build_graph(
            model_resource.as_ref(),
            self.base_task_options.device,
            buf,
        )?;

        Ok(ImageEmbedder {
            build_options: self,
            model_resource,
            graph,
            input_tensor_type,
        })
    }
}
