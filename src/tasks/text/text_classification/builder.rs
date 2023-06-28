use super::TextClassifier;
use crate::tasks::common::{BaseTaskOptions, ClassificationOptions};
use crate::{Error, TensorType};

/// Configure the build options of a new **Text Classification** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct TextClassifierBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) classification_options: ClassificationOptions,
}

impl Default for TextClassifierBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            classification_options: Default::default(),
        }
    }
}

impl TextClassifierBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            classification_options: Default::default(),
        }
    }

    base_task_options_impl!(TextClassifier);

    classification_options_impl!();

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(self, buffer: impl AsRef<[u8]>) -> Result<TextClassifier, Error> {
        classification_options_check!(self, classification_options);

        let buf = buffer.as_ref();
        // parse model and get model resources.
        let model_resource = crate::model::parse_model(buf)?;

        // check model
        model_base_check_impl!(model_resource, 1);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_text()?;

        let input_count = model_resource.input_tensor_count();
        if input_count != 1 && input_count != 3 {
            return Err(Error::ModelInconsistentError(format!(
                "Expect model input tensor count `1` or `3`, but got `{}`",
                input_count
            )));
        }
        for i in 0..input_count {
            let t = model_resource_check_and_get_impl!(model_resource, input_tensor_type, i);
            if t != TensorType::I32 {
                // todo: string type support
                return Err(Error::ModelInconsistentError(
                    "All input tensors should be int32 type".into(),
                ));
            }
        }

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.device,
        )
        .build_from_bytes([buf])?;

        return Ok(TextClassifier {
            build_options: self,
            model_resource,
            graph,
        });
    }
}
