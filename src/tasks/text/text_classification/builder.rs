use super::TextClassifier;
use crate::model::ModelResourceTrait;
use crate::tasks::common::{BaseTaskOptions, ClassificationOptions};
use crate::Error;
use wasi_nn_safe::TensorType;

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

    base_task_options_impl!();

    classification_options_impl!();

    /// Use the build options to create a new task instance.
    #[inline]
    pub fn finalize(mut self) -> Result<TextClassifier, Error> {
        classification_options_check!(self, classification_options);
        let buf = base_task_options_check_and_get_buf!(self);

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model::parse_model(buf.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

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
            self.base_task_options.execution_target,
        )
        .build_from_shared_slices([buf])?;

        return Ok(TextClassifier {
            build_options: self,
            model_resource,
            graph,
        });
    }
}
