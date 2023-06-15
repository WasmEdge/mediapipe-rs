use super::ImageSegmenter;
use crate::model::MemoryTextFile;
use crate::tasks::common::BaseTaskOptions;

/// Configure the build options of a new **Image Segmentation** task instance.
///
/// Methods can be chained on it in order to configure it.
///
/// default options:
/// * display_names_locale: "en"
/// * output_category_mask: true
/// * output_confidence_masks: false
pub struct ImageSegmenterBuilder {
    pub(super) base_task_options: BaseTaskOptions,

    /// The locale to use for display names specified through the TFLite Model
    /// Metadata, if any. Defaults to English.
    pub(super) display_names_locale: String,

    /// If set category_mask, segmentation mask will contain a uint8 image,
    /// where each pixel value indicates the winning category index.
    /// Default is true
    pub(super) output_category_mask: bool,

    /// If set confidence_masks, the segmentation masks are float images,
    /// where each float image represents the confidence score map of the category.
    /// Default is false
    pub(super) output_confidence_masks: bool,
}

impl Default for ImageSegmenterBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            display_names_locale: "en".into(),
            output_category_mask: true,
            output_confidence_masks: false,
        }
    }
}

impl ImageSegmenterBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Default::default()
    }

    base_task_options_impl!(ImageSegmenter);

    /// The locale to use for display names specified through the TFLite Model
    /// Metadata, if any. Defaults to English.
    #[inline(always)]
    pub fn display_names_locale(mut self, locale: String) -> Self {
        self.display_names_locale = locale;
        self
    }

    /// Set whether output the category mask.
    /// Segmentation mask will contain a uint8 image, where each pixel value indicates the winning category index.
    #[inline(always)]
    pub fn output_category_mask(mut self, output_category_mask: bool) -> Self {
        self.output_category_mask = output_category_mask;
        self
    }

    /// Set whether output the confidence masks.
    /// The segmentation masks are float images, where each float image represents the confidence score map of the category.
    #[inline(always)]
    pub fn output_confidence_masks(mut self, output_confidence_masks: bool) -> Self {
        self.output_confidence_masks = output_confidence_masks;
        self
    }

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<ImageSegmenter, crate::Error> {
        if !self.output_category_mask && !self.output_confidence_masks {
            return Err(crate::Error::ArgumentError(
                "At least one of the `output_category_mask` and `output_confidence_masks` be set."
                    .into(),
            ));
        }

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

        let (label, label_locale) =
            model_resource.output_tensor_labels_locale(0, self.display_names_locale.as_str())?;
        let mut text = MemoryTextFile::new(label);
        let mut labels = Vec::new();
        while let Some(l) = text.next_line() {
            labels.push(l.into())
        }

        let labels_locale = if let Some(f) = label_locale {
            let mut text = MemoryTextFile::new(f);
            let mut labels = Vec::new();
            while let Some(l) = text.next_line() {
                labels.push(l.into());
            }
            Some(labels)
        } else {
            None
        };

        let output_activation = model_resource.output_activation();
        return Ok(ImageSegmenter {
            build_options: self,
            model_resource,
            graph,
            labels,
            labels_locale,
            input_tensor_type,
            output_activation,
        });
    }
}
