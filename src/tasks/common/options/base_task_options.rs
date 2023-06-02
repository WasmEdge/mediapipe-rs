pub(crate) struct BaseTaskOptions {
    /// The model asset file contents.
    pub model_asset_buffer: Option<crate::SharedSlice<u8>>,

    /// The path to the model asset.
    pub model_asset_path: Option<std::path::PathBuf>,

    /// The device to run the models.
    pub execution_target: crate::Device,
}

impl Default for BaseTaskOptions {
    fn default() -> Self {
        Self {
            model_asset_buffer: None,
            model_asset_path: None,
            execution_target: crate::Device::CPU,
        }
    }
}

macro_rules! base_task_options_impl {
    () => {
        /// Set model asset data use [`crate::SharedSlice`]
        #[inline(always)]
        pub fn model_asset_slice(mut self, model_asset_slice: crate::SharedSlice<u8>) -> Self {
            self.base_task_options.model_asset_buffer = Some(model_asset_slice);
            self
        }

        /// Set model asset data using a data buffer
        #[inline(always)]
        pub fn model_asset_buffer(mut self, model_asset_buffer: Vec<u8>) -> Self {
            self.base_task_options.model_asset_buffer =
                Some(crate::SharedSlice::from(model_asset_buffer));
            self
        }

        /// Set model asset use a file path
        #[inline(always)]
        pub fn model_asset_path(mut self, model_asset_path: impl Into<std::path::PathBuf>) -> Self {
            self.base_task_options.model_asset_path = Some(model_asset_path.into());
            self
        }

        /// Set the device to run the models.
        #[inline(always)]
        pub fn execution_target(mut self, execution_target: crate::Device) -> Self {
            self.base_task_options.execution_target = execution_target;
            self
        }

        /// Set ```CPU``` device to run the models.
        #[inline(always)]
        pub fn cpu(mut self) -> Self {
            self.base_task_options.execution_target = crate::Device::CPU;
            self
        }

        /// Set ```GPU``` device to run the models.
        #[inline(always)]
        pub fn gpu(mut self) -> Self {
            self.base_task_options.execution_target = crate::Device::GPU;
            self
        }

        /// Set ```TPU``` device to run the models.
        #[inline(always)]
        pub fn tpu(mut self) -> Self {
            self.base_task_options.execution_target = crate::Device::TPU;
            self
        }
    };
}

macro_rules! base_task_options_check_and_get_buf {
    ( $self:ident ) => {{
        let a = $self.base_task_options.model_asset_path.is_some();
        let b = $self.base_task_options.model_asset_buffer.is_some();
        if a {
            if b {
                return Err(crate::Error::ArgumentError(
                    "Cannot use both `model_asset_path` and `model_asset_buffer`".into(),
                ));
            } else {
                let buf = std::fs::read($self.base_task_options.model_asset_path.as_ref().unwrap())
                    .map_err(|e| crate::Error::from(e))?;
                crate::SharedSlice::from(buf)
            }
        } else {
            if b {
                $self.base_task_options.model_asset_buffer.take().unwrap()
            } else {
                return Err(crate::Error::ArgumentError(
                    "Must use `model_asset_path` or `model_asset_buffer` to specify a model".into(),
                ));
            }
        }
    }};
}

macro_rules! base_task_options_get_impl {
    () => {
        /// Get the task running device.
        #[inline(always)]
        pub fn execution_target(&self) -> crate::Device {
            self.build_options.base_task_options.execution_target
        }
    };
}

macro_rules! model_base_check_impl {
    ( $model_resource:ident, $expect_input_count:expr, $expect_output_count:expr ) => {{
        let input_tensor_count = $model_resource.input_tensor_count();
        if input_tensor_count != $expect_input_count {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect model input tensor count `{}`, but got `{}`",
                $expect_input_count, input_tensor_count
            )));
        }
        let output_tensor_count = $model_resource.output_tensor_count();
        if output_tensor_count != $expect_output_count {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect model output tensor count `{}`, but got `{}`",
                $expect_output_count, output_tensor_count
            )));
        }
    }};

    ( $model_resource:ident, $expect_output_count:expr ) => {{
        let output_tensor_count = $model_resource.output_tensor_count();
        if output_tensor_count != $expect_output_count {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Expect model output tensor count `{}`, but got `{}`",
                $expect_output_count, output_tensor_count
            )));
        }
    }};
}
