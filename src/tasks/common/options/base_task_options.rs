pub(crate) struct BaseTaskOptions {
    /// The device to run the models.
    pub device: crate::Device,
}

impl Default for BaseTaskOptions {
    /// Default target is CPU
    fn default() -> Self {
        Self {
            device: crate::Device::CPU,
        }
    }
}

macro_rules! base_task_options_impl {
    ( $TypeName:ident ) => {
        /// Set execution device.
        #[inline(always)]
        pub fn device(mut self, device: crate::Device) -> Self {
            self.base_task_options.device = device;
            self
        }

        /// Set ```CPU``` device to run the models. (Default device)
        #[inline(always)]
        pub fn cpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::CPU;
            self
        }

        /// Set ```GPU``` device to run the models.
        #[inline(always)]
        pub fn gpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::GPU;
            self
        }

        /// Set ```TPU``` device to run the models.
        #[inline(always)]
        pub fn tpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::TPU;
            self
        }

        /// Use the current build options, read model from file to create a new task instance.
        #[inline(always)]
        pub fn build_from_file(
            self,
            file_path: impl AsRef<std::path::Path>,
        ) -> Result<$TypeName, crate::Error> {
            self.build_from_buffer(std::fs::read(file_path)?)
        }
    };
}

macro_rules! base_task_options_get_impl {
    () => {
        /// Get the task running device.
        #[inline(always)]
        pub fn device(&self) -> crate::Device {
            self.build_options.base_task_options.device
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
