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
        #[inline]
        pub fn device(mut self, device: crate::Device) -> Self {
            self.base_task_options.device = device;
            self
        }

        /// Set ```CPU``` device to run the models. (Default device)
        #[inline]
        pub fn cpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::CPU;
            self
        }

        /// Set ```GPU``` device to run the models.
        #[inline]
        pub fn gpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::GPU;
            self
        }

        /// Set ```TPU``` device to run the models.
        #[inline]
        pub fn tpu(mut self) -> Self {
            self.base_task_options.device = crate::Device::TPU;
            self
        }

        /// Use the current build options, read model from file to create a new task instance.
        #[inline]
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
        #[inline]
        pub fn device(&self) -> crate::Device {
            self.build_options.base_task_options.device
        }
    };
}
