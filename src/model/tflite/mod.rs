mod generated;

use super::*;
use generated::{tflite as tflite_model, tflite_metadata};

pub(crate) struct TfLiteModelResource<'buf> {
    output_tensor_metadata: Option<
        flatbuffers::Vector<
            'buf,
            flatbuffers::ForwardsUOffset<tflite_metadata::TensorMetadata<'buf>>,
        >,
    >,

    input_shape: Vec<Vec<usize>>,
    output_shape: Vec<Vec<usize>>,
    input_types: Vec<TensorType>,
    output_types: Vec<TensorType>,
    output_quantization_parameters: Vec<Option<QuantizationParameters>>,
    to_tensor_info: Vec<ToTensorInfo<'buf>>,
    output_name_map: HashMap<&'buf str, usize>,
    associated_files: Option<ZipFiles<'buf>>,
    // now it only used for image segmentation
    output_activation: Activation,
}

impl<'buf> TfLiteModelResource<'buf> {
    // TFL3
    pub(super) const HEAD_MAGIC: &'static [u8] = &[0x54, 0x46, 0x4c, 0x33];

    const METADATA_NAME: &'static str = "TFLITE_METADATA";

    pub(super) fn new(buf: &'buf [u8]) -> Result<Self, Error> {
        let associated_files = ZipFiles::try_new(buf).unwrap_or(None);
        let model = tflite_model::root_as_model(buf)?;
        let mut _self = Self {
            output_tensor_metadata: None,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            input_types: Vec::new(),
            output_types: Vec::new(),
            output_quantization_parameters: Vec::new(),
            to_tensor_info: Vec::new(),
            output_name_map: Default::default(),
            associated_files,
            output_activation: Default::default(),
        };
        _self.parse_subgraph(&model)?;
        let metadata = Self::parse_model_metadata(&model)?;
        if let Some(metadata) = metadata {
            _self.parse_model_metadata_content(&metadata)?;
        }
        Ok(_self)
    }

    #[inline]
    fn parse_subgraph(&mut self, model: &tflite_model::Model<'buf>) -> Result<(), Error> {
        let subgraph = match model.subgraphs() {
            Some(s) => {
                if s.len() < 1 {
                    return Err(Error::ModelParseError("Model subgraph is empty".into()));
                }
                s.get(0)
            }
            None => {
                return Err(Error::ModelParseError(format!("Model has no subgraph")));
            }
        };

        if let (Some(inputs), Some(outputs), Some(tensors)) =
            (subgraph.inputs(), subgraph.outputs(), subgraph.tensors())
        {
            self.input_shape.reserve(inputs.len());
            self.input_types.reserve(inputs.len());
            for i in 0..inputs.len() {
                let index = inputs.get(i) as usize;
                if index >= tensors.len() {
                    return Err(Error::ModelParseError(format!(
                        "Invalid tensor input: index `{}` larger than tensor number `{}`",
                        index,
                        tensors.len()
                    )));
                }
                let t = tensors.get(index);
                self.input_types.push(Self::tflite_type_parse(t.type_())?);
                if let Some(s) = t.shape() {
                    let len = s.len();
                    let mut shape = Vec::with_capacity(len);
                    for d in 0..len {
                        let val = s.get(d) as usize;
                        if val < 1 {
                            return Err(Error::ModelParseError(format!(
                                "Invalid model input `{}` shape `{}, size is `0`",
                                i, d
                            )));
                        }
                        shape.push(val);
                    }
                    self.input_shape.push(shape);
                } else {
                    return Err(Error::ModelParseError(format!(
                        "Missing tensor shape for input `{}`",
                        i
                    )));
                }
            }

            self.output_shape.reserve(outputs.len());
            self.output_types.reserve(outputs.len());
            for i in 0..outputs.len() {
                let index = outputs.get(i) as usize;
                if index >= tensors.len() {
                    return Err(Error::ModelParseError(format!(
                        "Invalid tensor output: index `{}` larger than tensor number `{}`",
                        index,
                        tensors.len()
                    )));
                }
                let t = tensors.get(index);
                let tensor_type = Self::tflite_type_parse(t.type_())?;
                self.output_types.push(tensor_type);

                if let Some(s) = t.shape() {
                    let len = s.len();
                    let mut shape = Vec::with_capacity(len);
                    for d in 0..len {
                        let val = s.get(d) as usize;
                        if val < 1 {
                            return Err(Error::ModelParseError(format!(
                                "Invalid model output `{}` shape `{}, size is `0`",
                                i, d
                            )));
                        }
                        shape.push(val);
                    }
                    self.output_shape.push(shape);
                } else {
                    return Err(Error::ModelParseError(format!(
                        "Missing tensor shape for output `{}`",
                        i
                    )));
                }

                if let Some(q) = t.quantization() {
                    if let (Some(z), Some(s)) = (q.zero_point(), q.scale()) {
                        if z.len() > 0 && s.len() > 0 {
                            while self.output_quantization_parameters.len() < i {
                                self.output_quantization_parameters.push(None);
                            }
                            self.output_quantization_parameters.push(Some(
                                QuantizationParameters {
                                    scale: s.get(0),
                                    zero_point: z.get(0) as i32,
                                },
                            ));
                        }
                    }
                }
            }
        } else {
            return Err(Error::ModelParseError(
                "Model must has inputs, outputs and tensors information.".into(),
            ));
        }
        Ok(())
    }

    #[inline]
    fn parse_model_metadata(
        model: &tflite_model::Model<'buf>,
    ) -> Result<Option<tflite_metadata::ModelMetadata<'buf>>, Error> {
        if let (Some(metadata_vec), Some(model_buffers)) = (model.metadata(), model.buffers()) {
            for i in 0..metadata_vec.len() {
                let m = metadata_vec.get(i);
                if m.name() == Some(Self::METADATA_NAME) {
                    let buf_index = m.buffer() as usize;
                    if buf_index < model_buffers.len() {
                        let data_option = model_buffers.get(buf_index).data();
                        if data_option.is_some() {
                            // todo: submit an issue to flatbuffers and fix the checked error in rust
                            let metadata = unsafe {
                                tflite_metadata::root_as_model_metadata_unchecked(
                                    data_option.unwrap().bytes(),
                                )
                            };
                            return Ok(Some(metadata));
                        }
                    }

                    return Err(Error::ModelParseError(format!(
                        "Missing model buffer (index = `{}`)",
                        buf_index
                    )));
                }
            }
        }
        Ok(None)
    }

    #[inline]
    fn parse_model_metadata_content(
        &mut self,
        metadata: &tflite_metadata::ModelMetadata<'buf>,
    ) -> Result<(), Error> {
        let subgraph = match metadata.subgraph_metadata() {
            Some(s) => {
                if s.len() < 1 {
                    return Ok(());
                }
                s.get(0)
            }
            None => {
                return Ok(());
            }
        };
        if let Some(input_tensors) = subgraph.input_tensor_metadata() {
            let len = input_tensors.len();
            for i in 0..len {
                let input = input_tensors.get(i);
                if input.content().is_none() {
                    continue;
                }
                let content = input.content().unwrap();

                #[cfg(feature = "vision")]
                if let Some(props) = content.content_properties_as_image_properties() {
                    self.parse_vision_model_input_info(i, &input, props)?;
                }

                #[cfg(feature = "audio")]
                if let Some(props) = content.content_properties_as_audio_properties() {
                    self.parse_audio_model_input_info(i, props)?;
                }
            }
        }

        #[cfg(feature = "text")]
        self.parse_text_model_input_info(&subgraph)?;

        self.output_tensor_metadata = subgraph.output_tensor_metadata();
        if let Some(output_tensors) = self.output_tensor_metadata {
            let len = output_tensors.len();
            for i in 0..len {
                let output = output_tensors.get(i);
                if let Some(name) = output.name() {
                    self.output_name_map.insert(name, i);
                }
            }
        }

        #[cfg(feature = "vision")]
        if let Some(custom_metadata) = subgraph.custom_metadata() {
            for i in 0..custom_metadata.len() {
                let m = custom_metadata.get(i);
                if let Some(name) = m.name() {
                    if name == generated::CUSTOM_SEGMENTATION_METADATA_NAME {
                        if let Some(data) = m.data() {
                            let meta = generated::custom_img_segmentation::root_as_image_segmenter_options(data.bytes())?;
                            let activation = meta.activation();

                            self.output_activation = if activation
                                == generated::custom_img_segmentation::Activation::NONE
                            {
                                Activation::None
                            } else if activation
                                == generated::custom_img_segmentation::Activation::SIGMOID
                            {
                                Activation::SIGMOID
                            } else if activation
                                == generated::custom_img_segmentation::Activation::SOFTMAX
                            {
                                Activation::SOFTMAX
                            } else {
                                return Err(crate::Error::ModelParseError(
                                    format!(
                                        "Invalid activation type found in CustomMetadata of ImageSegmenterOptions type: `{}`",
                                        activation.0
                                    ),
                                ));
                            };
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "vision")]
    #[inline]
    fn parse_vision_model_input_info(
        &mut self,
        i: usize,
        input: &tflite_metadata::TensorMetadata<'buf>,
        props: tflite_metadata::ImageProperties<'buf>,
    ) -> Result<(), Error> {
        let tensor_shape = if let Some(shape) = self.input_shape.get(i) {
            if let Ok(s) = crate::preprocess::vision::ImageLikeTensorShape::parse(
                ImageDataLayout::NHWC,
                shape.as_slice(),
            ) {
                s
            } else {
                return Ok(());
            }
        } else {
            return Ok(());
        };
        let (stats_min, stats_max) = if let Some(stats) = input.stats() {
            let min = if let Some(m) = stats.min() {
                m.iter().collect()
            } else {
                vec![]
            };
            let max = if let Some(m) = stats.max() {
                m.iter().collect()
            } else {
                vec![]
            };
            (min, max)
        } else {
            (vec![], vec![])
        };

        let mut normalization_options = (vec![], vec![]);
        if let Some(n) = input.process_units() {
            for i in 0..n.len() {
                if let Some(p) = n.get(i).options_as_normalization_options() {
                    normalization_options.0 = if let Some(m) = p.mean() {
                        m.iter().collect()
                    } else {
                        vec![]
                    };
                    normalization_options.1 = if let Some(m) = p.std_() {
                        m.iter().collect()
                    } else {
                        vec![]
                    };
                    break;
                }
            }
        }

        let color_space = match props.color_space() {
            tflite_metadata::ColorSpaceType::RGB => ImageColorSpaceType::RGB,
            tflite_metadata::ColorSpaceType::GRAYSCALE => ImageColorSpaceType::GRAYSCALE,
            _ => ImageColorSpaceType::UNKNOWN,
        };
        let img_info = ImageToTensorInfo {
            image_data_layout: ImageDataLayout::NHWC,
            color_space,
            tensor_type: self.input_types.get(i).unwrap().clone(),
            tensor_shape,
            stats_min,
            stats_max,
            normalization_options,
        };

        while self.to_tensor_info.len() < i {
            self.to_tensor_info.push(ToTensorInfo::new_none());
        }
        self.to_tensor_info.push(ToTensorInfo::new_image(img_info));

        Ok(())
    }

    #[cfg(feature = "audio")]
    #[inline]
    fn parse_audio_model_input_info(
        &mut self,
        i: usize,
        props: tflite_metadata::AudioProperties<'buf>,
    ) -> Result<(), Error> {
        let input_shape = self.input_shape.get(i).unwrap();
        let num_channels = props.channels() as usize;
        if num_channels == 0 {
            return Err(Error::ModelParseError(format!(
                "Audio input tensor `{}`, num channel cannot be zero",
                i
            )));
        }
        let input_buffer_size = input_shape.iter().fold(1, |mul, &val| mul * val);
        if input_buffer_size % num_channels != 0 {
            return Err(Error::ModelParseError(format!(
                "Input tensor size `{}` should be a multiplier of the number of channels `{}`",
                input_buffer_size, num_channels
            )));
        }
        let num_samples = *input_shape.last().unwrap() / num_channels;
        let audio_info = AudioToTensorInfo {
            num_channels,
            num_samples,
            sample_rate: props.sample_rate() as usize,
            num_overlapping_samples: 0,
            tensor_type: self.input_types.get(i).unwrap().clone(),
        };

        while self.to_tensor_info.len() < i {
            self.to_tensor_info.push(ToTensorInfo::new_none());
        }
        self.to_tensor_info
            .push(ToTensorInfo::new_audio(audio_info));
        Ok(())
    }

    #[cfg(feature = "text")]
    fn parse_text_model_input_info(
        &mut self,
        subgraph: &tflite_metadata::SubGraphMetadata<'buf>,
    ) -> Result<(), Error> {
        // bert model
        if let Some(process_units) = subgraph.input_process_units() {
            for i in 0..process_units.len() {
                if let Some(b) = process_units.get(i).options_as_bert_tokenizer_options() {
                    if self.input_shape.len() != 3 {
                        return Err(Error::ModelParseError(format!(
                            "Model input tensors must be `3` in bert model, but got `{}`",
                            self.input_shape.len()
                        )));
                    }
                    let max_seq_len = Self::get_max_seq_len(&self.input_shape)?;

                    let mut f = self.process_vocab_files(b.vocab_file())?;
                    let mut token_index_map = HashMap::new();
                    let mut index = 0;
                    while let Some(v) = f.next_line() {
                        token_index_map.insert(v, index);
                        index += 1;
                    }

                    let text_model_input =
                        TextToTensorInfo::new_bert_model(max_seq_len, token_index_map)?;
                    self.to_tensor_info.clear();
                    self.to_tensor_info
                        .push(ToTensorInfo::new_text(text_model_input));
                    break;
                }
            }
        }
        // regex model
        if self.to_tensor_info.is_empty() && self.input_types.len() == 1 {
            let input_tensor = subgraph.input_tensor_metadata().unwrap().get(0);
            if let Some(process_units) = input_tensor.process_units() {
                for i in 0..process_units.len() {
                    if let Some(r) = process_units.get(i).options_as_regex_tokenizer_options() {
                        let max_seq_len = Self::get_max_seq_len(&self.input_shape)?;
                        let delim_regex_pattern = match r.delim_regex_pattern() {
                            None => {
                                return Err(Error::ModelParseError(
                                    "Cannot find delim regex pattern information for regex model."
                                        .into(),
                                ));
                            }
                            Some(d) => d,
                        };

                        let mut f = self.process_vocab_files(r.vocab_file())?;
                        let mut token_index_map = HashMap::new();
                        loop {
                            let (token, index) = f.next_line_with_split_white_space();
                            if token.is_none() {
                                break;
                            }
                            let token = token.unwrap();
                            if index.is_none() {
                                return Err(Error::ModelParseError(format!(
                                    "Cannot found index in vocab file at `{}`",
                                    token
                                )));
                            }
                            let index = index.as_ref().unwrap().parse().map_err(|e| {
                                Error::ModelParseError(format!(
                                    "Cannot parse vocab file index: `{:?}`",
                                    e
                                ))
                            })?;
                            token_index_map.insert(token.clone(), index);
                        }

                        let text_model_input = TextToTensorInfo::new_regex_model(
                            max_seq_len,
                            delim_regex_pattern,
                            token_index_map,
                        )?;
                        self.to_tensor_info
                            .push(ToTensorInfo::new_text(text_model_input));
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "text")]
    #[inline]
    fn process_vocab_files(
        &mut self,
        files: Option<
            flatbuffers::Vector<
                'buf,
                flatbuffers::ForwardsUOffset<tflite_metadata::AssociatedFile<'buf>>,
            >,
        >,
    ) -> Result<MemoryTextFile<'buf>, Error> {
        if files.is_none() || files.unwrap().len() == 0 {
            return Err(Error::ModelParseError(
                "No vocab files have been found".into(),
            ));
        }
        let files = files.unwrap();
        if files.len() > 1 {
            // todo: multi vocab files
            return Err(Error::ModelParseError(format!(
                "Now only support one vocab file, but got `{}`",
                files.len()
            )));
        }
        let filename = if let Some(n) = files.get(0).name() {
            n
        } else {
            return Err(Error::ModelParseError(
                "Cannot get associated filename.".into(),
            ));
        };
        Ok(MemoryTextFile::new(self.get_file_content(filename)?))
    }

    // for bert and regex model.
    #[cfg(feature = "text")]
    #[inline(always)]
    fn get_max_seq_len(input_shape: &Vec<Vec<usize>>) -> Result<u32, Error> {
        if input_shape.is_empty() {
            return Err(Error::ModelParseError(format!(
                "Input tensor shape is empty!"
            )));
        }
        for shape in input_shape.iter() {
            if shape.len() != 2 {
                return Err(Error::ModelParseError(format!(
                    "Model should take 2-D input tensors, but got shape `{:?}`",
                    shape
                )));
            }
            if shape[0] != 1 {
                return Err(Error::ModelParseError(format!(
                    "Model batch size must be `1`, but got `{}`",
                    shape[0]
                )));
            }
        }
        let res = input_shape[0][1];
        for shape in input_shape.iter() {
            if res != shape[1] {
                return Err(Error::ModelParseError(format!(
                    "Model input tensors don't have the same shape."
                )));
            }
        }
        Ok(res as u32)
    }

    #[inline(always)]
    fn get_file_content(&self, filename: &str) -> Result<&'buf [u8], Error> {
        if self.associated_files.is_none() {
            return Err(Error::ModelParseError(
                "No associated files have been found in model asset.".into(),
            ));
        }
        match self.associated_files.as_ref().unwrap().get_file(filename) {
            Some(c) => Ok(c),
            None => {
                return Err(Error::ModelParseError(format!(
                    "Cannot find associated file `{}`",
                    filename
                )))
            }
        }
    }

    #[inline(always)]
    fn tflite_type_parse(tflite_type: tflite_model::TensorType) -> Result<TensorType, Error> {
        match tflite_type {
            tflite_model::TensorType::FLOAT32 => Ok(TensorType::F32),
            tflite_model::TensorType::UINT8 => Ok(TensorType::U8),
            tflite_model::TensorType::INT32 => Ok(TensorType::I32),
            tflite_model::TensorType::FLOAT16 => Ok(TensorType::F16),
            _ => Err(Error::ModelParseError(format!(
                "Unsupported tensor type `{:?}`",
                tflite_type
            ))),
        }
    }
}

impl<'buf> ModelResourceTrait for TfLiteModelResource<'buf> {
    fn model_backend(&self) -> GraphEncoding {
        return GraphEncoding::TensorflowLite;
    }

    fn input_tensor_count(&self) -> usize {
        self.input_shape.len()
    }

    fn output_tensor_count(&self) -> usize {
        self.output_shape.len()
    }

    fn input_tensor_type(&self, index: usize) -> Option<TensorType> {
        self.input_types.get(index).cloned()
    }

    fn output_tensor_type(&self, index: usize) -> Option<TensorType> {
        self.output_types.get(index).cloned()
    }

    fn input_tensor_shape(&self, index: usize) -> Option<&[usize]> {
        self.input_shape.get(index).map(|v| v.as_slice())
    }

    fn output_tensor_shape(&self, index: usize) -> Option<&[usize]> {
        self.output_shape.get(index).map(|v| v.as_slice())
    }

    fn output_tensor_name_to_index(&self, name: &'static str) -> Option<usize> {
        self.output_name_map.get(name).cloned()
    }

    fn output_tensor_quantization_parameters(
        &self,
        index: usize,
    ) -> Option<QuantizationParameters> {
        if let Some(i) = self.output_quantization_parameters.get(index) {
            i.clone()
        } else {
            None
        }
    }

    fn output_tensor_labels_locale(
        &self,
        index: usize,
        locale_name: &str,
    ) -> Result<(&[u8], Option<&[u8]>), Error> {
        let mut l = None;
        let mut locale = None;
        if let Some(o) = self.output_tensor_metadata {
            if index < o.len() {
                if let Some(files) = o.get(index).associated_files() {
                    for f in files.iter() {
                        let tp = f.type_();
                        if tp == tflite_metadata::AssociatedFileType::TENSOR_AXIS_LABELS
                            || tp == tflite_metadata::AssociatedFileType::TENSOR_VALUE_LABELS
                        {
                            let file_name = match f.name() {
                                Some(n) => n,
                                None => {
                                    return Err(Error::ModelParseError(format!(
                                        "Cannot find associated file's name"
                                    )))
                                }
                            };
                            let l_ref = if let Some(name) = f.locale() {
                                if name != locale_name {
                                    continue;
                                }
                                &mut locale
                            } else {
                                &mut l
                            };
                            *l_ref = Some(self.get_file_content(file_name)?);
                        }
                    }
                }
            }
        }
        if l.is_none() {
            return if locale.is_some() {
                Ok((locale.unwrap(), locale))
            } else {
                Err(Error::ModelInconsistentError(
                    "Missing model label file information.".into(),
                ))
            };
        }
        Ok((l.unwrap(), locale))
    }

    #[cfg(feature = "vision")]
    fn output_bounding_box_properties(&self, index: usize, slice: &mut [usize]) -> bool {
        if let Some(o) = self.output_tensor_metadata {
            if index < o.len() {
                if let Some(t) = o.get(index).content() {
                    if let Some(t) = t.content_properties_as_bounding_box_properties() {
                        if let Some(i) = t.index() {
                            if i.len() == 4 {
                                for j in 0..4 {
                                    slice[j] = i.get(j) as usize;
                                }
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn to_tensor_info(&self, input_index: usize) -> Option<&ToTensorInfo> {
        self.to_tensor_info.get(input_index)
    }

    fn output_activation(&self) -> Activation {
        self.output_activation
    }
}

// todo: The GPU backend isn't able to process int data. If the input tensor is quantized, forces the image preprocessing graph to use CPU backend.
