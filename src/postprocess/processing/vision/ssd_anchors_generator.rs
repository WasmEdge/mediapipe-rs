// These reference files are licensed under Apache 2.0, and originally developed by Google for Mediapipe:
// https://github.com/google/mediapipe/raw/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc

#[derive(Debug, Clone)]
pub(crate) struct Anchor {
    pub x_center: f32,
    pub y_center: f32,
    pub h: f32,
    pub w: f32,
}

/// Options to generate anchors for SSD object detection models.
pub(crate) struct SsdAnchorsBuilder {
    /// Size of input images.
    input_size_width: u32,
    input_size_height: u32,

    /// Min and max scales for generating anchor boxes on feature maps.
    min_scale: f32,
    max_scale: f32,

    /// The offset for the center of anchors. The value is in the scale of stride.
    /// E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
    anchor_offset_x: f32, // default 0.5
    anchor_offset_y: f32, // default 0.5

    /// Number of output feature maps to generate the anchors on.
    num_layers: usize,
    /// Sizes of output feature maps to create anchors. Either feature_map size or stride should be provided.
    feature_map_width: Vec<u32>,
    feature_map_height: Vec<u32>,

    /// Strides of each output feature maps.
    strides: Vec<i32>,

    /// List of different aspect ratio to generate anchors.
    aspect_ratios: Vec<f32>,

    /// A boolean to indicate whether the fixed 3 boxes per location is used in the lowest layer.
    reduce_boxes_in_lowest_layer: bool, // default false

    /// An additional anchor is added with this aspect ratio and a scale
    /// interpolated between the scale for a layer and the scale for the next layer
    /// (1.0 for the last layer). This anchor is not included if this value is 0.
    interpolated_scale_aspect_ratio: f32, // [default = 1.0]

    /// Whether use fixed width and height (e.g. both 1.0f) for each anchor.
    /// This option can be used when the predicted anchor width and height are in pixels.
    fixed_anchor_size: bool, // [default = false];

    /// Generates grid anchors on the fly corresponding to multiple CNN layers as
    /// described in:
    /// "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    ///  T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollar
    multiscale_anchor_generation: bool, // [default = false];

    /// minimum level in feature pyramid
    /// for multiscale_anchor_generation only!
    min_level: i32, // [default = 3];

    /// maximum level in feature pyramid
    /// for multiscale_anchor_generation only!
    max_level: i32, // [default = 7];

    /// Scale of anchor to feature stride
    /// for multiscale_anchor_generation only!
    anchor_scale: f32, // [default = 4.0];

    /// Number of intermediate scale each scale octave
    /// for multiscale_anchor_generation only!
    scales_per_octave: i32, // [default = 2];

    /// Whether to produce anchors in normalized coordinates.
    /// for multiscale_anchor_generation only!
    normalize_coordinates: bool, // [default = true];
}

#[allow(unused)]
impl SsdAnchorsBuilder {
    #[inline(always)]
    pub fn new(
        input_size_width: u32,
        input_size_height: u32,
        min_scale: f32,
        max_scale: f32,
        num_layers: usize,
    ) -> Self {
        Self {
            input_size_width,
            input_size_height,
            min_scale,
            max_scale,

            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,

            num_layers,
            feature_map_width: Vec::new(),
            feature_map_height: Vec::new(),
            strides: Vec::new(),
            aspect_ratios: Vec::new(),

            reduce_boxes_in_lowest_layer: false,  // default false
            interpolated_scale_aspect_ratio: 1.0, // [default = 1.0]
            fixed_anchor_size: false,             // [default = false];
            multiscale_anchor_generation: false,  // [default = false];
            min_level: 3,                         // [default = 3];
            max_level: 7,                         // [default = 7];
            anchor_scale: 4.0,                    // [default = 4.0];
            scales_per_octave: 2,                 // [default = 2];
            normalize_coordinates: true,          // [default = true];
        }
    }

    #[inline(always)]
    fn calculate_scale(&self, stride_index: usize, num_strides: usize) -> f32 {
        return if num_strides == 1 {
            (self.min_scale + self.max_scale) * 0.5f32
        } else {
            self.min_scale
                + (self.max_scale - self.min_scale) * (stride_index as f32)
                    / ((num_strides as f32) - 1.0f32)
        };
    }

    /// reference: https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    pub fn generate(self) -> Vec<Anchor> {
        let mut ans = Vec::new();

        let mut layer_id = 0;
        while layer_id < self.num_layers {
            let mut anchor_height = Vec::new();
            let mut anchor_width = Vec::new();
            let mut aspect_ratios = Vec::new();
            let mut scales = Vec::new();

            let mut last_same_stride_layer = layer_id;
            while last_same_stride_layer < self.strides.len()
                && self.strides[last_same_stride_layer] == self.strides[layer_id]
            {
                let scale = self.calculate_scale(last_same_stride_layer, self.strides.len());

                if last_same_stride_layer == 0 && self.reduce_boxes_in_lowest_layer {
                    // For first layer, it can be specified to use predefined anchors.
                    aspect_ratios.push(1.0f32);
                    aspect_ratios.push(2.0f32);
                    aspect_ratios.push(0.5f32);
                    scales.push(0.1);
                    scales.push(scale);
                    scales.push(scale);
                } else {
                    for aspect_ratio_id in 0..self.aspect_ratios.len() {
                        aspect_ratios.push(self.aspect_ratios[aspect_ratio_id]);
                        scales.push(scale);
                    }
                    if self.interpolated_scale_aspect_ratio > 0.0 {
                        let scale_next = if last_same_stride_layer == self.strides.len() - 1 {
                            1.0f32
                        } else {
                            self.calculate_scale(last_same_stride_layer + 1, self.strides.len())
                        };
                        scales.push((scale * scale_next).sqrt());
                        aspect_ratios.push(self.interpolated_scale_aspect_ratio);
                    }
                }
                last_same_stride_layer += 1;
            }

            for i in 0..aspect_ratios.len() {
                let ratio_sqrt = aspect_ratios[i].sqrt();
                anchor_height.push(scales[i] / ratio_sqrt);
                anchor_width.push(scales[i] * ratio_sqrt);
            }

            let feature_map_height;
            let feature_map_width;
            if !self.feature_map_height.is_empty() && !self.feature_map_width.is_empty() {
                feature_map_height = self.feature_map_height[layer_id];
                feature_map_width = self.feature_map_width[layer_id];
            } else {
                let stride = self.strides[layer_id];
                feature_map_height = (self.input_size_height as f32 / stride as f32).ceil() as u32;
                feature_map_width = (self.input_size_width as f32 / stride as f32).ceil() as u32;
            }

            for y in 0..feature_map_height {
                for x in 0..feature_map_width {
                    for anchor_id in 0..anchor_height.len() {
                        let x_center = (x as f32 + self.anchor_offset_x) / feature_map_width as f32;
                        let y_center =
                            (y as f32 + self.anchor_offset_y) / feature_map_height as f32;
                        let anchor = if self.fixed_anchor_size {
                            Anchor {
                                x_center,
                                y_center,
                                w: 1f32,
                                h: 1f32,
                            }
                        } else {
                            Anchor {
                                x_center,
                                y_center,
                                w: anchor_width[anchor_id],
                                h: anchor_height[anchor_id],
                            }
                        };
                        ans.push(anchor);
                    }
                }
            }
            layer_id = last_same_stride_layer;
        }
        ans
    }

    /// Size of input images.
    pub fn input_size_width(mut self, input_size_width: u32) -> Self {
        self.input_size_width = input_size_width;
        self
    }

    #[inline(always)]
    pub fn input_size_height(mut self, input_size_height: u32) -> Self {
        self.input_size_height = input_size_height;
        self
    }

    /// Min and max scales for generating anchor boxes on feature maps.
    pub fn min_scale(mut self, min_scale: f32) -> Self {
        self.min_scale = min_scale;
        self
    }

    #[inline(always)]
    pub fn max_scale(mut self, max_scale: f32) -> Self {
        self.max_scale = max_scale;
        self
    }

    /// The offset for the center of anchors. The value is in the scale of stride.
    /// E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
    #[inline(always)]
    pub fn anchor_offset_x(mut self, anchor_offset_x: f32) -> Self {
        self.anchor_offset_x = anchor_offset_x;
        self
    }

    #[inline(always)]
    pub fn anchor_offset_y(mut self, anchor_offset_y: f32) -> Self {
        self.anchor_offset_y = anchor_offset_y;
        self
    }

    /// Number of output feature maps to generate the anchors on.
    #[inline(always)]
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Sizes of output feature maps to create anchors. Either feature_map size or stride should be provided.
    #[inline(always)]
    pub fn feature_map_width(mut self, feature_map_width: Vec<u32>) -> Self {
        self.feature_map_width = feature_map_width;
        self
    }

    #[inline(always)]
    pub fn feature_map_height(mut self, feature_map_height: Vec<u32>) -> Self {
        self.feature_map_height = feature_map_height;
        self
    }

    /// Strides of each output feature maps.
    #[inline(always)]
    pub fn strides(mut self, strides: Vec<i32>) -> Self {
        self.strides = strides;
        self
    }

    /// List of different aspect ratio to generate anchors.
    #[inline(always)]
    pub fn aspect_ratios(mut self, aspect_ratios: Vec<f32>) -> Self {
        self.aspect_ratios = aspect_ratios;
        self
    }

    /// A boolean to indicate whether the fixed 3 boxes per location is used in the lowest layer.
    #[inline(always)]
    pub fn reduce_boxes_in_lowest_layer(mut self, reduce_boxes_in_lowest_layer: bool) -> Self {
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer;
        self
    }

    /// An additional anchor is added with this aspect ratio and a scale
    /// interpolated between the scale for a layer and the scale for the next layer
    /// (1.0 for the last layer). This anchor is not included if this value is 0.
    #[inline(always)]
    pub fn interpolated_scale_aspect_ratio(mut self, interpolated_scale_aspect_ratio: f32) -> Self {
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio;
        self
    }

    /// Whether use fixed width and height (e.g. both 1.0f) for each anchor.
    /// This option can be used when the predicted anchor width and height are in pixels.
    #[inline(always)]
    pub fn fixed_anchor_size(mut self, fixed_anchor_size: bool) -> Self {
        self.fixed_anchor_size = fixed_anchor_size;
        self
    }

    /// Generates grid anchors on the fly corresponding to multiple CNN layers as
    /// described in:
    /// "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    ///  T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollar
    #[inline(always)]
    pub fn multiscale_anchor_generation(mut self, multiscale_anchor_generation: bool) -> Self {
        self.multiscale_anchor_generation = multiscale_anchor_generation;
        self
    }

    /// minimum level in feature pyramid
    /// for multiscale_anchor_generation only!
    #[inline(always)]
    pub fn min_level(mut self, min_level: i32) -> Self {
        self.min_level = min_level;
        self
    }

    /// maximum level in feature pyramid
    /// for multiscale_anchor_generation only!
    #[inline(always)]
    pub fn max_level(mut self, max_level: i32) -> Self {
        self.max_level = max_level;
        self
    }

    /// Scale of anchor to feature stride
    /// for multiscale_anchor_generation only!
    #[inline(always)]
    pub fn anchor_scale(mut self, anchor_scale: f32) -> Self {
        self.anchor_scale = anchor_scale;
        self
    }

    /// Number of intermediate scale each scale octave
    /// for multiscale_anchor_generation only!
    #[inline(always)]
    pub fn scales_per_octave(mut self, scales_per_octave: i32) -> Self {
        self.scales_per_octave = scales_per_octave;
        self
    }

    /// Whether to produce anchors in normalized coordinates.
    /// for multiscale_anchor_generation only!
    #[inline(always)]
    pub fn normalize_coordinates(mut self, normalize_coordinates: bool) -> Self {
        self.normalize_coordinates = normalize_coordinates;
        self
    }
}
