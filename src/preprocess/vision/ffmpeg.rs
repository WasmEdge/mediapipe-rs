use super::*;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

type FFMpegVideoInput =
    common::ffmpeg_input::FFMpegInput<ffmpeg_next::decoder::Video, ffmpeg_next::frame::Video>;

/// FFMpeg Video Data, which can be used as vision tasks input.
pub struct FFMpegVideoData {
    source: FFMpegVideoInput,

    // immutable caches
    filter_desc_in: String,
    convert_to_ms: f64,

    // mutable caches
    scales: RefCell<HashMap<ScaleKey, ffmpeg_next::software::scaling::Context>>,
    scale_frame_buffer: RefCell<ffmpeg_next::frame::Video>,
}

impl FFMpegVideoData {
    /// Create a new instance from a FFMpeg input.
    #[inline(always)]
    pub fn new(input: ffmpeg_next::format::context::Input) -> Result<Self, Error> {
        let source = FFMpegVideoInput::new(input)?;
        let convert_to_ms = source.decoder.time_base().numerator() as f64
            / source.decoder.time_base().denominator() as f64
            * 1000.;
        // todo: fix the timebase is 0
        let time_base_num = if source.decoder.time_base().numerator() == 0 {
            1
        } else {
            source.decoder.time_base().numerator()
        };
        let filter_desc_in = format!(
            "buffer=video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
            source.decoder.width(),
            source.decoder.height(),
            ffmpeg_next::ffi::AVPixelFormat::from(source.decoder.format()) as u32,
            time_base_num,
            source.decoder.time_base().denominator(),
            source.decoder.aspect_ratio().numerator(),
            source.decoder.aspect_ratio().denominator()
        );
        Ok(Self {
            source,
            filter_desc_in,
            convert_to_ms,
            scales: RefCell::new(Default::default()),
            scale_frame_buffer: RefCell::new(ffmpeg_next::frame::Video::empty()),
        })
    }
}

impl VideoData for FFMpegVideoData {
    type Frame<'frame> = FFMpegFrame<'frame>;

    #[inline(always)]
    fn next_frame(&mut self) -> Result<Option<Self::Frame<'_>>, Error> {
        if !self.source.receive_frame()? {
            return Ok(None);
        }
        Ok(Some(FFMpegFrame(self)))
    }
}

pub struct FFMpegFrame<'a>(&'a mut FFMpegVideoData);

impl<'a> ImageToTensor for FFMpegFrame<'a> {
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        to_tensor_info: &ImageToTensorInfo,
        process_options: &ImageProcessingOptions,
        output_buffer: &mut T,
    ) -> Result<(), Error> {
        let src_width = self.0.source.frame.width();
        let src_height = self.0.source.frame.height();
        let mut scale_frame_buffer = self.0.scale_frame_buffer.borrow_mut();

        // crop and rotate, then scale using filter
        let data = if process_options.rotation != 0. || process_options.region_of_interest.is_some()
        {
            const IN_NODE: &'static str = "Parsed_buffer_0";
            const OUT_NODE_PREFIX: &'static str = "Parsed_buffersink_";
            let mut num_node = 3;

            // config filter desc
            let mut desc = if let Some(ref roi) = process_options.region_of_interest {
                let crop_x = (src_width as f32 * roi.x_min) as u32;
                let crop_y = (src_height as f32 * roi.y_min) as u32;
                let crop_w = (src_width as f32 * roi.width) as u32;
                let crop_h = (src_height as f32 * roi.height) as u32;
                num_node += 1;
                format!(
                    "[c_in];[c_in]crop={}:{}:{}:{}",
                    crop_w, crop_h, crop_x, crop_y
                )
            } else {
                String::new()
            };
            if process_options.rotation != 0. {
                num_node += 1;
                desc.extend(format!("[r_in];[r_in]rotate={}", process_options.rotation).chars());
            }
            let out_format = match to_tensor_info.color_space {
                ImageColorSpaceType::GRAYSCALE => {
                    ffmpeg_next::ffi::AVPixelFormat::AV_PIX_FMT_GRAY8 as u32
                }
                _ => ffmpeg_next::ffi::AVPixelFormat::AV_PIX_FMT_RGB24 as u32,
            };
            desc.extend(
                format!(
                    "[s_in];[s_in]scale={}:{}[f];[f]format=pix_fmts={}[out];[out]buffersink",
                    to_tensor_info.width(),
                    to_tensor_info.height(),
                    out_format
                )
                .chars(),
            );

            let mut filter_graph = ffmpeg_next::filter::Graph::new();
            filter_graph.parse(format!("{}{}", self.0.filter_desc_in, desc).as_str())?;
            filter_graph.validate()?;
            filter_graph
                .get(IN_NODE)
                .unwrap()
                .source()
                .add(&self.0.source.frame)?;
            filter_graph
                .get(format!("{}{}", OUT_NODE_PREFIX, num_node).as_str())
                .unwrap()
                .sink()
                .frame(&mut scale_frame_buffer)?;
            scale_frame_buffer.data(0)
        } else {
            // scale and convert format
            let scale_key = ScaleKey {
                src_w: src_width,
                src_h: src_height,
                dst_w: to_tensor_info.width(),
                dst_h: to_tensor_info.height(),
                dst_format: to_tensor_info.color_space,
            };
            let src_format = self.0.source.frame.format();
            let mut scales_cache = self.0.scales.borrow_mut();
            if let Some(scale_ctx) = cached_scale_ctx(&mut scales_cache, src_format, scale_key) {
                // scale frame
                scale_ctx.run(&self.0.source.frame, &mut scale_frame_buffer)?;
                scale_frame_buffer.data(0)
            } else {
                self.0.source.frame.data(0)
            }
        };

        match to_tensor_info.color_space {
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                let img = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
                    to_tensor_info.width(),
                    to_tensor_info.height(),
                    data,
                )
                .unwrap();
                image::rgb8_image_buffer_to_tensor(&img, to_tensor_info, output_buffer)?;
            }
            ImageColorSpaceType::GRAYSCALE => {
                todo!("gray image")
            }
        }

        Ok(())
    }

    /// return image size: (weight, height)
    fn image_size(&self) -> (u32, u32) {
        (
            self.0.source.decoder.width(),
            self.0.source.decoder.height(),
        )
    }

    /// return the current timestamp (ms)
    fn timestamp_ms(&self) -> Option<u64> {
        self.0
            .source
            .frame
            .timestamp()
            .map(|t| (t as f64 * self.0.convert_to_ms) as u64)
    }
}

#[derive(Hash, Eq, PartialEq)]
struct ScaleKey {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    dst_format: ImageColorSpaceType,
}

// get cached scale context
fn cached_scale_ctx(
    scales: &mut HashMap<ScaleKey, ffmpeg_next::software::scaling::Context>,
    src_format: ffmpeg_next::format::Pixel,
    key: ScaleKey,
) -> Option<&mut ffmpeg_next::software::scaling::Context> {
    // do not need to scale
    if key.src_w == key.dst_w
        && key.src_h == key.dst_h
        && match key.dst_format {
            ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                src_format == ffmpeg_next::format::Pixel::RGB24
            }
            ImageColorSpaceType::GRAYSCALE => src_format == ffmpeg_next::format::Pixel::GRAY8,
        }
    {
        return None;
    }

    Some(match scales.entry(key) {
        Entry::Occupied(s) => s.into_mut(),
        Entry::Vacant(v) => {
            // new scale context
            let key = v.key();
            let dst_format = match key.dst_format {
                ImageColorSpaceType::RGB | ImageColorSpaceType::UNKNOWN => {
                    ffmpeg_next::format::Pixel::RGB24
                }
                ImageColorSpaceType::GRAYSCALE => ffmpeg_next::format::Pixel::GRAY8,
            };
            let scale = ffmpeg_next::software::scaling::Context::get(
                src_format,
                key.src_w,
                key.src_h,
                dst_format,
                key.dst_w,
                key.dst_h,
                ffmpeg_next::software::scaling::Flags::BITEXACT
                    | ffmpeg_next::software::scaling::Flags::SPLINE,
            )
            .unwrap();
            v.insert(scale)
        }
    })
}
