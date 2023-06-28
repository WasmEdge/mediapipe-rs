use crate::Error;
use ffmpeg_next::codec::Context;
use ffmpeg_next::decoder::Opened;
use ffmpeg_next::format::context::Input;
use std::ops::DerefMut;

pub struct FFMpegInput<Decoder, Frame>
where
    Decoder: DerefMut<Target = Opened> + AsMut<Context>,
    Frame: DerefMut<Target = ffmpeg_next::frame::Frame>,
{
    input: Input,
    pub decoder: Decoder,
    pub frame: Frame,
    input_stream_index: usize,
    decoder_has_sent_eof: bool,
}

macro_rules! impl_new_func {
    ( $decoder:ident, $frame:ident, $decode_func:ident, $stream_type:ident ) => {
        impl FFMpegInput<ffmpeg_next::decoder::$decoder, ffmpeg_next::frame::$frame> {
            #[inline(always)]
            pub fn new(input: Input) -> Result<Self, Error> {
                let input_stream = input
                    .streams()
                    .best(ffmpeg_next::media::Type::$stream_type)
                    .ok_or(Error::ArgumentError(format!(
                        "Input Has no stream: `{:?}`.",
                        ffmpeg_next::media::Type::$stream_type
                    )))?;
                let input_stream_index = input_stream.index();
                let context = Context::from_parameters(input_stream.parameters())?;
                let mut decoder = context.decoder().$decode_func()?;
                decoder.set_parameters(input_stream.parameters())?;
                Ok(Self {
                    input,
                    decoder,
                    frame: ffmpeg_next::frame::$frame::empty(),
                    input_stream_index,
                    decoder_has_sent_eof: false,
                })
            }
        }
    };
}

#[cfg(feature = "vision")]
impl_new_func!(Video, Video, video, Video);
#[cfg(feature = "audio")]
impl_new_func!(Audio, Audio, audio, Audio);

impl<Decoder, Frame> FFMpegInput<Decoder, Frame>
where
    Decoder: DerefMut<Target = Opened> + AsMut<Context>,
    Frame: DerefMut<Target = ffmpeg_next::frame::Frame>,
{
    const RESOURCE_TEMPORARILY_UNAVAILABLE: ffmpeg_next::Error = ffmpeg_next::Error::Other {
        errno: ffmpeg_next::util::error::EAGAIN,
    };

    #[inline(always)]
    pub fn receive_frame(&mut self) -> Result<bool, Error> {
        while let Err(err) = self.decoder.receive_frame(&mut self.frame) {
            if err == Self::RESOURCE_TEMPORARILY_UNAVAILABLE {
                if !self.decoder_has_sent_eof {
                    let mut is_eof = true;
                    while let Some((stream, package)) = self.input.packets().next() {
                        if stream.index() != self.input_stream_index {
                            continue;
                        }
                        self.decoder.send_packet(&package)?;
                        is_eof = false;
                        break;
                    }
                    if is_eof {
                        self.decoder.send_eof()?;
                        self.decoder_has_sent_eof = true;
                    }
                }
            } else {
                return if let ffmpeg_next::Error::Eof = err {
                    Ok(false)
                } else {
                    Err(Error::from(err))
                };
            }
        }

        Ok(true)
    }
}
