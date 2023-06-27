use wasi_nn::Error as WasiNNError;

/// MediaPipe-rs API error enum.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Wasi-NN Error: {0}")]
    WasiNNError(#[from] WasiNNError),

    #[error("Argument Error: {0}")]
    ArgumentError(String),

    #[error("Model Binary Parse Error: {0}")]
    ModelParseError(String),

    #[error("ZIP File Parse Error: {0}")]
    ZipFileParseError(String),

    #[error("Model Inconsistent Error: {0}")]
    ModelInconsistentError(String),

    #[error("FlatBuffer Error: {0}")]
    FlatBufferError(#[from] flatbuffers::InvalidFlatbuffer),

    #[cfg(feature = "ffmpeg")]
    #[error("FFMpeg Error: {0}")]
    FFMpegError(#[from] ffmpeg_next::Error),

    #[cfg(feature = "audio")]
    #[error("Symphonia Error: {0}")]
    SymphoniaError(#[from] symphonia_core::errors::Error),
}
