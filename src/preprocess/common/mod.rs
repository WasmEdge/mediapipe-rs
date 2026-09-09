#[cfg(feature = "ffmpeg")]
pub(super) mod ffmpeg_input;

/// Write `values` as native-endian bytes into `out`. `out` must hold at least
/// `values.len() * size_of::<T>()` bytes.
#[cfg(any(feature = "audio", feature = "text"))]
pub(crate) fn write_ne_bytes<T: ToNeBytes>(out: &mut [u8], values: impl IntoIterator<Item = T>) {
    let mut slots = out.chunks_exact_mut(std::mem::size_of::<T>());
    for value in values {
        slots
            .next()
            .expect("output buffer length is checked by the caller")
            .copy_from_slice(value.to_ne_bytes().as_ref());
    }
}

#[cfg(any(feature = "audio", feature = "text"))]
pub(crate) trait ToNeBytes: Copy {
    type Bytes: AsRef<[u8]>;
    fn to_ne_bytes(self) -> Self::Bytes;
}

#[cfg(any(feature = "audio", feature = "text"))]
macro_rules! impl_to_ne_bytes {
    ( $( $t:ty ),* ) => {
        $(
            impl ToNeBytes for $t {
                type Bytes = [u8; std::mem::size_of::<$t>()];
                #[inline(always)]
                fn to_ne_bytes(self) -> Self::Bytes {
                    <$t>::to_ne_bytes(self)
                }
            }
        )*
    };
}

#[cfg(any(feature = "audio", feature = "text"))]
impl_to_ne_bytes!(i32, f32);
