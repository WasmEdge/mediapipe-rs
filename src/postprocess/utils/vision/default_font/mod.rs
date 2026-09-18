// NOTICE
// The file `Roboto-Regulat.ttf` is downloaded from https://fonts.google.com/specimen/Roboto.
// This font file is under Apache 2.0: https://github.com/googlefonts/roboto/blob/main/LICENSE

use ab_glyph::FontArc;
use std::sync::LazyLock;

const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("./Roboto-Regular.ttf");

static DEFAULT_FONT: LazyLock<FontArc> =
    LazyLock::new(|| FontArc::try_from_slice(DEFAULT_FONT_BYTES).unwrap());

/// Get default ascii font
pub fn default_font() -> &'static FontArc {
    &DEFAULT_FONT
}
