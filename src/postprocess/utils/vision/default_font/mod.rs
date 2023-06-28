// NOTICE
// The file `Roboto-Regulat.ttf` is downloaded from https://fonts.google.com/specimen/Roboto.
// This font file is under Apache 2.0: https://github.com/googlefonts/roboto/blob/main/LICENSE

use rusttype::Font;

const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("./Roboto-Regular.ttf");

lazy_static::lazy_static! {
    static ref DEFAULT_FONT: Font<'static> = Font::try_from_bytes(DEFAULT_FONT_BYTES).unwrap();
}

/// Get default ascii font
pub fn default_font() -> &'static Font<'static> {
    &DEFAULT_FONT
}
