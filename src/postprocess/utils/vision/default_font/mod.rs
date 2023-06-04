use rusttype::Font;

const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("./ascii.ttf");

lazy_static::lazy_static! {
    static ref DEFAULT_FONT: Font<'static> = Font::try_from_bytes(DEFAULT_FONT_BYTES).unwrap();
}

/// Get default ascii font
pub fn default_font() -> &'static Font<'static> {
    &DEFAULT_FONT
}
