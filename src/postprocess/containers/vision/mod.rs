macro_rules! min_f32 {
    ( $x:expr, $y:expr ) => {{
        let diff = $x - $y;
        if diff < 0. {
            $x
        } else {
            $y
        }
    }};
}

macro_rules! max_f32 {
    ( $x:expr, $y:expr ) => {{
        let diff = $x - $y;
        if diff > 0. {
            $x
        } else {
            $y
        }
    }};
}

mod crop_rect;
mod detection_result;
mod image_mask;
mod key_point;
mod landmark;
mod normalized_rect;
mod rect;
mod results_iter;

pub use crop_rect::*;
pub use detection_result::*;
pub use image_mask::*;
pub use key_point::*;
pub use landmark::*;
pub use normalized_rect::*;
pub use rect::*;
pub use results_iter::*;
