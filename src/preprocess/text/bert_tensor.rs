// These references files are licensed under Apache 2.0, and originally developed by Google:
// * https://github.com/tensorflow/text/blob/master/tensorflow_text/core/kernels/wordpiece_tokenizer.cc
// * https://github.com/tensorflow/text/blob/master/tensorflow_text/core/kernels/regex_split.cc
// * https://github.com/google/mediapipe/blob/master/mediapipe/tasks/cc/text/tokenizers/bert_tokenizer.cc

use super::*;

lazy_static::lazy_static! {
    static ref DELIM_REGEX: Regex = Regex::new(r"((\s+|[!-/]|[:-@]|[\[-`]|[{-~]|[\p{P}]|[\x{4E00}-\x{9FFF}]|[\x{3400}-\x{4DBF}]|[\x{20000}-\x{2A6DF}]|[\x{2A700}-\x{2B73F}]|[\x{2B740}-\x{2B81F}]|[\x{2B820}-\x{2CEAF}]|[\x{F900}-\x{FAFF}]|[\x{2F800}-\x{2FA1F}]))").unwrap();
    static ref INCLUDE_DELIM_REGEX: Regex = Regex::new(r"(([!-/]|[:-@]|[\[-`]|[{-~]|[\p{P}]|[\x{4E00}-\x{9FFF}]|[\x{3400}-\x{4DBF}]|[\x{20000}-\x{2A6DF}]|[\x{2A700}-\x{2B73F}]|[\x{2B740}-\x{2B81F}]|[\x{2B820}-\x{2CEAF}]|[\x{F900}-\x{FAFF}]|[\x{2F800}-\x{2FA1F}]))").unwrap();
}

pub(super) fn to_bert_tensors<T: AsMut<[E]>, E: AsMut<[u8]>>(
    s: &str,
    token_index_map: &HashMap<String, i32>,
    output_buffers: &mut T,
    max_seq_len: u32,
    classifier_token_id: i32,
    separator_token_id: i32,
) -> Result<(), Error> {
    // check outputs
    if output_buffers.as_mut().len() != 3 {
        return Err(Error::ModelInconsistentError(format!(
            "Bert model input must be `3` tensors, but got `{}`",
            output_buffers.as_mut().len()
        )));
    }
    let indices_size = max_seq_len as usize;
    let min_bytes = indices_size * std::mem::size_of::<i32>();
    for i in 0..3 {
        if output_buffers.as_mut()[i].as_mut().len() < min_bytes {
            return Err(Error::ModelInconsistentError(format!(
                "Expect input buffer `{}` at least `{}` bytes, but got `{}`",
                i,
                min_bytes,
                output_buffers.as_mut()[i].as_mut().len()
            )));
        }
    }
    // get buffer
    let input_ids = unsafe {
        core::slice::from_raw_parts_mut(
            output_buffers.as_mut()[0].as_mut().as_mut_ptr() as *mut i32,
            indices_size,
        )
    };
    let segment_ids = &mut output_buffers.as_mut()[1].as_mut()[..min_bytes];
    segment_ids.fill(0);
    let input_masks = unsafe {
        core::slice::from_raw_parts_mut(
            output_buffers.as_mut()[2].as_mut().as_mut_ptr() as *mut i32,
            indices_size,
        )
    };
    let mut index = 0;
    // [CLS]
    input_ids[index] = classifier_token_id;
    index += 1;

    // split string
    let mut string = s.to_ascii_lowercase();

    let mut now = string.as_mut_str();
    while let Some(m) = DELIM_REGEX.find(now) {
        let start = m.start();
        let end = m.end();
        if start != 0 {
            let token = &now[..start];
            do_word_piece_tokenize(input_ids, &mut index, token, token_index_map);
            if index >= indices_size {
                break;
            }
        }

        let delim_token = &now[start..end];
        // include delim token
        if INCLUDE_DELIM_REGEX.is_match(delim_token) {
            do_word_piece_tokenize(input_ids, &mut index, delim_token, token_index_map);
            if index >= indices_size {
                break;
            }
        }

        (_, now) = now.split_at_mut(end);
    }

    if !now.is_empty() {
        do_word_piece_tokenize(input_ids, &mut index, now, token_index_map);
    }

    // the last [SEP]
    if index < indices_size {
        // now is just replace the last token
        input_ids[index] = separator_token_id;
        index += 1;
    } else {
        input_ids[indices_size - 1] = separator_token_id;
    }

    // fill rest
    input_masks[..index].fill(1);
    if index < indices_size {
        input_ids[index..].fill(0);
        input_masks[index..].fill(0);
    }
    Ok(())
}

const DEFAULT_MAX_BYTES_PER_TOKEN: usize = 100;
// const DEFAULT_MAX_CHARS_PER_SUB_TOKEN: i32 = 100;
const DEFAULT_SUFFIX_INDICATOR: &'static str = "##";
// const DEFAULT_USE_UNKNOWN_TOKEN: bool = true;
const DEFAULT_UNKNOWN_TOKEN: &'static str = "[UNK]";
// const DEFAULT_SPLIT_UNKNOWN_CHARS: bool = false;

#[inline(always)]
fn do_word_piece_tokenize(
    input_ids: &mut [i32],
    index: &mut usize,
    token: &str,
    token_index_map: &HashMap<String, i32>,
) {
    if *index >= input_ids.len() {
        return;
    }

    if token.as_bytes().len() > DEFAULT_MAX_BYTES_PER_TOKEN {
        // use unknown token
        input_ids[*index] = *token_index_map.get(DEFAULT_UNKNOWN_TOKEN).unwrap_or(&0);
        *index += 1;
        if *index >= input_ids.len() {
            return;
        }
    }

    let token_len = token.len();
    // use string buffer to save temp string concat result
    let mut string_buffer = String::with_capacity(token_len + DEFAULT_SUFFIX_INDICATOR.len());
    let mut token_start = 0;
    while token_start < token_len {
        if let Some((token_end, token_index)) =
            longest_match_starting_at(token, token_start, token_index_map, &mut string_buffer)
        {
            // add sub word, and the token_index is the corresponding index
            input_ids[*index] = token_index;
            *index += 1;
            if *index >= input_ids.len() {
                return;
            }
            token_start = token_end
        } else {
            // no token found
            // default is using unknown token
            input_ids[*index] = *token_index_map.get(DEFAULT_UNKNOWN_TOKEN).unwrap_or(&0);
            *index += 1;
            return;
        }
    }
}

// return Option<(token_end, token_index)>
fn longest_match_starting_at(
    token: &str,
    token_start: usize,
    token_index_map: &HashMap<String, i32>,
    string_buffer: &mut String,
) -> Option<(usize, i32)> {
    // token: &str is a valid utf-8 string and indexed by utf-8 chars
    let mut token_end = token.len();
    while token_end > token_start {
        let str_to_lookup = if token_start > 0 {
            string_buffer.clear();
            string_buffer.extend(DEFAULT_SUFFIX_INDICATOR.chars());
            string_buffer.extend(token[token_start..token_end].chars());
            string_buffer.as_str()
        } else {
            &token[..token_end]
        };
        match token_index_map.get(str_to_lookup) {
            Some(token_index) => return Some((token_end, *token_index)),
            None => {} // default split unknown characters is false, so do nothing
        }
        token_end -= 1;
    }
    // no token found
    None
}
