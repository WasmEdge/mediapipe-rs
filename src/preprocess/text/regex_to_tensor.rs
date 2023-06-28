use super::*;

pub(super) fn regex_to_tensors<E: AsMut<[u8]>>(
    s: &str,
    delim_regex: &Regex,
    token_index_map: &HashMap<String, i32>,
    output_buffer: &mut E,
    max_seq_len: u32,
    unknown_id: i32,
    pad_id: i32,
) -> Result<(), Error> {
    let indices_size = max_seq_len as usize;
    if output_buffer.as_mut().len() < std::mem::size_of::<i32>() * indices_size {
        return Err(Error::ModelInconsistentError(format!(
            "Output buffer bytes is too small, expect `{}` but got `{}`",
            std::mem::size_of::<i32>() * indices_size,
            output_buffer.as_mut().len()
        )));
    }

    let buffer = unsafe {
        core::slice::from_raw_parts_mut(
            output_buffer.as_mut().as_mut_ptr() as *mut i32,
            indices_size,
        )
    };

    let mut index = 0;
    if let Some(id) = token_index_map.get(TextToTensorInfo::REGEX_START_TOKEN) {
        buffer[index] = *id;
        index += 1;
    }

    for words in delim_regex.split(s) {
        buffer[index] = *token_index_map.get(words).unwrap_or(&unknown_id);
        index += 1;
        if index == indices_size {
            break;
        }
    }

    while index < indices_size {
        buffer[index] = pad_id;
        index += 1;
    }

    Ok(())
}
