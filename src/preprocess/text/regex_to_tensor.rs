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
    let min_bytes = token_ids_bytes(max_seq_len)?;
    if output_buffer.as_mut().len() < min_bytes {
        return Err(Error::ModelInconsistentError(format!(
            "Output buffer bytes is too small, expect `{}` but got `{}`",
            min_bytes,
            output_buffer.as_mut().len()
        )));
    }

    let start = token_index_map
        .get(TextToTensorInfo::REGEX_START_TOKEN)
        .copied();
    let words = delim_regex
        .split(s)
        .map(|word| *token_index_map.get(word).unwrap_or(&unknown_id));
    let ids = start
        .into_iter()
        .chain(words)
        .chain(std::iter::repeat(pad_id))
        .take(indices_size);
    write_ne_bytes(output_buffer.as_mut(), ids);
    Ok(())
}
