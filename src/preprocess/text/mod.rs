mod bert_tensor;
mod regex_to_tensor;

use super::*;
use regex::Regex;
use std::borrow::Cow;
use std::collections::HashMap;

/// Text model input interface. Every Text data implement the [`TextToTensors`] trait can be used as text tasks input.
/// Now the builtin impl: [`str`], [`String`], [`Cow<'a, str>`].
pub trait TextToTensors {
    fn to_tensors<T: AsMut<[E]>, E: AsMut<[u8]>>(
        &self,
        to_tensor_info: &TextToTensorInfo,
        output_buffers: &mut T,
    ) -> Result<(), Error>;
}

/// Necessary information for the text to tensors.
#[derive(Debug)]
pub enum TextToTensorInfo {
    /// A BERT-based model.
    BertModel {
        token_index_map: HashMap<String, i32>,

        /// maximum input sequence length for the bert and regex model.
        max_seq_len: u32,

        classifier_token_id: i32,
        separator_token_id: i32,
    },
    /// A model expecting input passed through a regex-based tokenizer.
    RegexModel {
        /// delim regex pattern
        delim_regex: Regex,

        token_index_map: HashMap<String, i32>,

        /// maximum input sequence length for the bert and regex model.
        max_seq_len: u32,

        unknown_id: i32,
        pad_id: i32,
    },
    /// A model taking a string tensor input.
    StringModel,
    /// A UniversalSentenceEncoder-based model.
    UseModel,
}

/// Byte size of `max_seq_len` token ids; fails when it does not fit in `usize`.
fn token_ids_bytes(max_seq_len: u32) -> Result<usize, Error> {
    (max_seq_len as usize)
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| {
            Error::ModelInconsistentError(format!(
                "Max seq length `{}` is too large for this target",
                max_seq_len
            ))
        })
}

macro_rules! check_map {
    ( $token_index_map:ident, $val:expr ) => {
        match $token_index_map.get($val) {
            Some(v) => v.clone(),
            None => {
                return Err(Error::ModelInconsistentError(format!(
                    "Vocabulary file doesn't have `{}` token.",
                    $val
                )));
            }
        }
    };
}

impl TextToTensorInfo {
    pub const REGEX_START_TOKEN: &'static str = "<START>";
    pub const REGEX_PAD_TOKEN: &'static str = "<PAD>";
    pub const REGEX_UNKNOWN_TOKEN: &'static str = "<UNKNOWN>";
    pub const BERT_CLASSIFIER_TOKEN: &'static str = "[CLS]";
    pub const BERT_SEPARATOR_TOKEN: &'static str = "[SEP]";

    pub fn new_regex_model(
        max_seq_len: u32,
        delim_regex_pattern: &str,
        token_index_map: HashMap<String, i32>,
    ) -> Result<Self, Error> {
        // rust regex has no \'
        let delim_regex_pattern = delim_regex_pattern.replace(r"\'", r"'");
        let delim_regex =
            Regex::new(format!("({})", delim_regex_pattern).as_str()).map_err(|e| {
                Error::ModelInconsistentError(format!(
                    "Cannot parse delim regex pattern: `{:?}`",
                    e
                ))
            })?;
        let pad_id = check_map!(token_index_map, Self::REGEX_PAD_TOKEN);
        let unknown_id = check_map!(token_index_map, Self::REGEX_UNKNOWN_TOKEN);
        Ok(Self::RegexModel {
            delim_regex,
            token_index_map,
            max_seq_len,
            unknown_id,
            pad_id,
        })
    }

    pub fn new_bert_model(
        max_seq_len: u32,
        token_index_map: HashMap<String, i32>,
    ) -> Result<Self, Error> {
        if max_seq_len < 2 {
            return Err(Error::ModelInconsistentError(
                "Bert model max seq length must be at least `2`".into(),
            ));
        }
        let classifier_token_id = check_map!(token_index_map, Self::BERT_CLASSIFIER_TOKEN);
        let separator_token_id = check_map!(token_index_map, Self::BERT_SEPARATOR_TOKEN);
        Ok(Self::BertModel {
            max_seq_len,
            token_index_map,
            classifier_token_id,
            separator_token_id,
        })
    }
}

impl TextToTensors for &str {
    fn to_tensors<T: AsMut<[E]>, E: AsMut<[u8]>>(
        &self,
        to_tensor_info: &TextToTensorInfo,
        output_buffers: &mut T,
    ) -> Result<(), Error> {
        match to_tensor_info {
            TextToTensorInfo::BertModel {
                max_seq_len,
                token_index_map,
                classifier_token_id,
                separator_token_id,
                ..
            } => {
                debug_assert_eq!(output_buffers.as_mut().len(), 3);
                return bert_tensor::to_bert_tensors(
                    self,
                    token_index_map,
                    output_buffers,
                    *max_seq_len,
                    *classifier_token_id,
                    *separator_token_id,
                );
            }
            TextToTensorInfo::RegexModel {
                delim_regex,
                token_index_map,
                max_seq_len,
                unknown_id,
                pad_id,
                ..
            } => {
                debug_assert_eq!(output_buffers.as_mut().len(), 1);
                return regex_to_tensor::regex_to_tensors(
                    self,
                    delim_regex,
                    token_index_map,
                    &mut output_buffers.as_mut()[0],
                    *max_seq_len,
                    *unknown_id,
                    *pad_id,
                );
            }
            TextToTensorInfo::StringModel | TextToTensorInfo::UseModel => {
                Err(Error::ModelInconsistentError(
                    "String tensor and Universal Sentence Encoder text models are not supported"
                        .into(),
                ))
            }
        }
    }
}

impl TextToTensors for String {
    #[inline(always)]
    fn to_tensors<T: AsMut<[E]>, E: AsMut<[u8]>>(
        &self,
        to_tensor_info: &TextToTensorInfo,
        output_buffers: &mut T,
    ) -> Result<(), Error> {
        self.as_str().to_tensors(to_tensor_info, output_buffers)
    }
}

impl<'a> TextToTensors for Cow<'a, str> {
    #[inline(always)]
    fn to_tensors<T: AsMut<[E]>, E: AsMut<[u8]>>(
        &self,
        to_tensor_info: &TextToTensorInfo,
        output_buffers: &mut T,
    ) -> Result<(), Error> {
        match self {
            Cow::Borrowed(s) => (*s).to_tensors(to_tensor_info, output_buffers),
            Cow::Owned(s) => s.to_tensors(to_tensor_info, output_buffers),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn map(pairs: &[(&str, i32)]) -> HashMap<String, i32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn to_i32(buf: &[u8]) -> Vec<i32> {
        buf.chunks_exact(4)
            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_regex_model_to_tensors() {
        let vocab = map(&[("<START>", 1), ("<PAD>", 0), ("<UNKNOWN>", 2), ("hello", 3)]);
        let info = TextToTensorInfo::new_regex_model(4, r"\s+", vocab.clone()).unwrap();
        let mut buffers = [vec![0u8; 16]];
        "hello world".to_tensors(&info, &mut buffers).unwrap();
        assert_eq!(to_i32(&buffers[0]), [1, 3, 2, 0]);

        "a b c d e f".to_tensors(&info, &mut buffers).unwrap();
        assert_eq!(to_i32(&buffers[0]), [1, 2, 2, 2]);

        let mut short = [vec![0u8; 15]];
        assert!("hello".to_tensors(&info, &mut short).is_err());

        // the sequence limit also applies to the start token
        let info = TextToTensorInfo::new_regex_model(1, r"\s+", vocab.clone()).unwrap();
        let mut one = [vec![0u8; 4]];
        "hello world".to_tensors(&info, &mut one).unwrap();
        assert_eq!(to_i32(&one[0]), [1]);

        let info = TextToTensorInfo::new_regex_model(0, r"\s+", vocab).unwrap();
        let mut none = [Vec::<u8>::new()];
        "hello world".to_tensors(&info, &mut none).unwrap();
    }

    #[test]
    fn test_bert_model_to_tensors() {
        let info = TextToTensorInfo::new_bert_model(
            5,
            map(&[
                ("[CLS]", 101),
                ("[SEP]", 102),
                ("[UNK]", 100),
                ("hello", 7592),
                ("world", 2088),
            ]),
        )
        .unwrap();
        let mut buffers = [vec![1u8; 20], vec![1u8; 20], vec![1u8; 20]];
        "Hello world".to_tensors(&info, &mut buffers).unwrap();
        assert_eq!(to_i32(&buffers[0]), [101, 7592, 2088, 102, 0]);
        assert_eq!(to_i32(&buffers[1]), [0, 0, 0, 0, 0]);
        assert_eq!(to_i32(&buffers[2]), [1, 1, 1, 1, 0]);

        "hello world hello world hello"
            .to_tensors(&info, &mut buffers)
            .unwrap();
        assert_eq!(to_i32(&buffers[0]), [101, 7592, 2088, 7592, 102]);
        assert_eq!(to_i32(&buffers[2]), [1, 1, 1, 1, 1]);
    }

    /// `max_seq_len * 4` overflows `usize` on wasm32; the buffer check must reject it
    /// instead of allocating the token ids.
    #[test]
    fn test_huge_max_seq_len_is_rejected() {
        let max_seq_len = 1 << 30;
        let regex = TextToTensorInfo::new_regex_model(
            max_seq_len,
            r"\s+",
            map(&[("<START>", 1), ("<PAD>", 0), ("<UNKNOWN>", 2)]),
        )
        .unwrap();
        let mut buffers = [vec![0u8; 16]];
        assert!("hello".to_tensors(&regex, &mut buffers).is_err());

        let bert =
            TextToTensorInfo::new_bert_model(max_seq_len, map(&[("[CLS]", 101), ("[SEP]", 102)]))
                .unwrap();
        let mut buffers = [vec![0u8; 16], vec![0u8; 16], vec![0u8; 16]];
        assert!("hello".to_tensors(&bert, &mut buffers).is_err());
    }
}
