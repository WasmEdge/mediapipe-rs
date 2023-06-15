// ZIP format spec: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT

use crate::Error;
use std::borrow::Cow;
use std::collections::HashMap;

macro_rules! read_le_u16 {
    ( $buf:expr, $offset:expr ) => {{
        let start = $offset;
        let low = $buf[start] as u16;
        let high = $buf[start + 1] as u16;
        (high << 8) | low
    }};
}

macro_rules! read_le_u32 {
    ( $buf:expr, $offset:expr ) => {{
        let start = $offset;
        let arr = [
            $buf[start],
            $buf[start + 1],
            $buf[start + 2],
            $buf[start + 3],
        ];
        u32::from_le_bytes(arr)
    }};
}

// todo: support zip64
/*
4.3.16  End of central directory record:

      end of central dir signature    4 bytes  (0x06054b50)
      number of this disk             2 bytes
      number of the disk with the
      start of the central directory  2 bytes
      total number of entries in the
      central directory on this disk  2 bytes
      total number of entries in
      the central directory           2 bytes
      size of the central directory   4 bytes
      offset of start of central
      directory with respect to
      the starting disk number        4 bytes
      .ZIP file comment length        2 bytes
      .ZIP file comment       (variable size)
 */
struct EndOfCentralDirectoryRecord<'buf> {
    buf: &'buf [u8],
}

impl<'buf> EndOfCentralDirectoryRecord<'buf> {
    const HEAD_SIGNATURE: &'static [u8] = &[0x50, 0x4b, 0x05, 0x06];
    const MIN_SIZE: usize = 20; // at least 20 bytes

    const NUMBER_OF_THIS_DISK_POS: usize = 4;
    const DISK_WHERE_CENTRAL_DIRECTORY_STARTS_POS: usize = 6;
    const NUMBER_OF_CENTRAL_DIRECTORY_RECORDS_ON_THIS_DISK_POS: usize = 8;
    const TOTAL_NUMBER_OF_CENTRAL_DIRECTORY_RECORDS_POS: usize = 10;
    const SIZE_OF_CENTRAL_DIRECTORY_POS: usize = 12;
    const OFFSET_OF_START_OF_CENTRAL_DIRECTORY_POS: usize = 16;
    const COMMENT_LENGTH_POS: usize = 20;
    const COMMENT_POS: usize = 22;

    #[inline(always)]
    fn number_of_this_disk(&self) -> u16 {
        read_le_u16!(self.buf, Self::NUMBER_OF_THIS_DISK_POS)
    }

    #[inline(always)]
    fn disk_where_central_directory_starts(&self) -> u16 {
        read_le_u16!(self.buf, Self::DISK_WHERE_CENTRAL_DIRECTORY_STARTS_POS)
    }

    #[inline(always)]
    fn number_of_central_directory_records_on_this_disk(&self) -> u16 {
        read_le_u16!(
            self.buf,
            Self::NUMBER_OF_CENTRAL_DIRECTORY_RECORDS_ON_THIS_DISK_POS
        )
    }

    #[inline(always)]
    fn total_number_of_central_directory_records(&self) -> u16 {
        read_le_u16!(
            self.buf,
            Self::TOTAL_NUMBER_OF_CENTRAL_DIRECTORY_RECORDS_POS
        )
    }

    #[inline(always)]
    fn size_of_central_directory(&self) -> u32 {
        read_le_u32!(self.buf, Self::SIZE_OF_CENTRAL_DIRECTORY_POS)
    }

    #[inline(always)]
    fn offset_of_start_of_central_directory(&self) -> u32 {
        read_le_u32!(self.buf, Self::OFFSET_OF_START_OF_CENTRAL_DIRECTORY_POS)
    }

    #[inline(always)]
    fn comment_length(&self) -> u16 {
        read_le_u16!(self.buf, Self::COMMENT_LENGTH_POS)
    }

    #[inline(always)]
    fn comment(&self) -> &'buf [u8] {
        &self.buf[Self::COMMENT_POS..Self::COMMENT_POS + self.comment_length() as usize]
    }

    #[inline(always)]
    fn try_find_start_pos(buf: &'buf [u8]) -> usize {
        let mut start_pos = match buf.len().checked_sub(Self::MIN_SIZE) {
            None => {
                return buf.len();
            }
            Some(p) => p,
        };
        loop {
            if &buf[start_pos..start_pos + 4] == Self::HEAD_SIGNATURE {
                return start_pos;
            }
            start_pos = match start_pos.checked_sub(1) {
                None => {
                    return buf.len();
                }
                Some(p) => p,
            };
        }
    }

    #[inline(always)]
    fn new_with_start_pos(buf: &'buf [u8], start_pos: usize) -> Result<Self, Error> {
        if start_pos >= buf.len() {
            return Err(Error::ZipFileParseError(format!(
                "End of central directory start pos error `{}` >= `{}`",
                start_pos,
                buf.len()
            )));
        }
        let res = Self {
            buf: &buf[start_pos..],
        };
        if res.number_of_this_disk() != 0
            || res.disk_where_central_directory_starts() != 0
            || res.number_of_central_directory_records_on_this_disk()
                != res.total_number_of_central_directory_records()
        {
            return Err(Error::ZipFileParseError(
                "Unsupported multi-disk zip file.".into(),
            ));
        }
        let central_directory = res.offset_of_start_of_central_directory() as usize;
        if central_directory >= start_pos
            || central_directory + res.size_of_central_directory() as usize > buf.len()
        {
            return Err(Error::ZipFileParseError(
                "Central directory information error".into(),
            ));
        }
        if start_pos + Self::COMMENT_POS + res.comment_length() as usize > buf.len() {
            return Err(Error::ZipFileParseError(
                "Comment length is too long".into(),
            ));
        }
        return Ok(res);
    }

    #[inline]
    fn new(buf: &'buf [u8]) -> Result<Self, Error> {
        let start_pos = Self::try_find_start_pos(buf);
        if start_pos == 0 {
            return Err(Error::ZipFileParseError(
                "Cannot find end of central directory.".into(),
            ));
        }
        Self::new_with_start_pos(buf, start_pos)
    }
}

/*
  4.3.12  Central directory structure:
     [central directory header 1]
     .
     .
     .
     [central directory header n]
     [digital signature]

     File header:
       central file header signature   4 bytes  (0x02014b50)
       version made by                 2 bytes
       version needed to extract       2 bytes
       general purpose bit flag        2 bytes
       compression method              2 bytes
       last mod file time              2 bytes
       last mod file date              2 bytes
       crc-32                          4 bytes
       compressed size                 4 bytes
       uncompressed size               4 bytes
       file name length                2 bytes
       extra field length              2 bytes
       file comment length             2 bytes
       disk number start               2 bytes
       internal file attributes        2 bytes
       external file attributes        4 bytes
       relative offset of local header 4 bytes

       file name (variable size)
       extra field (variable size)
       file comment (variable size)
*/
struct CentralDirectory<'buf> {
    buf: &'buf [u8],
}

impl<'buf> CentralDirectory<'buf> {
    const MIN_SIZE: usize = 46;
    const HEAD_SIGNATURE: &'static [u8] = &[0x50, 0x4b, 0x01, 0x02];
    const VERSION_MADE_BY_POS: usize = 4;
    const VERSION_NEEDED_TO_EXTRACT_POS: usize = 6;
    const GENERAL_PURPOSE_BIT_FLAG_POS: usize = 8;
    const COMPRESSION_METHOD_POS: usize = 10;
    const LAST_MOD_FILE_TIME_POS: usize = 12;
    const LAST_MOD_FILE_DATE_POS: usize = 14;
    const CRC_32_POS: usize = 16;
    const COMPRESSED_SIZE_POS: usize = 20;
    const UNCOMPRESSED_SIZE_POS: usize = 24;
    const FILE_NAME_LENGTH_POS: usize = 28;
    const EXTRA_FIELD_LENGTH_POS: usize = 30;
    const FILE_COMMENT_LENGTH_POS: usize = 32;
    const DISK_NUMBER_START_POS: usize = 34;
    const INTERNAL_FILE_ATTRIBUTES_POS: usize = 36;
    const EXTERNAL_FILE_ATTRIBUTES_POS: usize = 38;
    const RELATIVE_OFFSET_OF_LOCAL_HEADER_POS: usize = 42;
    const FILE_NAME_POS: usize = 46;

    #[inline]
    fn new(buf: &'buf [u8]) -> Result<Self, Error> {
        let len = buf.len();
        if len < Self::MIN_SIZE {
            return Err(Error::ZipFileParseError(format!(
                "Central Directory buffer size `{}` is too short!",
                len
            )));
        }

        if &buf[..4] != Self::HEAD_SIGNATURE {
            return Err(Error::ZipFileParseError(format!(
                "Invalid Central Directory head magic `{:?}`",
                &buf[..4]
            )));
        }

        let res = Self { buf };
        if len < res.size() {
            return Err(Error::ZipFileParseError(format!(
                "Buffer length `{}` is too short! file name length is `{}`, extra field length is `{}`, file comment length is `{}`",
                len, res.file_comment_length(), res.extra_field_length(), res.file_comment_length()
            )));
        }
        if res.disk_number_start() != 0 {
            return Err(Error::ZipFileParseError(format!(
                "Disk number must be `0`, but get `{}`. (not support zip64 now)",
                res.disk_number_start()
            )));
        }
        if res.compression_method() != 0 {
            return Err(Error::ZipFileParseError(
                "Only support `Store` compression method".into(),
            ));
        }
        let file_size = res.uncompressed_size() as usize;
        if res.compressed_size() as usize != file_size {
            return Err(Error::ZipFileParseError(format!(
                "Compressed size `{}` != Uncompressed size `{}`",
                res.compressed_size(),
                file_size
            )));
        }

        Ok(res)
    }

    #[inline(always)]
    fn version_made_by(&self) -> u16 {
        read_le_u16!(self.buf, Self::VERSION_MADE_BY_POS)
    }

    #[inline(always)]
    fn version_needed_to_extract(&self) -> u16 {
        read_le_u16!(self.buf, Self::VERSION_NEEDED_TO_EXTRACT_POS)
    }

    #[inline(always)]
    fn general_purpose_bit_flag(&self) -> u16 {
        read_le_u16!(self.buf, Self::GENERAL_PURPOSE_BIT_FLAG_POS)
    }

    #[inline(always)]
    fn compression_method(&self) -> u16 {
        read_le_u16!(self.buf, Self::COMPRESSION_METHOD_POS)
    }

    #[inline(always)]
    fn last_mod_file_time(&self) -> u16 {
        read_le_u16!(self.buf, Self::LAST_MOD_FILE_TIME_POS)
    }

    #[inline(always)]
    fn last_mod_file_date(&self) -> u16 {
        read_le_u16!(self.buf, Self::LAST_MOD_FILE_DATE_POS)
    }

    #[inline(always)]
    fn crc_32(&self) -> u32 {
        read_le_u32!(self.buf, Self::CRC_32_POS)
    }

    #[inline(always)]
    fn compressed_size(&self) -> u32 {
        read_le_u32!(self.buf, Self::COMPRESSED_SIZE_POS)
    }

    #[inline(always)]
    fn uncompressed_size(&self) -> u32 {
        read_le_u32!(self.buf, Self::UNCOMPRESSED_SIZE_POS)
    }

    #[inline(always)]
    fn file_name_length(&self) -> u16 {
        read_le_u16!(self.buf, Self::FILE_NAME_LENGTH_POS)
    }

    #[inline(always)]
    fn extra_field_length(&self) -> u16 {
        read_le_u16!(self.buf, Self::EXTRA_FIELD_LENGTH_POS)
    }

    #[inline(always)]
    fn file_comment_length(&self) -> u16 {
        read_le_u16!(self.buf, Self::FILE_COMMENT_LENGTH_POS)
    }

    #[inline(always)]
    fn disk_number_start(&self) -> u16 {
        read_le_u16!(self.buf, Self::DISK_NUMBER_START_POS)
    }

    #[inline(always)]
    fn internal_file_attributes(&self) -> u16 {
        read_le_u16!(self.buf, Self::INTERNAL_FILE_ATTRIBUTES_POS)
    }

    #[inline(always)]
    fn external_file_attributes(&self) -> u32 {
        read_le_u32!(self.buf, Self::EXTERNAL_FILE_ATTRIBUTES_POS)
    }

    #[inline(always)]
    fn relative_offset_of_local_header(&self) -> u32 {
        read_le_u32!(self.buf, Self::RELATIVE_OFFSET_OF_LOCAL_HEADER_POS)
    }

    #[inline(always)]
    fn file_name(&self) -> &'buf [u8] {
        &self.buf[Self::FILE_NAME_POS..Self::FILE_NAME_POS + self.file_name_length() as usize]
    }

    #[inline(always)]
    fn extra_field(&self) -> &'buf [u8] {
        let start = Self::FILE_NAME_POS + self.file_name_length() as usize;
        &self.buf[start..start + self.extra_field_length() as usize]
    }

    #[inline(always)]
    fn file_comment(&self) -> &'buf [u8] {
        let start = Self::FILE_NAME_POS
            + self.file_name_length() as usize
            + self.extra_field_length() as usize;
        &self.buf[start..start + self.file_name_length() as usize]
    }

    fn size(&self) -> usize {
        Self::FILE_NAME_POS
            + self.file_name_length() as usize
            + self.extra_field_length() as usize
            + self.file_comment_length() as usize
    }
}

/*
  local file header signature     4 bytes  (0x04034b50)
     version needed to extract       2 bytes
     general purpose bit flag        2 bytes
     compression method              2 bytes
     last mod file time              2 bytes
     last mod file date              2 bytes
     crc-32                          4 bytes
     compressed size                 4 bytes
     uncompressed size               4 bytes
     file name length                2 bytes
     extra field length              2 bytes

     file name (variable size)
     extra field (variable size)
*/
struct LocalFileHeader<'buf> {
    buf: &'buf [u8],
}

impl<'buf> LocalFileHeader<'buf> {
    const HEAD_MAGIC: &'static [u8] = &[0x50, 0x4b, 0x03, 0x04];
    const MIN_SIZE: usize = 28;
    const VERSION_NEEDED_TO_EXTRACT_POS: usize = 4;
    const GENERAL_PURPOSE_BIT_FLAG_POS: usize = 6;
    const COMPRESSION_METHOD_POS: usize = 8;
    const LAST_MOD_FILE_TIME_POS: usize = 10;
    const LAST_MOD_FILE_DATA_POS: usize = 12;
    const CRC_32_POS: usize = 14;
    const COMPRESSED_SIZE_POS: usize = 18;
    const UNCOMPRESSED_SIZE_POS: usize = 22;
    const FILE_NAME_LENGTH_POS: usize = 26;
    const EXTRA_FIELD_LENGTH_POS: usize = 28;
    const FILE_NAME_POS: usize = 30;

    #[inline]
    fn new(buf: &'buf [u8]) -> Result<Self, Error> {
        let res = Self { buf };
        let len = buf.len();
        if len < Self::MIN_SIZE {
            return Err(Error::ZipFileParseError(format!(
                "Local file header is too short `{}`",
                len
            )));
        }
        if &buf[..4] != Self::HEAD_MAGIC {
            return Err(Error::ZipFileParseError(format!(
                "Invalid head magic for local head magic: `{:?}`",
                &buf[..4]
            )));
        }

        if len < res.size() {
            return Err(Error::ZipFileParseError(format!(
                "Expect local file header size at least `{}`, but got `{}`",
                res.size(),
                len
            )));
        }
        Ok(res)
    }

    #[inline(always)]
    fn version_needed_to_extract(&self) -> u16 {
        read_le_u16!(self.buf, Self::VERSION_NEEDED_TO_EXTRACT_POS)
    }

    #[inline(always)]
    fn general_purpose_bit_flag(&self) -> u16 {
        read_le_u16!(self.buf, Self::GENERAL_PURPOSE_BIT_FLAG_POS)
    }

    #[inline(always)]
    fn compression_method(&self) -> u16 {
        read_le_u16!(self.buf, Self::COMPRESSION_METHOD_POS)
    }

    #[inline(always)]
    fn last_mod_file_time(&self) -> u16 {
        read_le_u16!(self.buf, Self::LAST_MOD_FILE_TIME_POS)
    }

    #[inline(always)]
    fn last_mod_file_data(&self) -> u16 {
        read_le_u16!(self.buf, Self::LAST_MOD_FILE_DATA_POS)
    }

    #[inline(always)]
    fn crc_32(&self) -> u32 {
        read_le_u32!(self.buf, Self::CRC_32_POS)
    }

    #[inline(always)]
    fn compressed_size(&self) -> u32 {
        read_le_u32!(self.buf, Self::COMPRESSED_SIZE_POS)
    }

    #[inline(always)]
    fn uncompressed_size(&self) -> u32 {
        read_le_u32!(self.buf, Self::UNCOMPRESSED_SIZE_POS)
    }

    #[inline(always)]
    fn file_name_length(&self) -> u16 {
        read_le_u16!(self.buf, Self::FILE_NAME_LENGTH_POS)
    }

    #[inline(always)]
    fn extra_field_length(&self) -> u16 {
        read_le_u16!(self.buf, Self::EXTRA_FIELD_LENGTH_POS)
    }

    #[inline(always)]
    fn file_name(&self) -> &'buf [u8] {
        let len = self.file_name_length() as usize;
        &self.buf[Self::FILE_NAME_POS..Self::FILE_NAME_POS + len]
    }

    #[inline(always)]
    fn extra_field(&self) -> &'buf [u8] {
        let start = Self::FILE_NAME_POS + self.file_name_length() as usize;
        let end = start + self.extra_field_length() as usize;
        &self.buf[start..end]
    }

    #[inline(always)]
    fn size(&self) -> usize {
        Self::FILE_NAME_POS + self.file_name_length() as usize + self.extra_field_length() as usize
    }
}

pub(crate) struct ZipFiles<'buf> {
    buf: &'buf [u8],
    end_of_central_directory_record: EndOfCentralDirectoryRecord<'buf>,
    files: HashMap<Cow<'buf, str>, std::ops::Range<usize>>,
}

impl<'buf> ZipFiles<'buf> {
    #[inline]
    pub fn try_new(buf: &'buf [u8]) -> Result<Option<Self>, Error> {
        let start_pos = EndOfCentralDirectoryRecord::try_find_start_pos(buf);
        if start_pos == buf.len() {
            return Ok(None);
        }
        Self::new_with_start_pos(buf, start_pos).map(|z| Some(z))
    }

    #[inline]
    pub fn new(buf: &'buf [u8]) -> Result<Self, Error> {
        let start_pos = EndOfCentralDirectoryRecord::try_find_start_pos(buf);
        if start_pos == buf.len() {
            return Err(Error::ZipFileParseError(
                "Cannot find end of central directory.".into(),
            ));
        }
        Self::new_with_start_pos(buf, start_pos)
    }

    #[inline]
    fn new_with_start_pos(buf: &'buf [u8], start_pos: usize) -> Result<Self, Error> {
        let r = EndOfCentralDirectoryRecord::new_with_start_pos(buf, start_pos)?;
        let num = r.total_number_of_central_directory_records();
        let mut start = r.offset_of_start_of_central_directory() as usize;
        let end = r.size_of_central_directory() as usize + start;

        let mut files = HashMap::new();
        for _ in 0..num {
            let c = CentralDirectory::new(&buf[start..end])?;
            let filename = String::from_utf8_lossy(c.file_name());
            let local_file_head_offset = c.relative_offset_of_local_header() as usize;
            let local_file_head = LocalFileHeader::new(&buf[local_file_head_offset..])?;
            let file_offset = local_file_head_offset + local_file_head.size();
            files.insert(
                filename,
                file_offset..(file_offset + c.uncompressed_size() as usize),
            );

            start += c.size();
        }
        Ok(Self {
            buf,
            end_of_central_directory_record: r,
            files,
        })
    }

    #[inline(always)]
    pub fn get_file(&self, name: &str) -> Option<&'buf [u8]> {
        self.files.get(name).map(|r| &self.buf[r.clone()])
    }

    #[inline(always)]
    pub fn get_file_offset(&self, name: &str) -> Option<std::ops::Range<usize>> {
        self.files.get(name).cloned()
    }

    /// Copy all file contents to HashMap<filename, file contents>
    #[inline]
    pub fn copy_contents(&self) -> HashMap<String, Vec<u8>> {
        self.files
            .iter()
            .map(|(name, range)| (name.to_string(), self.buf[range.clone()].to_vec()))
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::model::zip::{
        CentralDirectory, EndOfCentralDirectoryRecord, LocalFileHeader, ZipFiles,
    };

    const ZIP_PATH: &'static str = "assets/testdata/test.zip";
    const MODEL_PATH: &'static str =
        "assets/models/image_classification/efficientnet_lite0_fp32.tflite";

    #[test]
    fn test_end_of_central_directory_record() {
        assert!(EndOfCentralDirectoryRecord::new(&[]).is_err());

        let buf = std::fs::read(ZIP_PATH).unwrap();
        let r = EndOfCentralDirectoryRecord::new(buf.as_slice()).unwrap();
        assert_eq!(r.number_of_this_disk(), 0);
        assert_eq!(r.disk_where_central_directory_starts(), 0);
        assert_eq!(r.number_of_central_directory_records_on_this_disk(), 2);
        assert_eq!(r.total_number_of_central_directory_records(), 2);
        assert_eq!(r.size_of_central_directory(), 150);
        assert_eq!(r.offset_of_start_of_central_directory(), 130);
        assert_eq!(r.comment_length(), 0);
        assert_eq!(r.comment(), []);
    }

    #[test]
    fn test_central_directory_record() {
        let buf = std::fs::read(ZIP_PATH).unwrap();
        let r = EndOfCentralDirectoryRecord::new(buf.as_slice()).unwrap();
        let num = r.total_number_of_central_directory_records();
        let mut start = r.offset_of_start_of_central_directory() as usize;
        let end = r.size_of_central_directory() as usize + start;

        let c = CentralDirectory::new(&buf[start..end]).unwrap();
        let filename = String::from_utf8_lossy(c.file_name());
        assert_eq!(filename, "1.txt");
        assert_eq!(c.uncompressed_size(), 2);
        let offset = c.relative_offset_of_local_header() as usize;
        let local_file_header = LocalFileHeader::new(&buf[offset..]).unwrap();
        let file_offset = offset + local_file_header.size();
        let file_content = &buf[file_offset..file_offset + c.uncompressed_size() as usize];
        assert_eq!(String::from_utf8_lossy(file_content), "1\n");
        start += c.size();

        let c = CentralDirectory::new(&buf[start..end]).unwrap();
        let filename = String::from_utf8_lossy(c.file_name());
        assert_eq!(filename, "2.txt");
        assert_eq!(c.uncompressed_size(), 2);
        let offset = c.relative_offset_of_local_header() as usize;
        let local_file_header = LocalFileHeader::new(&buf[offset..]).unwrap();
        let file_offset = offset + local_file_header.size();
        let file_content = &buf[file_offset..file_offset + c.uncompressed_size() as usize];
        assert_eq!(String::from_utf8_lossy(file_content), "2\n");
    }

    #[test]
    fn test_zip_file() {
        let buf = std::fs::read(ZIP_PATH).unwrap();
        let zip_file = ZipFiles::new(buf.as_slice()).unwrap();
        assert_eq!(zip_file.files.len(), 2);
        assert_eq!(zip_file.get_file("1.txt").unwrap(), &[49, 10]);
        assert_eq!(zip_file.get_file("2.txt").unwrap(), &[50, 10]);
    }
}
