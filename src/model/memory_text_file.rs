use std::borrow::Cow;

pub(crate) struct MemoryTextFile<'buf> {
    cur: &'buf [u8],
}

impl<'buf> MemoryTextFile<'buf> {
    const NEW_LINE: u8 = 0x0A as u8;
    const WHITE_SPACE: u8 = ' ' as u8;

    #[inline(always)]
    pub(crate) fn new(buf: &'buf [u8]) -> Self {
        Self { cur: buf }
    }

    #[inline(always)]
    pub(crate) fn next_line(&mut self) -> Option<Cow<'buf, str>> {
        if self.cur.len() == 0 {
            return None;
        }

        match self.cur.iter().position(|c| *c == Self::NEW_LINE) {
            Some(p) => {
                let s = String::from_utf8_lossy(&self.cur[..p]);
                self.cur = if p + 1 < self.cur.len() {
                    &self.cur[p + 1..]
                } else {
                    &[]
                };
                Some(s)
            }
            None => {
                let s = String::from_utf8_lossy(self.cur);
                self.cur = &[];
                Some(s)
            }
        }
    }

    // assert all chars are ascii after ` ` in this line
    // if no white space, return the first string
    #[inline(always)]
    pub(crate) fn next_line_with_split_white_space(
        &mut self,
    ) -> (Option<Cow<'buf, str>>, Option<Cow<'buf, str>>) {
        if self.cur.len() == 0 {
            return (None, None);
        }

        let slice = match self.cur.iter().position(|c| *c == Self::NEW_LINE) {
            Some(p) => {
                let slice = &self.cur[..p];
                self.cur = if p + 1 < self.cur.len() {
                    &self.cur[p + 1..]
                } else {
                    &[]
                };
                slice
            }
            None => {
                let slice = self.cur;
                self.cur = &[];
                slice
            }
        };

        match slice.iter().rposition(|c| *c == Self::WHITE_SPACE) {
            None => (Some(String::from_utf8_lossy(slice)), None),
            Some(p) => (
                Some(String::from_utf8_lossy(&slice[..p])),
                Some(String::from_utf8_lossy(&slice[p + 1..])),
            ),
        }
    }
}
