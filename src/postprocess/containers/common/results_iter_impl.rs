macro_rules! results_iter_impl {
    () => {
        /// poll all results
        #[inline(always)]
        pub fn collect<B: FromIterator<TaskSession::Result>>(self) -> B
        where
            Self: Sized,
        {
            todo!()
        }

        /// poll all results and save to [`Vec`]
        #[inline(always)]
        pub fn to_vec(mut self) -> Result<Vec<TaskSession::Result>, crate::Error> {
            let mut ans = Vec::new();
            while let Some(r) = self.next()? {
                ans.push(r);
            }
            Ok(ans)
        }
    };
}
