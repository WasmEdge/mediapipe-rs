macro_rules! results_iter_impl {
    () => {
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
