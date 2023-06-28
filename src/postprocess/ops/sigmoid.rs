pub(crate) trait Sigmoid {
    fn sigmoid_inplace(&mut self);
}

impl Sigmoid for Vec<f32> {
    #[inline(always)]
    fn sigmoid_inplace(&mut self) {
        self.as_mut_slice().sigmoid_inplace()
    }
}

impl Sigmoid for [f32] {
    #[inline(always)]
    fn sigmoid_inplace(&mut self) {
        self.iter_mut()
            .for_each(|z| *z = 1f32 / (1f32 + (-(*z)).exp()));
    }
}

impl Sigmoid for f32 {
    #[inline(always)]
    fn sigmoid_inplace(&mut self) {
        *self = 1f32 / (1f32 + (-(*self)).exp())
    }
}
