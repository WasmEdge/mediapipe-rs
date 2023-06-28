pub trait Softmax {
    fn softmax_inplace(&mut self);
}

impl Softmax for Vec<f32> {
    #[inline(always)]
    fn softmax_inplace(&mut self) {
        self.as_mut_slice().softmax_inplace()
    }
}

impl Softmax for [f32] {
    #[inline(always)]
    fn softmax_inplace(&mut self) {
        let mut sum = 0f32;
        for i in self.iter_mut() {
            *i = i.exp();
            sum += *i;
        }

        for i in self.iter_mut() {
            *i = *i / sum;
        }
    }
}
