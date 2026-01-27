use crate::arch::loss::LossFn;
use ndarray::ArrayView2;

pub trait Model {
    fn backprop<'a, L, I>(&mut self, params: &mut [f32], grad: &mut [f32], loss: &L, batches: I)
    where
        L: LossFn,
        I: IntoIterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>;
}
