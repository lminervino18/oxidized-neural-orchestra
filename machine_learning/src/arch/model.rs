use crate::{arch::loss::LossFn, optimization::Optimizer};
use ndarray::ArrayView2;

pub trait Model {
    fn size(&self) -> usize;

    fn backprop<'a, L, O, I>(
        &mut self,
        params: &mut [f32],
        grad: &mut [f32],
        loss: &L,
        optimizer: &mut O,
        batches: I,
    ) where
        L: LossFn,
        O: Optimizer,
        I: Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>;
}
