use crate::{arch::loss::LossFn, optimization::Optimizer};
use ndarray::ArrayView2;

pub trait Model {
    // esto puede ser distinto para cada modelo concreto...
    fn size(&self) -> usize;

    /// Computes the gradient of the loss function with respect to the parameters of the model over
    /// the provided batches. **`params` gets updated** for each batch according to the
    /// optimization algorithm.
    ///
    /// # Arguments
    /// * `params` - The model's parameters.
    /// * `grad` - A buffer for writting the computed gradient on each batch pass.
    /// * `loss` - The loss function.
    /// * `optimizer` - The optimizer that dictates how to update the weights on each gradient
    /// calculation.
    /// * `batches` - The batches of data.
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
