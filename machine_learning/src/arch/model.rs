use crate::{arch::loss::LossFn, error::Result, optimization::Optimizer, training::ParamManager};
use ndarray::ArrayView2;

pub trait Model {
    /// Returns the amount of parameters in the model.
    fn size(&self) -> usize;

    /// Computes the gradient of the loss function with respect to the parameters of the model over
    /// the provided batches. **`params` gets updated** for each batch according to the
    /// optimization algorithm.
    ///
    /// # Arguments
    /// * `params` - The model's parameters.
    /// * `grad` - A buffer for writing the computed gradient on each batch pass.
    /// * `loss` - The loss function.
    /// * `optimizer` - The optimizer that dictates how to update the weights on each gradient calculation.
    /// * `batches` - The batches of data.
    ///
    /// # Returns
    /// The epoch loss.
    fn backprop<'a, 'b, L, O, I>(
        &mut self,
        params: &mut ParamManager<'b>,
        grad: &mut [f32],
        loss: &L,
        optimizer: &mut O,
        batches: I,
    ) -> Result<f32>
    where
        L: LossFn,
        O: Optimizer,
        I: Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>;
}
