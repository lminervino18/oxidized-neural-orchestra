use crate::{
    optimization::Optimizer,
    storage::{ParameterHandle, Result},
};

/// Executes a single parameter update step.
///
/// A `Synchronizer` coordinates the application of gradients and produces updated model parameters.
#[allow(unused)]
#[trait_variant::make(Synchronizer: Send)]
pub trait SynchronizerTemplate: Clone {
    /// Should implement a step in the training process, meaning accumulating this gradient and updating the parameters.
    ///
    /// # Arguments
    /// * `handle` - The parameter handle holding the parameters of the model.
    /// * `grad` - The incoming gradient to accumulate.
    /// * `params` - Where to write the resultant parameters.
    ///
    /// # Returns
    /// An error if there's a size mismatch between `grad`, `params` and the size of the storage.
    async fn step<O>(
        &self,
        handle: &ParameterHandle<O>,
        grad: &[f32],
        params: &mut [f32],
    ) -> Result<()>
    where
        O: Optimizer + Send;
}
