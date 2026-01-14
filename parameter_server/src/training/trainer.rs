use crate::{
    optimization::Optimizer,
    storage::{ParameterHandle, Result},
};

/// Executes a single parameter update step.
///
/// A `Trainer` coordinates the application of gradients and produces updated model weights.
#[trait_variant::make(Trainer: Send)]
pub trait TrainerTemplate: Clone {
    /// Should implement a step in the traning process, meaning accumulating this gradient and updating the weights.
    ///
    /// # Arguments
    /// * `handle` - The parameter handle holding the weights of the model.
    /// * `grad` - The incoming gradient to accumulate.
    /// * `weights` - Where to write the resultant weights.
    ///
    /// # Returns
    /// An error if there's a size mismatch between `grad`, `weights` and the size of the storage.
    async fn step<O>(
        &self,
        handle: &ParameterHandle<O>,
        grad: &[f32],
        weights: &mut [f32],
    ) -> Result<()>
    where
        O: Optimizer + Send;
}
