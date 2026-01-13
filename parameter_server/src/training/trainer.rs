use crate::{optimization::Optimizer, storage::ParameterHandle};

/// Executes a single parameter update step.
///
/// A `Trainer` coordinates the application of gradients and produces updated model weights.
#[trait_variant::make(Trainer: Send)]
pub trait TrainerTemplate: Clone {
    /// Should implement a step in the traning process, meaning accumulating this gradient and updating the weights.
    ///
    /// # Arguments
    /// * `grad` - The incoming gradient to accumulate.
    /// * `weights` - Where to write the resultant weights.
    async fn step<O>(&self, handle: &ParameterHandle<O>, grad: &[f32], weights: &mut [f32])
    where
        O: Optimizer + Send;
}
