/// Executes a single parameter update step.
///
/// A `Trainer` coordinates the application of gradients and produces updated model weights.
#[trait_variant::make(Trainer: Send)]
pub trait TrainerTemplate: Clone {
    /// Should read the model's weights and write them into `weights`.
    ///
    /// # Arguments
    /// * `weights` - Where the inner weights should be written to.
    async fn pull_weights(&self, weights: &mut [f32]);

    /// Should implement a step in the traning process, meaning accumulating this gradient and updating the weights.
    ///
    /// # Arguments
    /// * `grad` - The incoming gradient to accumulate.
    /// * `weights` - Where to write the resultant weights.
    async fn step(&self, grad: &[f32], weights: &mut [f32]);
}
