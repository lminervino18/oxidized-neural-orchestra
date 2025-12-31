/// Executes a single parameter update step.
///
/// An `Trainer` coordinates the application of a gradient and produces
/// updated model weights, potentially involving asynchronous operations.
#[trait_variant::make(Trainer: Send)]
pub trait TrainerTemplate: Clone {
    /// Should implement a step in the traning process, meaning accumulating this gradient and updating weights.
    ///
    /// # Arguments
    /// * `grad` - The incoming gradient to accumulate.
    /// * `weights` - Where the next state of the weights will be written to.
    async fn step(&self, grad: &[f32], weights: &mut [f32]);
}
