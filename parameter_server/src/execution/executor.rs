/// Executes a single parameter update step.
///
/// An `Executor` coordinates the application of a gradient and produces
/// updated model weights, potentially involving asynchronous operations.
pub trait Executor {
    /// Should implement a step in the traning process, meaning accumulating this gradient and updating weights.
    ///
    /// # Arguments
    /// * `grad` - The incoming gradient to accumulate.
    /// * `weights` - Where to write the weights using the engine's `pull_weights`.
    async fn step(&self, grad: &[f32], weights: &mut [f32]);
}
