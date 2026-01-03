/// Defines the strategy for updating model parameters based on calculated gradients.
///
/// The `Optimizer` trait is responsible for the mathematical transition of weights from state `t` to `t+1`.
pub trait Optimizer {
    /// Updates the provided slice of weights using the accumulated gradients.
    ///
    /// # Arguments
    /// * `grad` - The accumulated gradients corresponding to the `weights` slice.
    /// * `weights` - A mutable slice of the current parameter values for a specific shard.
    fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]);
}
