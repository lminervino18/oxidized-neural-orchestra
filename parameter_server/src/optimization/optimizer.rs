/// Defines the strategy for updating model parameters based on calculated gradients.
pub trait Optimizer {
    /// Updates the provided slice of weights using the accumulated gradients.
    ///
    /// # Arguments
    /// * `grad` - A reference to the model's gradient.
    /// * `weights` - The weights to update.
    fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]);
}
