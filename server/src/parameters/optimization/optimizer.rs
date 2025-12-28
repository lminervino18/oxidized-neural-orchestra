/// Defines the strategy for updating model parameters based on calculated gradients.
///
/// The `Optimizer` trait is responsible for themathematical transition of weights
/// from state `t` to `t+1`.
///
/// # Thread Safety and Performance
///
/// Since the `ParameterStore` shards the model, thismethod will be called concurrently across
/// different shards.
pub trait Optimizer {
    /// Updates the provided slice of weights using the accumulated gradients.
    ///
    /// # Arguments
    /// * `weights` - A mutable slice of the current parameter values for a specific shard.
    /// * `grad` - The accumulated gradients corresponding to the `weights` slice.
    ///
    /// # Panics
    /// Implementations should generally expect `weights.len() == grad.len()`.
    fn update_weights(&mut self, weights: &mut [f32], grad: &[f32]);
}
