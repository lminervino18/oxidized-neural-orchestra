use crate::Result;

/// Defines the strategy for updating model parameters based on calculated gradients.
pub trait Optimizer {
    /// Updates the provided slice of parameters using the accumulated gradients.
    ///
    /// # Arguments
    /// * `grad` - A reference to the model's gradient.
    /// * `params` - The parameters to update.
    ///
    /// # Returns
    /// An error if there's a mismatch in the sizes of `grad` and `params`.
    fn update_params(&mut self, grad: &[f32], params: &mut [f32]) -> Result<()>;
}
