/// This trait represents the *learning rule* that is to be taken when updating the parameters of a model.
pub trait Optimizer: Send {
    /// Updates the provided parameters using the provided gradient according to the optimizer's
    /// learning rule.
    ///
    /// # Arguments
    /// * `params` - The parameters to be updated
    /// * `grad` - The gradient used to update the parameters
    fn update_params(&mut self, params: &mut [f32], grad: &[f32]);
}
