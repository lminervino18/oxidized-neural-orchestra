use super::Optimizer;

/// Gradient descent optimization algorithm.
pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    /// Returns a new `GradientDescent`.
    ///
    /// # Arguments
    /// * `learning_rate` - The *length* of the steps taken on `update_params`.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    /// Updates the parameters according to the algorithm's learning rule, that is, making a step in
    /// the opposite direction of the gradient, with a length of `learning_rate`.
    ///
    /// # Arguments
    /// * `params` - The parameters that are going to be modified.
    /// * `grad` - The gradient used for taking the step.
    fn update_params(&mut self, params: &mut [f32], grad: &[f32]) {
        let lr = self.learning_rate;

        for (w, g) in params.iter_mut().zip(grad) {
            *w -= lr * g;
        }
    }
}
