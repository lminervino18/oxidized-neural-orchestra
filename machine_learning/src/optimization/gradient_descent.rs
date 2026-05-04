use comms::floats::FloatPositive;

use super::Optimizer;
use crate::{MlErr, Result};

/// Gradient descent optimization algorithm.
pub struct GradientDescent {
    learning_rate: FloatPositive,
}

impl GradientDescent {
    /// Creates a new `GradientDescent` optimizer.
    ///
    /// # Args
    /// * `learning_rate` - The *length* of the steps taken on `update_params`.
    ///
    /// # Returns
    /// A new `GradientDescent` instance.
    pub fn new(learning_rate: FloatPositive) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    /// Updates the parameters according to the algorithm's learning rule, that is, making a step in
    /// the opposite direction of the gradient, with a length of `learning_rate`.
    ///
    /// # Args
    /// * `grad` - The gradient used for taking the step.
    /// * `params` - The parameters that are going to be modified.
    ///
    /// # Returns
    /// A size mismatch error if the lengths of `grad` and `params` mismatch.
    fn update_params(&mut self, grad: &[f32], params: &mut [f32]) -> Result<()> {
        if grad.len() != params.len() {
            return Err(MlErr::SizeMismatch {
                what: "grad and params",
                got: grad.len(),
                expected: params.len(),
            });
        }

        let lr = self.learning_rate;

        for (w, g) in params.iter_mut().zip(grad) {
            *w -= *lr * g;
        }

        Ok(())
    }

    fn learning_rate(&self) -> FloatPositive {
        self.learning_rate
    }
}
