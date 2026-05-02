use comms::Float01;

use super::Optimizer;
use crate::{MlErr, Result};

/// Gradient descent optimization algorithm with momentum.
pub struct GradientDescentWithMomentum {
    learning_rate: f32,
    momentum: Float01,
    velocity: Box<[f32]>,
}

impl GradientDescentWithMomentum {
    /// Creates a new `GradientDescentWithMomentum` optimizer.
    ///
    /// # Args
    /// * `len` - The amount of parameters this instance should hold.
    /// * `learning_rate` - The small coefficient that modulates the amount of training per update.
    /// * `momentum` - Hyperparameter to the optimization algorithm.
    ///
    /// # Returns
    /// A new `GradientDescentWithMomentum` instance.
    pub fn new(len: usize, learning_rate: f32, momentum: Float01) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: vec![0.; len].into_boxed_slice(),
        }
    }
}

impl Optimizer for GradientDescentWithMomentum {
    /// Updates the parameters according to the algorithm's learning rule, that is, making a step in
    /// the opposite direction of the gradient, with a length of `learning_rate` taking into account
    /// previous movements using the momentum.
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
        let mu = self.momentum;

        params
            .iter_mut()
            .zip(grad)
            .zip(self.velocity.iter_mut())
            .for_each(|((p, g), v)| {
                *v = (*mu * *v) + g;
                *p -= lr * *v;
            });

        Ok(())
    }
}
