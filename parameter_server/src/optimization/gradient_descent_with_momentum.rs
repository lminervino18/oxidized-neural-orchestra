use super::Optimizer;
use crate::storage::{Result, SizeMismatchErr};

#[derive(Debug)]
pub struct GradientDescentWithMomentum {
    learning_rate: f32,
    momentum: f32,
    velocity: Box<[f32]>,
}

impl GradientDescentWithMomentum {
    /// Creates a new `GradientDescentWithMomentum` optimizer.
    ///
    /// # Arguments
    /// * `len` - The amount of parameters this instance should hold.
    /// * `learning_rate` - The small coefficient that modulates the amount of training per update.
    /// * `momentum` - Hyperparameter to the optimization algorithm.
    ///
    /// # Returns
    /// A new `GradientDescentWithMomentum` instance.
    pub fn new(len: usize, learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: vec![0.; len].into_boxed_slice(),
        }
    }
}

impl Optimizer for GradientDescentWithMomentum {
    fn update_params(&mut self, grad: &[f32], params: &mut [f32]) -> Result<()> {
        if grad.len() != params.len() {
            return Err(SizeMismatchErr);
        }

        let lr = self.learning_rate;
        let mu = self.momentum;

        params
            .iter_mut()
            .zip(grad)
            .zip(self.velocity.iter_mut())
            .for_each(|((p, g), v)| {
                *v = (mu * *v) + g;
                *p -= lr * *v;
            });

        Ok(())
    }
}
