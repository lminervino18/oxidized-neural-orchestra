use super::Optimizer;
use crate::storage::{Result, SizeMismatchErr};

#[derive(Debug)]
pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    /// Creates a new `GradientDescent` optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - The small coefficient that modulates the amount of training per update.
    ///
    /// # Returns
    /// A new `GradientDescent` instance.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn update_params(&mut self, grad: &[f32], params: &mut [f32]) -> Result<()> {
        if grad.len() != params.len() {
            return Err(SizeMismatchErr);
        }

        let lr = self.learning_rate;

        for (p, g) in params.iter_mut().zip(grad) {
            *p -= lr * g;
        }

        Ok(())
    }
}
