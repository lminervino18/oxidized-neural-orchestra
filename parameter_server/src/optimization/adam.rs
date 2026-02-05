use super::Optimizer;
use crate::storage::{Result, SizeMismatchErr};

#[derive(Debug)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    beta1_t: f32,
    beta2_t: f32,
    v: Box<[f32]>,
    s: Box<[f32]>,
    epsilon: f32,
}

impl Adam {
    /// Creates a new `Adam` optimizer.
    ///
    /// # Arguments
    /// * `len` - The amount of parameters this instance should hold.
    /// * `learning_rate` - The small coefficient that modulates the amount of training per update.
    /// * `beta1`, `beta2`, `epsilon` - Hyperparameters to the optimization algorithm.
    ///
    /// # Returns
    /// A new `Adam` instance.
    pub fn new(len: usize, learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            beta1_t: 1.,
            beta2_t: 1.,
            v: vec![0.; len].into_boxed_slice(),
            s: vec![0.; len].into_boxed_slice(),
            epsilon,
        }
    }
}

impl Optimizer for Adam {
    fn update_params(&mut self, grad: &[f32], params: &mut [f32]) -> Result<()> {
        if grad.len() != params.len() {
            return Err(SizeMismatchErr);
        }

        let Self {
            learning_rate: lr,
            beta1: b1,
            beta2: b2,
            epsilon: eps,
            ..
        } = *self;

        self.beta1_t *= b1;
        self.beta2_t *= b2;

        let bc1 = 1. - self.beta1_t;
        let bc2 = 1. - self.beta2_t;
        let step_size = lr * (bc2.sqrt() / bc1);

        params
            .iter_mut()
            .zip(grad)
            .zip(self.v.iter_mut())
            .zip(self.s.iter_mut())
            .for_each(|(((p, g), v), s)| {
                *v = b1 * *v + (1. - b1) * g;
                *s = b2 * *s + (1. - b2) * g.powi(2);
                *p -= step_size * *v / (s.sqrt() + eps);
            });

        Ok(())
    }
}
