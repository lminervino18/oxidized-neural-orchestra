use super::Optimizer;

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
    pub fn new(len: usize, learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: vec![0.; len].into_boxed_slice(),
        }
    }
}

impl Optimizer for GradientDescentWithMomentum {
    fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]) {
        let lr = self.learning_rate;
        let mu = self.momentum;

        weights
            .iter_mut()
            .zip(grad)
            .zip(self.velocity.iter_mut())
            .for_each(|((w, g), v)| {
                *v = (mu * *v) + g;
                *w -= lr * *v;
            });
    }
}
