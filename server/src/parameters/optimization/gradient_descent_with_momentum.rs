use super::Optimizer;

#[derive(Debug)]
pub struct GradientDescentWithMomentum {
    learning_rate: f32,
    momentum: f32,
    velocity: Box<[f32]>,
}

impl GradientDescentWithMomentum {
    pub fn new(len: usize, learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: vec![0.; len].into_boxed_slice(),
        }
    }
}

impl Optimizer for GradientDescentWithMomentum {
    fn update_weights(&mut self, weights: &mut [f32], grad: &[f32]) {
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
