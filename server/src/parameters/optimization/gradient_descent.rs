use super::Optimizer;

#[derive(Debug)]
pub struct GradientDescent {
    lr: f32,
}

impl Optimizer for GradientDescent {
    fn update_weights(&mut self, weights: &mut [f32], grad: &[f32]) {
        for (w, g) in weights.iter_mut().zip(grad) {
            *w -= self.lr * g;
        }
    }
}
