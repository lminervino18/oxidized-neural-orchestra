use super::Optimizer;

#[derive(Debug)]
pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn update_weights(&mut self, weights: &mut [f32], grad: &[f32]) {
        let lr = self.learning_rate;

        for (w, g) in weights.iter_mut().zip(grad) {
            *w -= lr * g;
        }
    }
}
