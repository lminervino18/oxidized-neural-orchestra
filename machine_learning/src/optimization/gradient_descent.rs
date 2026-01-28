use super::Optimizer;

pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn update_params(&mut self, params: &mut [f32], grad: &[f32]) {
        let lr = self.learning_rate;

        for (w, g) in params.iter_mut().zip(grad) {
            *w -= lr * g;
        }
    }
}
