use super::Optimizer;
use crate::arch::Model;

pub struct GradientDescent<M: Model> {
    model: M,
    learning_rate: f32,
}

impl<M: Model> GradientDescent<M> {
    pub fn new(model: M, learning_rate: f32) -> Self {
        Self {
            model,
            learning_rate,
        }
    }
}

impl<M: Model> Optimizer for GradientDescent<M> {
    fn calculate_gradient() {}

    fn update_weights() {
        todo!()
    }
}
