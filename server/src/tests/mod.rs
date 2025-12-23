mod parameter_server;

use crate::optimization::Optimizer;

pub struct TestOptimizer {}

impl Optimizer for TestOptimizer {
    fn update_weights(&mut self, weights: &mut [f32], gradient: &[f32]) {
        for (w, g) in weights.iter_mut().zip(gradient) {
            *w = *g;
        }
    }
}
