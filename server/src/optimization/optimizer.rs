pub trait Optimizer {
    fn update_weights(&mut self, weights: &mut [f32], gradient: &[f32]);
}
