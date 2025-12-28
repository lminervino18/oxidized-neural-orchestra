pub trait Optimizer: Clone {
    fn update_weights(&mut self, weights: &mut [f32], grad: &[f32]);
}
