pub trait Optimizer {
    fn update_params(&mut self, params: &mut [f32], grad: &[f32]);
}
