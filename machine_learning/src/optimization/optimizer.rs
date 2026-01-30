/// This trait represents the *learning rule* that is to be taken when updating the parameters of a model.
pub trait Optimizer {
    fn update_params(&mut self, params: &mut [f32], grad: &[f32]);
}
