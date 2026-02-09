/// This trait acts as an interface for the `Worker` to use `machine_learning` related components.
pub trait Trainer {
    fn train(&mut self, params: &mut [f32]) -> (&[f32], f32);
}
