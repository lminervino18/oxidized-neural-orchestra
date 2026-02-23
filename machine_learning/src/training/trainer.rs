use crate::middleware::ParamManager;

/// This trait acts as an interface for the `Worker` to use `machine_learning` related components.
pub trait Trainer {
    fn train(&mut self, param_manager: &mut ParamManager<'_>) -> Vec<f32>;
}
