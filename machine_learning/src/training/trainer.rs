use super::ParamManager;
use crate::error::Result;

/// This trait acts as an interface for the `Worker` to use `machine_learning` related components.
pub trait Trainer {
    fn train<'a>(&'a mut self, params: &mut ParamManager<'a>) -> Result<(&'a [f32], Vec<f32>)>;
}
