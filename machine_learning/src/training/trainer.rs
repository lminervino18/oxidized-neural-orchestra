use crate::{Result, param_manager::ParamManager};

/// The result of a training call.
///
/// Either the training must go on or the last call was the last.
pub struct TrainResult<'trainer> {
    pub losses: &'trainer [f32],
    pub was_last: bool,
}

/// This trait generalizes all the different concrete `ModelTrainer` variations between optimizers, loss functions, ...
pub trait Trainer {
    /// Performs a single training 'cycle'.
    ///
    /// This cycle could involve one or more epochs.
    ///
    /// # Arguments
    /// * `param_manager` - The manager of parameters for this training cycle.
    ///
    /// # Returns
    /// A training result declaring if the trianing has finished or should continue.
    fn train(&mut self, param_manager: &mut ParamManager<'_>) -> Result<TrainResult<'_>>;
}
