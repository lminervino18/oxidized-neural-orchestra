use crate::{Result, param_manager::ParamManager};

/// The result of a training call.
///
/// Either the training must go on or the last call was the last.
pub struct TrainResult<'trainer> {
    pub losses: &'trainer [f32],
    pub was_last: bool,
}

/// This trait generalizes all the different concrete `ModelTrainer` variations between optimizers, loss functions, ...
pub trait Trainer: Send {
    /// Performs a single training 'cycle'.
    ///
    /// This cycle could involve one or more epochs.
    ///
    /// # Args
    /// * `param_manager` - The manager of parameters for this training cycle.
    ///
    /// # Returns
    /// A training result declaring if the trianing has finished or should continue.
    fn train(&mut self, param_manager: &mut ParamManager<'_>) -> Result<TrainResult<'_>>;

    /// Optimizes the parameters in the param manager.
    ///
    /// # Args
    /// * `param_manager` - The manager of parameters for this optimization.
    ///
    /// # Returns
    /// An error if there's a mismatch in the sizes of the grad and param buffers.
    fn optimize<'mw>(&mut self, param_manager: &mut ParamManager<'mw>) -> Result<()>;
}
