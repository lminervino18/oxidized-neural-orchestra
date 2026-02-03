use comms::specs::worker::ModelSpec;
use machine_learning::{MlError, StepStats};

/// Worker-local optimization interface.
pub trait Optimizer {
    /// Computes gradients for the given weights.
    ///
    /// # Args
    /// * `weights` - Current model parameters.
    /// * `grads` - Output buffer for gradients.
    ///
    /// # Returns
    /// Step-level statistics on success.
    ///
    /// # Errors
    /// Returns `MlError` when inputs violate shape invariants or are invalid.
    fn gradient(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError>;

    /// Applies a local update to the weights.
    ///
    /// # Args
    /// * `weights` - Mutable model parameters to update.
    /// * `grads` - Gradients to apply.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// Returns `MlError` when inputs violate shape invariants or are invalid.
    fn update_weights(&mut self, weights: &mut [f32], grads: &[f32]) -> Result<(), MlError>;
}

/// Worker-local optimizer implementation selected from `ModelSpec`.
pub enum OptimizerImpl {
    Noop,
    Mock,
    Unsupported,
}

impl OptimizerImpl {
    /// Builds an optimizer implementation from `ModelSpec`.
    ///
    /// # Args
    /// * `model` - Model specification.
    ///
    /// # Returns
    /// A concrete optimizer implementation.
    pub fn from_model_spec(model: &ModelSpec) -> Self {
        match model {
            ModelSpec::Noop => Self::Noop,
            ModelSpec::Mock => Self::Mock,
            ModelSpec::FeedForward { .. } => Self::Unsupported,
        }
    }
}

impl Optimizer for OptimizerImpl {
    fn gradient(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        match self {
            OptimizerImpl::Noop => Ok(StepStats::new(1, 0)),
            OptimizerImpl::Mock => {
                if weights.len() != grads.len() {
                    return Err(MlError::ShapeMismatch {
                        what: "params",
                        got: weights.len(),
                        expected: grads.len(),
                    });
                }

                for (g, w) in grads.iter_mut().zip(weights.iter()) {
                    *g = 2.0 * *w;
                }

                Ok(StepStats::new(1, 0))
            }
            OptimizerImpl::Unsupported => Err(MlError::InvalidInput("model is not supported yet")),
        }
    }

    fn update_weights(&mut self, _weights: &mut [f32], _grads: &[f32]) -> Result<(), MlError> {
        match self {
            OptimizerImpl::Unsupported => Err(MlError::InvalidInput("model is not supported yet")),
            _ => Ok(()),
        }
    }
}
