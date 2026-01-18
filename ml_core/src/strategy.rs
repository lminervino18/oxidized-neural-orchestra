use crate::{MlError, StepStats};

/// Abstraction over a local training computation executed by a worker.
///
/// Implementations encapsulate all model-, data-, and loss-specific logic.
/// The worker treats this trait as a black box that maps weights to gradients.
///
/// This trait represents the *training policy boundary*: it is the only
/// interface the infrastructure requires to run distributed training.
/// The specific composition of datasets, models, losses, batching policies,
/// and optimizers is intentionally outside of the worker and lives behind
/// implementations of this trait.
pub trait TrainStrategy: Send {
    /// Executes one local training step.
    ///
    /// # Args
    /// * `weights` - Read-only slice containing the current model parameters.
    /// * `grads` - Mutable gradient buffer provided by the worker.
    ///
    /// # Returns
    /// On success, returns step-level statistics (`StepStats`). On failure, returns an `MlError`.
    ///
    /// # Errors
    /// Implementations should return:
    /// - `MlError::ShapeMismatch` when shapes/lengths do not match expectations.
    /// - `MlError::InvalidInput` for invalid domain inputs.
    ///
    /// # Panics
    /// Implementations should not panic; they should report failures via `MlError`.
    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError>;
}
