use std::num::NonZeroUsize;

use crate::MlError;

/// A pure computational model.
///
/// A `Model` defines how to evaluate a function and accumulate parameter gradients.
/// It does not:
/// - own parameters,
/// - access datasets,
/// - implement training loops.
pub trait Model: Send + Sync {
    /// Input type consumed by the model.
    type Input: Send + Sync;

    /// Output type produced by the model.
    type Output: Send + Sync;

    /// Error signal passed to the backward pass.
    ///
    /// For supervised learning, this is often dL/dy, but the exact meaning is
    /// defined by the training strategy.
    type ErrorSignal: Send + Sync;

    /// Returns the number of scalar parameters expected in `weights` and `grads`.
    fn num_params(&self) -> NonZeroUsize;

    /// Computes the model output for a given input.
    ///
    /// # Errors
    /// Returns `MlError` if invariants are violated (e.g., shape mismatch).
    fn forward(&self, weights: &[f32], input: &Self::Input) -> Result<Self::Output, MlError>;

    /// Accumulates gradients into `grads` given an error signal.
    ///
    /// Implementations must add to `grads` rather than overwrite it, enabling
    /// microbatching and gradient accumulation in training strategies.
    ///
    /// # Errors
    /// Returns `MlError` if invariants are violated.
    fn backward(
        &self,
        weights: &[f32],
        input: &Self::Input,
        error: &Self::ErrorSignal,
        grads: &mut [f32],
    ) -> Result<(), MlError>;
}
