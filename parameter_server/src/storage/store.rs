use super::Result;

/// Defines the strategy to handle the model's parameters, either block when reading and
/// writing or embrace race conditions to benefit performance over training stability.
///
/// Parameter stores must implement `Clone`, this way the storage can be distributed between
/// workers easily just by cloning the instance.
pub trait Store: Clone {
    /// Returns the size of the storage.
    ///
    /// # Returns
    /// The amount of parameters in the storage.
    fn len(&self) -> usize;

    /// Accumulates a new gradient into the storage.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if the length of `grad` and the size of the storage mismatch.
    fn accumulate(&self, grad: &[f32]) -> Result<()>;

    /// Applies the accumulated gradients into the storage's parameters.
    fn update_params(&self);

    /// Writes the parameters' values into the given output buffer.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the parameters will be copied.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if the length of `out` and the size of the storage mismatch.
    fn pull_params(&self, out: &mut [f32]) -> Result<()>;
}
