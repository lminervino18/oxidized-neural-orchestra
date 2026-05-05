use crate::storage::{Result, Store, StoreHandle};

/// Executes a single parameter update step.
///
/// A `Synchronizer` coordinates the application of gradients and produces updated model parameters.
#[allow(unused)]
#[trait_variant::make(Synchronizer: Send)]
pub trait SynchronizerTemplate: Clone {
    /// Accumulates `grad` and updates model parameters, writing the result into `params`.
    ///
    /// # Args
    /// * `handle` - The parameter store shared across all worker tasks on this server.
    /// * `grad` - The incoming gradient to accumulate for this step.
    /// * `params` - Buffer to write the updated parameters into after the step.
    ///
    /// # Returns
    /// An error if there is a size mismatch between `grad`, `params`, or the store.
    async fn step<S>(
        &self,
        handle: &StoreHandle<S>,
        grad: &[f32],
        params: &mut [f32],
    ) -> Result<()>
    where
        S: Store + Send + Sync;
}
