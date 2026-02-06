use super::Synchronizer;
use crate::storage::{Result, Store, StoreHandle};

/// Skips synchronization between workers for it's operations, will process incoming gradients immediately.
#[derive(Clone)]
pub struct NoBlockingSync;

impl NoBlockingSync {
    /// Creates a new `NonBlockingSync` synchronizer.
    ///
    /// # Returns
    /// A new `NonBlockingSync` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl Synchronizer for NoBlockingSync {
    async fn step<S>(&self, handle: &StoreHandle<S>, grad: &[f32], params: &mut [f32]) -> Result<()>
    where
        S: Store + Send + Sync,
    {
        handle.accumulate(grad).await?;
        handle.update_params().await;
        handle.pull_params(params).await?;
        Ok(())
    }
}
