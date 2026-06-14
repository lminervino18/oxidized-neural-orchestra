use tokio::task;

use super::Synchronizer;
use crate::storage::{Result, Store};

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
    async fn step<PS>(&self, handle: &PS, grad: &[f32], params: &mut [f32]) -> Result<()>
    where
        PS: Store + Send + Sync,
    {
        task::block_in_place(|| {
            handle.accumulate(grad)?;
            handle.update_params();
            handle.pull_params(params)?;
            Ok(())
        })
    }
}
