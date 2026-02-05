use super::Synchronizer;
use crate::{
    optimization::Optimizer,
    storage::{ParameterHandle, Result},
};

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
    async fn step<O>(
        &self,
        handle: &ParameterHandle<O>,
        grad: &[f32],
        params: &mut [f32],
    ) -> Result<()>
    where
        O: Optimizer + Send,
    {
        handle.accumulate(grad).await?;
        handle.update_params().await;
        handle.pull_params(params).await?;
        Ok(())
    }
}
