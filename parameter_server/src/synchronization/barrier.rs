use std::sync::Arc;

use tokio::sync::Barrier;

use super::Synchronizer;
use crate::{
    optimization::Optimizer,
    storage::{Result, StoreHandle},
};

/// Synchronizes parameter updates across multiple workers using a barrier.
#[derive(Clone)]
pub struct BarrierSync {
    barrier: Arc<Barrier>,
}

impl BarrierSync {
    /// Creates a new `BarrierSync` synchronizer.
    ///
    /// # Arguments
    /// * `barrier_size` - The amount of workers to wait on until updating the parameters of the model.
    ///
    /// # Returns
    /// A new `BarrierSync` instance.
    pub fn new(barrier_size: usize) -> Self {
        Self {
            barrier: Arc::new(Barrier::new(barrier_size)),
        }
    }
}

impl Synchronizer for BarrierSync {
    async fn step<O>(&self, handle: &StoreHandle<O>, grad: &[f32], params: &mut [f32]) -> Result<()>
    where
        O: Optimizer + Send,
    {
        handle.accumulate(grad).await?;

        if self.barrier.wait().await.is_leader() {
            handle.update_params().await;
        }

        self.barrier.wait().await;
        handle.pull_params(params).await?;
        Ok(())
    }
}
