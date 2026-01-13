use std::sync::Arc;

use tokio::sync::Barrier;

use crate::{optimization::Optimizer, storage::ParameterHandle, training::Trainer};

/// A trainer that synchronizes parameter updates across multiple workers using a barrier.
#[derive(Clone)]
pub struct BarrierSyncTrainer {
    barrier: Arc<Barrier>,
}

impl BarrierSyncTrainer {
    /// Creates a new `BarrierSyncTrainer`.
    ///
    /// # Arguments
    /// * `barrier_size` - The amount of workers to wait on until updating the weights of the model.
    pub fn new(barrier_size: usize) -> Self {
        Self {
            barrier: Arc::new(Barrier::new(barrier_size)),
        }
    }
}

impl Trainer for BarrierSyncTrainer {
    async fn step<O>(&self, handle: &ParameterHandle<O>, grad: &[f32], weights: &mut [f32])
    where
        O: Optimizer + Send,
    {
        handle.accumulate(grad).await;

        if self.barrier.wait().await.is_leader() {
            handle.update_weights().await;
        }

        self.barrier.wait().await;
        handle.pull_weights(weights).await;
    }
}
