use std::sync::Arc;

use tokio::sync::Barrier;

use crate::{
    optimization::Optimizer,
    storage::{ParameterHandle, ParameterStore},
    training::Trainer,
};

/// A trainer that synchronizes parameter updates across multiple workers using a barrier.
pub struct BarrierSyncTrainer<O: Optimizer> {
    handle: ParameterHandle<O>,
    barrier: Arc<Barrier>,
}

impl<O: Optimizer> Clone for BarrierSyncTrainer<O> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            barrier: Arc::clone(&self.barrier),
        }
    }
}

impl<O: Optimizer> BarrierSyncTrainer<O> {
    /// Creates a new `BarrierSyncTrainer`.
    ///
    /// # Arguments
    /// * `barrier_size` - The amount of workers to wait on until updating the weights of the model.
    /// * `store` - The underlying parameter store.
    pub fn new(barrier_size: usize, store: ParameterStore<O>) -> Self {
        Self {
            handle: ParameterHandle::new(store),
            barrier: Arc::new(Barrier::new(barrier_size)),
        }
    }
}

impl<O: Optimizer + Send> Trainer for BarrierSyncTrainer<O> {
    async fn pull_weights(&self, weights: &mut [f32]) {
        self.handle.pull_weights(weights).await;
    }

    async fn step(&self, grad: &[f32], weights: &mut [f32]) {
        let Self { handle, barrier } = self;

        handle.accumulate(grad).await;

        if barrier.wait().await.is_leader() {
            handle.update_weights().await;
        }

        barrier.wait().await;
        handle.pull_weights(weights).await;
    }
}
