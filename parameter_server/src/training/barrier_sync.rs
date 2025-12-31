use std::sync::Arc;

use tokio::sync::Barrier;

use crate::{
    parameters::{ParameterHandle, ParameterStore, optimization::Optimizer},
    training::Trainer,
};

/// A trainer that synchronizes parameter updates across multiple workers using a barrier.
pub struct BarrierSync<O: Optimizer> {
    engine: ParameterHandle<O>,
    barrier: Arc<Barrier>,
}

impl<O: Optimizer> Clone for BarrierSync<O> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            barrier: Arc::clone(&self.barrier),
        }
    }
}

impl<O: Optimizer> BarrierSync<O> {
    /// Creates a new `BarrierSync`.
    ///
    /// # Arguments
    /// * `store` - A trainable set of parameters.
    /// * `barrier_size` - The size of the barrier for accumulating workers before and after weight update.
    pub fn new(store: ParameterStore<O>, barrier_size: usize) -> Self {
        Self {
            engine: ParameterHandle::new(store),
            barrier: Arc::new(Barrier::new(barrier_size)),
        }
    }
}

impl<O: Optimizer + Send> Trainer for BarrierSync<O> {
    /// The synchronous implementation of a training step.
    async fn step(&self, grad: &[f32], weights: &mut [f32]) {
        let Self { engine, barrier } = self;

        engine.accumulate(grad).await;

        if barrier.wait().await.is_leader() {
            engine.update_weights().await;
        }

        barrier.wait().await;
        engine.pull_weights(weights).await;
    }
}
