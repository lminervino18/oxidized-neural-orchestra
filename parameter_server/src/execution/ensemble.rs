use std::sync::Arc;

use tokio::sync::Barrier;

use crate::{
    execution::executor::Executor,
    parameters::{ParameterEngine, ParameterStore, optimization::Optimizer},
};

/// An executor that synchronizes parameter updates across multiple workers using a barrier.
#[derive(Clone)]
pub struct EnsembleExec<O: Optimizer> {
    engine: ParameterEngine<O>,
    barrier: Arc<Barrier>,
}

impl<O: Optimizer + Send> EnsembleExec<O> {
    /// Creates a new `EnsembleExec`.
    ///
    /// # Arguments
    /// * `store` - The parameter store to use as engine producer.
    /// * `barrier_size` - The size of the barrier for accumulating workers before and after weight update.
    pub fn new(store: ParameterStore<O>, barrier_size: usize) -> Self {
        Self {
            engine: ParameterEngine::new(store.handle()),
            barrier: Arc::new(Barrier::new(barrier_size)),
        }
    }
}

impl<O: Optimizer + Send> Executor for EnsembleExec<O> {
    /// The synchronous implementation of a training step.
    async fn step(&self, grad: &[f32], weights: &mut [f32]) {
        self.engine.accumulate(grad).await;

        if self.barrier.wait().await.is_leader() {
            self.engine.update_weights().await;
        }

        self.barrier.wait().await;
        self.engine.pull_weights(weights).await;
    }
}
