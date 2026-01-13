use crate::{optimization::Optimizer, storage::ParameterHandle, training::Trainer};

/// A trainer that doesn't synchronize it's operations, will process incoming gradients immediately.
#[derive(Clone)]
pub struct NonBlockingTrainer;

impl NonBlockingTrainer {
    /// Creates a new `NonBlockingTrainer`.
    pub fn new() -> Self {
        Self {}
    }
}

impl Trainer for NonBlockingTrainer {
    async fn step<O>(&self, handle: &ParameterHandle<O>, grad: &[f32], weights: &mut [f32])
    where
        O: Optimizer + Send,
    {
        handle.accumulate(grad).await;
        handle.update_weights().await;
        handle.pull_weights(weights).await;
    }
}
