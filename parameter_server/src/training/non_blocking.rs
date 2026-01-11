use crate::{
    optimization::Optimizer,
    storage::{ParameterHandle, ParameterStore},
    training::Trainer,
};

/// A trainer that doesn't synchronize it's operations, will process incoming gradients immediately.
pub struct NonBlockingTrainer<O: Optimizer> {
    handle: ParameterHandle<O>,
}

impl<O: Optimizer> Clone for NonBlockingTrainer<O> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

impl<O: Optimizer> NonBlockingTrainer<O> {
    /// Creates a new `NonBlockingTrainer`.
    ///
    /// # Arguments
    /// * `store` - The underlying parameter store.
    pub fn new(store: ParameterStore<O>) -> Self {
        Self {
            handle: ParameterHandle::new(store),
        }
    }
}

impl<O: Optimizer + Send> Trainer for NonBlockingTrainer<O> {
    async fn pull_weights(&self, weights: &mut [f32]) {
        self.handle.pull_weights(weights).await;
    }

    async fn step(&self, grad: &[f32], weights: &mut [f32]) {
        let Self { handle } = self;

        handle.accumulate(grad).await;
        handle.update_weights().await;
        handle.pull_weights(weights).await;
    }
}
