use crate::{
    parameters::{ParameterHandle, ParameterStore, optimization::Optimizer},
    training::Trainer,
};

/// A trainer that doesn't synchronize it's operations, will process incoming gradients immediately.
pub struct NonBlocking<O: Optimizer> {
    handle: ParameterHandle<O>,
}

impl<O: Optimizer> Clone for NonBlocking<O> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

impl<O: Optimizer> NonBlocking<O> {
    /// Creates a new `NonBlocking`.
    ///
    /// # Arguments
    /// * `store` - A trainable set of parameters.
    pub fn new(store: ParameterStore<O>) -> Self {
        Self {
            handle: ParameterHandle::new(store),
        }
    }
}

impl<O: Optimizer + Send> Trainer for NonBlocking<O> {
    /// The asynchronous implementation of a traning step.
    async fn step(&self, grad: &[f32], weights: &mut [f32]) {
        let Self { handle: engine } = self;

        engine.accumulate(grad).await;
        engine.update_weights().await;
        engine.pull_weights(weights).await;
    }
}
