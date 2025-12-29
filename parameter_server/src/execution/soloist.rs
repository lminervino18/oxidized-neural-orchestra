use crate::{
    execution::executor::Executor,
    parameters::{ParameterEngine, ParameterStore, optimization::Optimizer},
};

/// An executor that doesn't synchronize it's operations, will process incoming gradients as a "soloist".
pub struct SoloistExec<O: Optimizer> {
    engine: ParameterEngine<O>,
}

impl<O: Optimizer> Clone for SoloistExec<O> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
        }
    }
}

impl<O: Optimizer> SoloistExec<O> {
    /// Creates a new `SoloistExec`.
    ///
    /// # Arguments
    /// * `store` - The parameter store to use as engine producer.
    pub fn new(store: ParameterStore<O>) -> Self {
        Self {
            engine: ParameterEngine::new(store.handle()),
        }
    }
}

impl<O: Optimizer + Send> Executor for SoloistExec<O> {
    /// The asynchronous implementation of a traning step.
    async fn step(&self, grad: &[f32], weights: &mut [f32]) {
        self.engine.accumulate(grad).await;
        self.engine.update_weights().await;
        self.engine.pull_weights(weights).await;
    }
}
