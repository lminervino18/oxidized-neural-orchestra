use tokio::task;

use crate::parameters::{ParameterStore, optimization::Optimizer};

/// Drives parameter operations in the async runtime.
///
/// Bridges async execution and CPU-bound parameter updates.
pub struct ParameterHandle<O: Optimizer>(ParameterStore<O>);

impl<O: Optimizer> Clone for ParameterHandle<O> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<O: Optimizer> ParameterHandle<O> {
    /// Creates a new `ParameterHandle`
    ///
    /// # Arguments
    /// * `store` - The underlying parameter store.
    pub fn new(store: ParameterStore<O>) -> Self {
        Self(store)
    }
}

impl<O: Optimizer + Send> ParameterHandle<O> {
    pub async fn accumulate(&self, grad: &[f32]) {
        task::block_in_place(|| self.0.accumulate(grad));
    }

    pub async fn update_weights(&self) {
        task::block_in_place(|| self.0.update_weights());
    }

    pub async fn pull_weights(&self, out: &mut [f32]) {
        task::block_in_place(|| self.0.pull_weights(out));
    }
}
