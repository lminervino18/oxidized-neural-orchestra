use tokio::task;

use crate::parameters::{ParameterHandle, optimization::Optimizer};

/// Drives parameter operations in the async runtime.
///
/// Bridges async execution and CPU-bound parameter updates.
#[derive(Clone)]
pub struct ParameterEngine<O: Optimizer>(ParameterHandle<O>);

impl<O: Optimizer> ParameterEngine<O> {
    /// Creates a new `ParameterEngine`
    ///
    /// # Arguments
    /// * `handle` - The underlying parameter handle.
    pub fn new(handle: ParameterHandle<O>) -> Self {
        Self(handle)
    }
}

impl<O: Optimizer + Send> ParameterEngine<O> {
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
