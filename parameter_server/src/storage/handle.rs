use tokio::task;

use crate::{optimization::Optimizer, storage::ParameterStore};

/// The actual interface to interact with a `ParameterStore`.
///
/// It bridges the async world with the blocking CPU-bound implementation of the `ParameterStore`.
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
    /// Async call to the CPU-bounded implementation of `ParameterStore::accumulate`.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    ///
    /// # Panics
    /// If the length of `grad` doesn't match the total number of parameters.
    pub async fn accumulate(&self, grad: &[f32]) {
        task::block_in_place(|| self.0.accumulate(grad));
    }

    /// Async call to the CPU-bounded implementation of `ParameterStore::update_weights`.
    pub async fn update_weights(&self) {
        task::block_in_place(|| self.0.update_weights());
    }

    /// Async call to the CPU-bounded implementatino of `ParameterStore::pull_weights`.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the weights will be copied.
    ///
    /// # Panics
    /// If the length of `out` doesn't match the total number of parameters.
    pub async fn pull_weights(&self, out: &mut [f32]) {
        task::block_in_place(|| self.0.pull_weights(out));
    }
}
