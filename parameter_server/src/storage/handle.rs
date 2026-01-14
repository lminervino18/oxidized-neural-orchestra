use std::ops::Deref;

use tokio::task;

use super::{ParameterStore, Result};
use crate::optimization::Optimizer;

/// The actual interface to interact with a `ParameterStore`.
///
/// It bridges the async world with the blocking CPU-bound implementation of the `ParameterStore`.
pub struct ParameterHandle<O: Optimizer>(ParameterStore<O>);

impl<O: Optimizer> Clone for ParameterHandle<O> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<O: Optimizer> Deref for ParameterHandle<O> {
    type Target = ParameterStore<O>;

    fn deref(&self) -> &Self::Target {
        &self.0
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
    /// # Returns
    /// A `SizeMismatchErr` if `grad` isn't the same size as this shard.
    pub async fn accumulate(&self, grad: &[f32]) -> Result<()> {
        task::block_in_place(|| self.0.accumulate(grad))
    }

    /// Async call to the CPU-bounded implementation of `ParameterStore::update_weights`.
    pub async fn update_weights(&self) {
        task::block_in_place(|| self.0.update_weights());
    }

    /// Async call to the CPU-bounded implementation of `ParameterStore::pull_weights`.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the weights will be copied.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if there is a size mismatch in any of the inner shards.
    pub async fn pull_weights(&self, out: &mut [f32]) -> Result<()> {
        task::block_in_place(|| self.0.pull_weights(out))
    }
}
