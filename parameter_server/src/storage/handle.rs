use std::ops::Deref;

use tokio::task;

use super::{Result, Store};

/// The actual interface to interact with a `Store`.
///
/// It bridges the async runtime with the blocking CPU-bound implementation of the `Store`.
pub struct StoreHandle<S: Store>(S);

impl<S: Store> Clone for StoreHandle<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<S: Store> Deref for StoreHandle<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Store> StoreHandle<S> {
    /// Creates a new `StoreHandle`
    ///
    /// # Arguments
    /// * `store` - The underlying parameter store.
    ///
    /// # Returns
    /// A new `StoreHandle` instance.
    pub fn new(store: S) -> Self {
        Self(store)
    }
}

impl<S: Store> StoreHandle<S> {
    /// Async call to the synchronous implementation of `Store::accumulate`.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if `grad` isn't the same size as this shard.
    pub async fn accumulate(&self, grad: &[f32]) -> Result<()> {
        task::block_in_place(|| self.0.accumulate(grad))
    }

    /// Async call to the synchronous implementation of `Store::update_params`.
    pub async fn update_params(&self) {
        task::block_in_place(|| self.0.update_params());
    }

    /// Async call to the synchronous implementation of `Store::pull_params`.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the parameters will be copied.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if there is a size mismatch in any of the inner shards.
    pub async fn pull_params(&self, out: &mut [f32]) -> Result<()> {
        task::block_in_place(|| self.0.pull_params(out))
    }
}
