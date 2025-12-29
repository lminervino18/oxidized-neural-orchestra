use tokio::task::JoinSet;

use crate::parameters::{ParameterHandle, ParameterStore, optimization::Optimizer};

/// An Asynchronous Parallel executor for high-throughput parameter training.
///
/// `WaitFree` implements an execution model where workers operate independently. There is not global
/// synchronization; workers pull the current weights, compute gradients, and push updates as fast as
/// their local resources allow.
pub struct WaitFree<O: Optimizer> {
    store: ParameterStore<O>,
    futs: JoinSet<()>,
}

impl<O: Optimizer> WaitFree<O> {
    /// Creates a new `WaitFree` executor.
    ///
    /// # Arguments
    /// * `store` - The underlying parameter store to be managed.
    pub fn new(store: ParameterStore<O>) -> Self {
        Self {
            store,
            futs: JoinSet::new(),
        }
    }

    /// Spawns a training task into the executor's runtime.
    pub fn spawn<F, Fut>(&mut self, train_fn: F)
    where
        F: FnOnce(ParameterHandle<O>) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.futs.spawn(train_fn(self.store.handle()));
    }
}

impl<O: Optimizer + Send> WaitFree<O> {
    /// Gracefully waits for all spawned workers to complete and performs a final weight update.
    ///
    /// This should be called at the end of traning to ensure that any straggling gradients are
    /// applied to the final model state in the `store`.
    pub async fn join_all(&mut self) {
        while self.futs.join_next().await.is_some() {}
        self.store.update_weights();
    }
}
