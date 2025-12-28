use std::sync::Arc;

use tokio::{sync::Barrier, task::JoinSet};

use crate::parameters::{Optimizer, ParameterHandle, ParameterStore};

/// A Bulk Synchrnous Parallel executor for distributed parameter training.
///
/// `BulkSync` coordinates multiple worker tasks that operate in lock-step. It manages
/// a shared `Barrier` and a `ParameterStore`, ensuring that all workers synchronize
/// their gradients before the global model weights are updated.
pub struct BulkSync<O: Optimizer> {
    store: ParameterStore<O>,
    futs: JoinSet<()>,
    barrier: Arc<Barrier>,
}

impl<O: Optimizer> BulkSync<O> {
    /// Creates a new `BulkSync` executor.
    ///
    /// # Arguments
    /// * `store` - The underlying parameter store to be managed.
    /// * `workers` - The exact number of participants the `Barrier` expects.
    pub fn new(store: ParameterStore<O>, workers: usize) -> Self {
        Self {
            store,
            futs: JoinSet::new(),
            barrier: Arc::new(Barrier::new(workers)),
        }
    }

    /// Spawns a training task into the executor's runtime.
    ///
    /// # Arguments
    /// * `train_fn` - A closure that manages the server side training loop.
    pub fn spawn<F, Fut>(&mut self, train_fn: F)
    where
        F: FnOnce(ParameterHandle<O>, Arc<Barrier>) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.futs
            .spawn(train_fn(self.store.handle(), self.barrier.clone()));
    }
}

impl<O: Optimizer + Send> BulkSync<O> {
    /// Gracefully waits for all spawned workers to complete and performs a final weight update.
    ///
    /// This should be called at the end of traning to ensure that any straggling gradients are
    /// applied to the final model state in the `store`.
    pub async fn join_all(&mut self) {
        while let Some(..) = self.futs.join_next().await {}
        self.store.update_weights();
    }
}
