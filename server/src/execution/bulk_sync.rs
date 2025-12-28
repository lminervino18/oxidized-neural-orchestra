use std::sync::Arc;

use tokio::{sync::Barrier, task::JoinSet};

use crate::parameters::{Optimizer, ParameterHandle, ParameterStore};

pub struct BulkSync<O: Optimizer> {
    store: ParameterStore<O>,
    futs: JoinSet<()>,
    barrier: Arc<Barrier>,
}

impl<O: Optimizer> BulkSync<O> {
    pub fn new(store: ParameterStore<O>, workers: usize) -> Self {
        Self {
            store,
            futs: JoinSet::new(),
            barrier: Arc::new(Barrier::new(workers)),
        }
    }

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
    pub async fn join_all(&mut self) {
        while let Some(..) = self.futs.join_next().await {}
        self.store.update_weights();
    }
}
