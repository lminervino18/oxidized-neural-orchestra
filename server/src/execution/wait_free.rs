use tokio::task::JoinSet;

use crate::parameters::{Optimizer, ParameterHandle, ParameterStore};

pub struct WaitFree<O: Optimizer> {
    store: ParameterStore<O>,
    futs: JoinSet<()>,
}

impl<O: Optimizer> WaitFree<O> {
    pub fn new(store: ParameterStore<O>) -> Self {
        Self {
            store,
            futs: JoinSet::new(),
        }
    }

    pub fn spawn<F, Fut>(&mut self, train_fn: F)
    where
        F: FnOnce(ParameterHandle<O>) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.futs.spawn(train_fn(self.store.handle()));
    }
}

impl<O: Optimizer + Send> WaitFree<O> {
    pub async fn join_all(&mut self) {
        while let Some(..) = self.futs.join_next().await {}
        self.store.update_weights();
    }
}
