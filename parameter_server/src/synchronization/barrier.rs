use std::{num::NonZeroUsize, sync::Arc};

use tokio::task;

use super::{DynBarrier, Synchronizer};
use crate::storage::{Result, Store};

/// Synchronizes parameter updates across multiple workers by waiting for every worker
/// to collaborate on the current gradient aggregation using a barrier.
#[derive(Clone)]
pub struct BarrierSync {
    barrier: Arc<DynBarrier>,
}

impl BarrierSync {
    /// Creates a new `BarrierSync` synchronizer.
    ///
    /// # Args
    /// * `size` - The number of workers expected to synchronize each epoch.
    ///
    /// # Returns
    /// A new `BarrierSync` instance.
    pub fn new(size: NonZeroUsize) -> Self {
        Self {
            barrier: Arc::new(DynBarrier::new(size)),
        }
    }
}

impl Drop for BarrierSync {
    fn drop(&mut self) {
        self.barrier.acquire();
    }
}

impl Synchronizer for BarrierSync {
    async fn step<PS>(&self, store: &PS, grad: &[f32], params: &mut [f32]) -> Result<()>
    where
        PS: Store + Send + Sync,
    {
        task::block_in_place(|| {
            store.accumulate(grad)?;
            self.barrier.wait_with(|| store.update_params());
            store.pull_params(params)
        })
    }
}
