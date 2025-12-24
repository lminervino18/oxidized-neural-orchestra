use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use rayon::prelude::*;

use crate::parameter_server::ShardedGradient;

/// A handle used by workers to push new gradients to the parameter server.
#[derive(Debug, Clone)]
pub struct PSClient {
    params: usize,
    active_idx: Arc<AtomicU8>,
    tables: [Arc<ShardedGradient>; 2],
}

impl PSClient {
    /// Creates a new `PSClient`.
    ///
    /// # Arguments
    /// * `params` - Total number of parameters in the model.
    /// * `active_idx` - The atomic active table index.
    /// * `tables` - Both sharded gradient tables.
    pub fn new(
        params: usize,
        active_idx: Arc<AtomicU8>,
        tables: [Arc<ShardedGradient>; 2],
    ) -> Self {
        Self {
            params,
            active_idx,
            tables,
        }
    }

    /// Accumulates a local gradient into the server's active sharded table.
    ///
    /// # Arguments
    /// * `grad` - A slice into a new gradient.
    ///
    /// # Returns
    /// `Result<(), usize>` - Fails if there is a mismatch between `params` and `grad`'s length.
    pub fn accumulate(&self, grad: &[f32]) -> Result<(), usize> {
        if grad.len() != self.params {
            return Err(self.params);
        }

        let active_idx = self.active_idx.load(Ordering::Acquire) as usize;
        let &ShardedGradient {
            ref shards,
            shard_size,
        } = self.tables[active_idx].as_ref();

        grad.par_chunks(shard_size)
            .zip(shards)
            .for_each(|(grad_chunk, locked_shard)| {
                let mut nums = locked_shard.lock();
                for (acc, g) in nums.iter_mut().zip(grad_chunk) {
                    *acc += g;
                }
            });

        Ok(())
    }
}
