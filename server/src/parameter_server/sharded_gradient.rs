use std::num::NonZeroUsize;

use parking_lot::Mutex;

/// A sharded buffer for accumulating gradients across multiple threads.
///
/// It divides a large parameter vector into smaller chunks (shards), each protected
/// by its own `Mutex`. This reduces lock contention when multiple workers attempt to
/// accumulate gradients simultaneously.
#[derive(Debug)]
pub struct ShardedGradient {
    pub shards: Box<[Mutex<Box<[f32]>>]>,
    pub shard_size: usize,
}

impl ShardedGradient {
    /// Creates a new `ShardedGradient` table.
    ///
    /// # Arguments
    /// * `params` - Total number of parameters in the model.
    /// * `shards_amount` - The amount of shards to partition the gradient.
    pub fn new(params: usize, shards_amount: NonZeroUsize) -> Self {
        let n = shards_amount.get();
        let shard_size = (params + n - 1) / n;

        let shards = (0..n)
            .map(|i| {
                let start = i * shard_size;
                let end = (start + shard_size).min(params);
                Mutex::new(vec![0.; end - start].into_boxed_slice())
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { shards, shard_size }
    }
}
