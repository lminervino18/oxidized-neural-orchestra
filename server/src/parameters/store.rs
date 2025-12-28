use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8},
    },
};

use crate::parameters::{Optimizer, ParameterHandle, ParameterShard};

/// A `ParameterHandle` factory.
///
/// It's responsible for the initial allocation of the weights, gradients and optimizers. It partitions the model into uniform shards.
///
/// After initialization will serve as a factory of `ParameterHandle`s.
#[derive(Debug)]
pub struct ParameterStore<O: Optimizer>(ParameterHandle<O>);

impl<O: Optimizer> ParameterStore<O> {
    /// Creates a new `ParameterStore`.
    ///
    /// # Arguments
    /// * `params` - Total number of parameters in the model.
    /// * `shard_amount` - The amount of shards to partition the model.
    /// * `optimizer` - The optimization algorithm.
    pub fn new(params: usize, shard_amount: NonZeroUsize, optimizer: O) -> Self {
        let n = shard_amount.get();
        let shard_size = (params + n - 1) / n;

        let shards: Vec<_> = (0..n)
            .map(|i| {
                let start = i * shard_size;
                let end = (start + shard_size).min(params);
                ParameterShard::new(end - start, optimizer.clone())
            })
            .collect();

        let handle = ParameterHandle::new(
            Arc::new(AtomicU8::new(0)),
            Arc::new(AtomicBool::new(false)),
            Arc::from(shards),
            shard_size,
            params,
        );

        Self(handle)
    }

    /// Creates a new `ParameterHandle` bound to this store's data.
    ///
    /// # Returns
    /// A new `ParameterHandle`.
    pub fn handle(&self) -> ParameterHandle<O> {
        self.0.clone()
    }
}
