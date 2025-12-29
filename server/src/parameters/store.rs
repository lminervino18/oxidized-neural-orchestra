use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8},
    },
};

use super::{ParameterHandle, ParameterShard, optimization::Optimizer};

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
    /// * `factory` - An `Optimizer` factory closure.
    pub fn new<F>(params: usize, shard_amount: NonZeroUsize, mut factory: F) -> Self
    where
        F: FnMut(usize) -> O,
    {
        let n = shard_amount.get();
        let shard_size = params.div_ceil(n);

        let shards: Vec<_> = (0..n)
            .map(|i| {
                let start = i * shard_size;
                let end = (start + shard_size).min(params);
                let len = end - start;
                ParameterShard::new(len, factory(len))
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

impl<O: Optimizer + Send> ParameterStore<O> {
    pub fn update_weights(&self) {
        self.0.update_weights();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddOptimizer;

    impl Optimizer for AddOptimizer {
        fn update_weights(&mut self, weights: &mut [f32], grad: &[f32]) {
            weights.iter_mut().zip(grad).for_each(|(w, g)| *w += g);
        }
    }

    #[test]
    fn test_store_initialization_and_flow() {
        const PARAMS: usize = 100;
        const SHARD_AMOUNT: NonZeroUsize = NonZeroUsize::new(4).unwrap();

        let store = ParameterStore::new(PARAMS, SHARD_AMOUNT, |_| AddOptimizer);
        let handle = store.handle();

        let grad = [1.0; PARAMS];
        handle.accumulate(&grad);
        store.update_weights();

        let mut weights = [0.0; PARAMS];
        handle.pull_weights(&mut weights);

        for (i, &w) in weights.iter().enumerate() {
            assert_eq!(w, 1.0, "Weight mismatch at index {i}");
        }
    }

    #[test]
    fn test_ragged_edge_distribution() {
        const PARAMS: usize = 105;
        const SHARD_AMOUNT: NonZeroUsize = NonZeroUsize::new(10).unwrap();

        let store = ParameterStore::new(PARAMS, SHARD_AMOUNT, |_| AddOptimizer);
        let handle = store.handle();

        let grad = [1.0; PARAMS];
        handle.accumulate(&grad);
        store.update_weights();

        let mut weights = [0.0; PARAMS];
        handle.pull_weights(&mut weights);
        assert_eq!(weights.len(), PARAMS);
    }
}
