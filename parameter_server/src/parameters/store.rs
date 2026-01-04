use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

use rayon::prelude::*;

use super::{ParameterShard, optimization::Optimizer};

/// Provides the primary interface for workers to contribute gradients and update weights.
#[derive(Debug)]
pub struct ParameterStore<O: Optimizer> {
    active_idx: Arc<AtomicU8>,
    updating: Arc<AtomicBool>,
    shards: Arc<[ParameterShard<O>]>,
    shard_size: usize,
    params: usize,
}

impl<O: Optimizer> Clone for ParameterStore<O> {
    fn clone(&self) -> Self {
        Self {
            active_idx: Arc::clone(&self.active_idx),
            updating: Arc::clone(&self.updating),
            shards: Arc::clone(&self.shards),
            shard_size: self.shard_size,
            params: self.params,
        }
    }
}

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

        Self {
            active_idx: Arc::new(AtomicU8::new(0)),
            updating: Arc::new(AtomicBool::new(false)),
            shards: Arc::from(shards),
            shard_size,
            params,
        }
    }
}

impl<O: Optimizer + Send> ParameterStore<O> {
    /// Accumulates a new gradient into the active gradient buffer.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    pub(super) fn accumulate(&self, grad: &[f32]) {
        let active_idx = self.active_idx.load(Ordering::Acquire) as usize;

        self.shards
            .par_iter()
            .zip(grad.par_chunks(self.shard_size))
            .for_each(|(shard, grad_slice)| {
                shard.accumulate(active_idx, grad_slice);
            });
    }

    /// Swaps the active gradient buffer and applies the frozen gradient to the weights.
    ///
    /// This triggers a parallel update across all shards.
    pub(super) fn update_weights(&self) {
        let success = self
            .updating
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok();

        if success {
            let frozen_idx = self.active_idx.fetch_xor(1, Ordering::AcqRel) as usize;

            self.shards
                .par_iter()
                .for_each(|shard| shard.update_weights(frozen_idx));

            self.updating.store(false, Ordering::Release);
        }
    }

    /// Gathers all the sharded weights into a local buffer.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the weights will be copied.
    ///
    /// # Panics
    /// If the length of `out` doesn't match the total number of parameters.
    pub(super) fn pull_weights(&self, out: &mut [f32]) {
        self.shards
            .par_iter()
            .zip(out.par_chunks_mut(self.shard_size))
            .for_each(|(shard, out_slice)| {
                shard.pull_weights(out_slice);
            });
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    struct AddOptimizer;

    impl Optimizer for AddOptimizer {
        fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]) {
            weights.iter_mut().zip(grad).for_each(|(w, g)| *w += g);
        }
    }

    fn create_test_store(params: usize, shard_amount: usize) -> ParameterStore<AddOptimizer> {
        ParameterStore::new(params, NonZeroUsize::new(shard_amount).unwrap(), |_| {
            AddOptimizer
        })
    }

    #[test]
    fn test_handle_ragged_shards() {
        const PARAMS: usize = 15;
        const SHARD_AMOUNT: usize = 2;

        let store = create_test_store(PARAMS, SHARD_AMOUNT);
        let grad = [1.0; PARAMS];

        store.accumulate(&grad);
        store.update_weights();

        let mut out = [0.0; PARAMS];
        store.pull_weights(&mut out);
        assert_eq!(out, [1.0; PARAMS]);
    }

    #[test]
    fn test_handle_buffer_swap() {
        const PARAMS: usize = 10;
        const SHARD_AMOUNT: usize = 10;

        let store = create_test_store(PARAMS, SHARD_AMOUNT);
        store.accumulate(&[1.0; PARAMS]);

        store.update_weights();
        assert_eq!(store.active_idx.load(Ordering::Acquire), 1);
        store.accumulate(&[5.0; PARAMS]);

        let mut weights = [0.0; PARAMS];
        store.pull_weights(&mut weights);
        assert_eq!(weights, [1.0; PARAMS]);

        store.update_weights();
        store.pull_weights(&mut weights);
        assert_eq!(weights, [6.0; PARAMS]);
    }

    #[test]
    fn test_update_locking_mechanism() {
        const PARAMS: usize = 10;
        const SHARD_AMOUNT: usize = 10;

        let store = create_test_store(PARAMS, SHARD_AMOUNT);
        store.updating.store(true, Ordering::SeqCst);

        let active_idx = store.active_idx.load(Ordering::Acquire);
        store.update_weights();
        assert_eq!(store.active_idx.load(Ordering::Acquire), active_idx);

        store.updating.store(false, Ordering::Release);
        store.update_weights();
        assert_ne!(store.active_idx.load(Ordering::SeqCst), active_idx);
    }

    #[test]
    fn test_store_initialization_and_flow() {
        const PARAMS: usize = 100;
        const SHARD_AMOUNT: NonZeroUsize = NonZeroUsize::new(4).unwrap();

        let store = ParameterStore::new(PARAMS, SHARD_AMOUNT, |_| AddOptimizer);

        let grad = [1.0; PARAMS];
        store.accumulate(&grad);
        store.update_weights();

        let mut weights = [0.0; PARAMS];
        store.pull_weights(&mut weights);

        for (i, &w) in weights.iter().enumerate() {
            assert_eq!(w, 1.0, "Weight mismatch at index {i}");
        }
    }

    #[test]
    fn test_ragged_edge_distribution() {
        const PARAMS: usize = 105;
        const SHARD_AMOUNT: NonZeroUsize = NonZeroUsize::new(10).unwrap();

        let store = ParameterStore::new(PARAMS, SHARD_AMOUNT, |_| AddOptimizer);

        let grad = [1.0; PARAMS];
        store.accumulate(&grad);
        store.update_weights();

        let mut weights = [0.0; PARAMS];
        store.pull_weights(&mut weights);
        assert_eq!(weights.len(), PARAMS);
    }
}
