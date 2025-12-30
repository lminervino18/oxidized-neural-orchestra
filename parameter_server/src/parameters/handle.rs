use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, Ordering},
};

use rayon::prelude::*;

use super::{ParameterShard, optimization::Optimizer};

/// Provides the primary interface for workers to contribute gradients and update weights.
#[derive(Debug)]
pub struct ParameterHandle<O: Optimizer> {
    active_idx: Arc<AtomicU8>,
    updating: Arc<AtomicBool>,
    shards: Arc<[ParameterShard<O>]>,
    shard_size: usize,
    params: usize,
}

impl<O: Optimizer> Clone for ParameterHandle<O> {
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

impl<O: Optimizer> ParameterHandle<O> {
    /// Creates a new `ParameterHandle`.
    ///
    /// Intended to be used only by `ParameterStore`.
    pub(super) fn new(
        active_idx: Arc<AtomicU8>,
        updating: Arc<AtomicBool>,
        shards: Arc<[ParameterShard<O>]>,
        shard_size: usize,
        params: usize,
    ) -> Self {
        Self {
            active_idx,
            updating,
            shards,
            shard_size,
            params,
        }
    }
}

impl<O: Optimizer + Send> ParameterHandle<O> {
    /// Accumulates a new gradient into the active gradient buffer.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    pub fn accumulate(&self, grad: &[f32]) {
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
    pub fn update_weights(&self) {
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
    pub fn pull_weights(&self, out: &mut [f32]) {
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

    use crate::parameters::ParameterStore;

    use super::*;

    struct AddOptimizer;

    impl Optimizer for AddOptimizer {
        fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]) {
            weights.iter_mut().zip(grad).for_each(|(w, g)| *w += g);
        }
    }

    fn create_test_handle(params: usize, shard_amount: usize) -> ParameterHandle<AddOptimizer> {
        ParameterStore::new(params, NonZeroUsize::new(shard_amount).unwrap(), |_| {
            AddOptimizer
        })
        .handle()
    }

    #[test]
    fn test_handle_ragged_shards() {
        const PARAMS: usize = 15;
        const SHARD_AMOUNT: usize = 2;

        let handle = create_test_handle(PARAMS, SHARD_AMOUNT);
        let grad = [1.0; PARAMS];

        handle.accumulate(&grad);
        handle.update_weights();

        let mut out = [0.0; PARAMS];
        handle.pull_weights(&mut out);
        assert_eq!(out, [1.0; PARAMS]);
    }

    #[test]
    fn test_handle_buffer_swap() {
        const PARAMS: usize = 10;
        const SHARD_AMOUNT: usize = 10;

        let handle = create_test_handle(PARAMS, SHARD_AMOUNT);
        handle.accumulate(&[1.0; PARAMS]);

        handle.update_weights();
        assert_eq!(handle.active_idx.load(Ordering::Acquire), 1);
        handle.accumulate(&[5.0; PARAMS]);

        let mut weights = [0.0; PARAMS];
        handle.pull_weights(&mut weights);
        assert_eq!(weights, [1.0; PARAMS]);

        handle.update_weights();
        handle.pull_weights(&mut weights);
        assert_eq!(weights, [6.0; PARAMS]);
    }

    #[test]
    fn test_update_locking_mechanism() {
        const PARAMS: usize = 10;
        const SHARD_AMOUNT: usize = 10;

        let handle = create_test_handle(PARAMS, SHARD_AMOUNT);
        handle.updating.store(true, Ordering::SeqCst);

        let active_idx = handle.active_idx.load(Ordering::Acquire);
        handle.update_weights();
        assert_eq!(handle.active_idx.load(Ordering::Acquire), active_idx);

        handle.updating.store(false, Ordering::Release);
        handle.update_weights();
        assert_ne!(handle.active_idx.load(Ordering::SeqCst), active_idx);
    }
}
