use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, Ordering},
};

use rayon::prelude::*;

use super::{Optimizer, ParameterShard};

/// Provides the primary interface for workers to contribute gradients and update weights.
#[derive(Debug)]
pub struct ParameterHandle<O: Optimizer> {
    active_idx: Arc<AtomicU8>,
    updating: Arc<AtomicBool>,
    shards: Arc<[ParameterShard<O>]>,
    shard_size: usize,
    params: usize,
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
            let frozen_idx = self.active_idx.fetch_xor(1, Ordering::SeqCst) as usize;

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
