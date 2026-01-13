use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

use rayon::prelude::*;

use crate::{
    initialization::WeightGen,
    optimization::Optimizer,
    storage::{ParameterShard, SizeMismatchErr},
};

/// The primary storage of weights and accumulated gradients.
///
/// These methods are private to the module, they become available
/// through the async interface of a `ParameterHandle`.
#[derive(Debug)]
pub struct ParameterStore<O: Optimizer> {
    params: usize,
    active_idx: Arc<AtomicU8>,
    updating: Arc<AtomicBool>,
    shards: Arc<[ParameterShard<O>]>,
    shard_size: NonZeroUsize,
}

impl<O: Optimizer> Clone for ParameterStore<O> {
    fn clone(&self) -> Self {
        Self {
            params: self.params,
            active_idx: Arc::clone(&self.active_idx),
            updating: Arc::clone(&self.updating),
            shards: Arc::clone(&self.shards),
            shard_size: self.shard_size,
        }
    }
}

impl<O: Optimizer> ParameterStore<O> {
    /// Creates a new `ParameterStore`.
    ///
    /// # Arguments
    /// * `shard_size` - The maximum amount of parameters per shard.
    /// * `weight_gen` - A weight generator.
    /// * `optimizer_factory` - An `Optimizer` factory closure.
    pub fn new<W, F>(shard_size: NonZeroUsize, mut weight_gen: W, mut optimizer_factory: F) -> Self
    where
        W: WeightGen,
        F: FnMut(usize) -> O,
    {
        let mut params = 0;
        let mut shards = Vec::new();

        while let Some(weights) = weight_gen.sample(shard_size.get()) {
            params += weights.len();
            let optimizer = optimizer_factory(weights.len());
            let shard = ParameterShard::new(weights, optimizer);
            shards.push(shard);
        }

        Self {
            params,
            active_idx: Arc::new(AtomicU8::new(0)),
            updating: Arc::new(AtomicBool::new(false)),
            shards: Arc::from(shards),
            shard_size,
        }
    }

    /// Returns the size of the storage.
    ///
    /// # Returns
    /// The amount of parameters in the storage.
    pub fn len(&self) -> usize {
        self.params
    }
}

impl<O: Optimizer + Send> ParameterStore<O> {
    /// Accumulates a new gradient into the active gradient buffer.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    ///
    /// # Panics
    /// If the length of `grad` doesn't match the total number of parameters.
    pub(super) fn accumulate(&self, grad: &[f32]) {
        let active_idx = self.active_idx.load(Ordering::Acquire) as usize;

        self.shards
            .par_iter()
            .zip(grad.par_chunks(self.shard_size.get()))
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
    /// # Returns
    /// A `SizeMismatchErr` if there is a size mismatch in any of the inner shards.
    pub(super) fn pull_weights(&self, out: &mut [f32]) -> Result<(), SizeMismatchErr> {
        self.shards
            .par_iter()
            .zip(out.par_chunks_mut(self.shard_size.get()))
            .try_for_each(|(shard, out_slice)| shard.pull_weights(out_slice))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::initialization::ConstWeightGen;

    struct AddOptimizer;

    impl Optimizer for AddOptimizer {
        fn update_weights(&mut self, grad: &[f32], weights: &mut [f32]) {
            weights.iter_mut().zip(grad).for_each(|(w, g)| *w += g);
        }
    }

    fn create_test_store(params: usize, shard_size: usize) -> ParameterStore<AddOptimizer> {
        let shard_size = NonZeroUsize::new(shard_size).unwrap();
        let weight_gen = ConstWeightGen::new(0., params);
        ParameterStore::new(shard_size, weight_gen, |_| AddOptimizer)
    }

    #[test]
    fn test_handle_ragged_shards() {
        const PARAMS: usize = 15;
        const SHARD_SIZE: usize = 8;

        let store = create_test_store(PARAMS, SHARD_SIZE);
        let grad = [1.0; PARAMS];

        store.accumulate(&grad);
        store.update_weights();

        let mut out = [0.0; PARAMS];
        store.pull_weights(&mut out).unwrap();
        assert_eq!(out, [1.0; PARAMS]);
    }

    #[test]
    fn test_handle_buffer_swap() {
        const PARAMS: usize = 10;
        const SHARD_SIZE: usize = 1;

        let store = create_test_store(PARAMS, SHARD_SIZE);
        store.accumulate(&[1.0; PARAMS]);

        store.update_weights();
        assert_eq!(store.active_idx.load(Ordering::Acquire), 1);
        store.accumulate(&[5.0; PARAMS]);

        let mut weights = [0.0; PARAMS];
        store.pull_weights(&mut weights).unwrap();
        assert_eq!(weights, [1.0; PARAMS]);

        store.update_weights();
        store.pull_weights(&mut weights).unwrap();
        assert_eq!(weights, [6.0; PARAMS]);
    }

    #[test]
    fn test_update_locking_mechanism() {
        const PARAMS: usize = 10;
        const SHARD_SIZE: usize = 1;

        let store = create_test_store(PARAMS, SHARD_SIZE);
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
        const SHARD_SIZE: usize = 25;

        let store = create_test_store(PARAMS, SHARD_SIZE);

        let grad = [1.0; PARAMS];
        store.accumulate(&grad);
        store.update_weights();

        let mut weights = [0.0; PARAMS];
        store.pull_weights(&mut weights).unwrap();

        for (i, &w) in weights.iter().enumerate() {
            assert_eq!(w, 1.0, "Weight mismatch at index {i}");
        }
    }

    #[test]
    fn test_ragged_edge_distribution() {
        const PARAMS: usize = 105;
        const SHARD_SIZE: usize = 10;

        let store = create_test_store(PARAMS, SHARD_SIZE);

        let grad = [1.0; PARAMS];
        store.accumulate(&grad);
        store.update_weights();

        let mut weights = [0.0; PARAMS];
        store.pull_weights(&mut weights).unwrap();
        assert_eq!(weights.len(), PARAMS);
    }
}
