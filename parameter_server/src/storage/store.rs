use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

use rayon::prelude::*;

use super::{ParameterShard, Result};
use crate::{initialization::ParamGen, optimization::Optimizer};

/// The primary storage of parameters and accumulated gradients.
///
/// These methods are private to the module, they become available
/// through the async interface of a `ParameterHandle`.
#[derive(Debug)]
pub struct ParameterStore<O: Optimizer> {
    nparams: usize,
    active_idx: Arc<AtomicU8>,
    updating: Arc<AtomicBool>,
    shards: Arc<[ParameterShard<O>]>,
    shard_size: NonZeroUsize,
}

impl<O: Optimizer> Clone for ParameterStore<O> {
    fn clone(&self) -> Self {
        Self {
            nparams: self.nparams,
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
    /// * `param_gen` - A parameter generator.
    /// * `optimizer_factory` - An `Optimizer` factory closure.
    ///
    /// # Returns
    /// A new `ParameterStore` instance.
    pub fn new<PG, OF>(
        shard_size: NonZeroUsize,
        mut parameter_gen: PG,
        mut optimizer_factory: OF,
    ) -> Self
    where
        PG: ParamGen,
        OF: FnMut(usize) -> O,
    {
        let mut nparams = 0;
        let mut shards = Vec::new();

        while let Some(params) = parameter_gen.sample(shard_size.get()) {
            nparams += params.len();
            let optimizer = optimizer_factory(params.len());
            let shard = ParameterShard::new(params, optimizer);
            shards.push(shard);
        }

        Self {
            nparams,
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
        self.nparams
    }
}

impl<O: Optimizer + Send> ParameterStore<O> {
    /// Accumulates a new gradient into the active gradient buffer.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if there is a size mismatch in any of the inner shards.
    pub(super) fn accumulate(&self, grad: &[f32]) -> Result<()> {
        let active_idx = self.active_idx.load(Ordering::Acquire) as usize;

        self.shards
            .par_iter()
            .zip(grad.par_chunks(self.shard_size.get()))
            .try_for_each(|(shard, grad_slice)| shard.accumulate(active_idx, grad_slice))
    }

    /// Swaps the active gradient buffer and applies the frozen gradient to the parameters.
    ///
    /// This triggers a parallel update across all shards.
    pub(super) fn update_params(&self) {
        let success = self
            .updating
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok();

        if success {
            let frozen_idx = self.active_idx.fetch_xor(1, Ordering::AcqRel) as usize;

            self.shards
                .par_iter()
                .for_each(|shard| shard.update_params(frozen_idx));

            self.updating.store(false, Ordering::Release);
        }
    }

    /// Gathers all the sharded parameters into a local buffer.
    ///
    /// # Arguments
    /// * `out` - A mutable slice where the parameters will be copied.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if there is a size mismatch in any of the inner shards.
    pub(super) fn pull_params(&self, out: &mut [f32]) -> Result<()> {
        let shard_size = self.shard_size.get();

        self.shards
            .par_iter()
            .zip(out.par_chunks_mut(shard_size))
            .try_for_each(|(shard, out_slice)| shard.pull_params(out_slice))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::initialization::ConstParamGen;

    struct AddOptimizer;

    impl Optimizer for AddOptimizer {
        fn update_params(&mut self, grad: &[f32], params: &mut [f32]) -> Result<()> {
            params.iter_mut().zip(grad).for_each(|(w, g)| *w += g);
            Ok(())
        }
    }

    fn create_test_store(params: usize, shard_size: usize) -> ParameterStore<AddOptimizer> {
        let shard_size = NonZeroUsize::new(shard_size).unwrap();
        let param_gen = ConstParamGen::new(0., params);
        ParameterStore::new(shard_size, param_gen, |_| AddOptimizer)
    }

    #[test]
    fn test_handle_ragged_shards() {
        const PARAMS: usize = 15;
        const SHARD_SIZE: usize = 8;

        let store = create_test_store(PARAMS, SHARD_SIZE);
        let grad = [1.0; PARAMS];

        store.accumulate(&grad).unwrap();
        store.update_params();

        let mut out = [0.0; PARAMS];
        store.pull_params(&mut out).unwrap();
        assert_eq!(out, [1.0; PARAMS]);
    }

    #[test]
    fn test_handle_buffer_swap() {
        const PARAMS: usize = 10;
        const SHARD_SIZE: usize = 1;

        let store = create_test_store(PARAMS, SHARD_SIZE);
        store.accumulate(&[1.0; PARAMS]).unwrap();

        store.update_params();
        assert_eq!(store.active_idx.load(Ordering::Acquire), 1);
        store.accumulate(&[5.0; PARAMS]).unwrap();

        let mut params = [0.0; PARAMS];
        store.pull_params(&mut params).unwrap();
        assert_eq!(params, [1.0; PARAMS]);

        store.update_params();
        store.pull_params(&mut params).unwrap();
        assert_eq!(params, [6.0; PARAMS]);
    }

    #[test]
    fn test_update_locking_mechanism() {
        const PARAMS: usize = 10;
        const SHARD_SIZE: usize = 1;

        let store = create_test_store(PARAMS, SHARD_SIZE);
        store.updating.store(true, Ordering::SeqCst);

        let active_idx = store.active_idx.load(Ordering::Acquire);
        store.update_params();
        assert_eq!(store.active_idx.load(Ordering::Acquire), active_idx);

        store.updating.store(false, Ordering::Release);
        store.update_params();
        assert_ne!(store.active_idx.load(Ordering::SeqCst), active_idx);
    }

    #[test]
    fn test_store_initialization_and_flow() {
        const PARAMS: usize = 100;
        const SHARD_SIZE: usize = 25;

        let store = create_test_store(PARAMS, SHARD_SIZE);

        let grad = [1.0; PARAMS];
        store.accumulate(&grad).unwrap();
        store.update_params();

        let mut params = [0.0; PARAMS];
        store.pull_params(&mut params).unwrap();

        for (i, &w) in params.iter().enumerate() {
            assert_eq!(w, 1.0, "Parameter mismatch at index {i}");
        }
    }

    #[test]
    fn test_ragged_edge_distribution() {
        const PARAMS: usize = 105;
        const SHARD_SIZE: usize = 10;

        let store = create_test_store(PARAMS, SHARD_SIZE);

        let grad = [1.0; PARAMS];
        store.accumulate(&grad).unwrap();
        store.update_params();

        let mut params = [0.0; PARAMS];
        store.pull_params(&mut params).unwrap();
        assert_eq!(params.len(), PARAMS);
    }
}
