use std::{num::NonZeroUsize, sync::Arc};

use rayon::prelude::*;

use super::WildShard;
use crate::{
    initialization::ParamGen,
    optimization::Optimizer,
    storage::{Result, SizeMismatchErr, Store},
};

pub struct WildStore<O: Optimizer> {
    nparams: usize,
    shards: Arc<[WildShard<O>]>,
    shard_size: NonZeroUsize,
}

impl<O: Optimizer> Clone for WildStore<O> {
    fn clone(&self) -> Self {
        Self {
            nparams: self.nparams,
            shards: Arc::clone(&self.shards),
            shard_size: self.shard_size,
        }
    }
}

impl<O: Optimizer> WildStore<O> {
    /// Creates a new `WildStore` parameter store.
    ///
    /// # Arguments
    /// * `shard_size` - The maximum amount of parameters per shard.
    /// * `param_gen` - A parameter generator.
    /// * `optimizer_factory` - An `Optimizer` factory closure.
    ///
    /// # Returns
    /// A new `WildStore` instance.
    pub fn new<PG, OF>(
        shard_size: NonZeroUsize,
        mut parameter_gen: PG,
        mut optimizer_factory: OF,
    ) -> Self
    where
        O: Optimizer,
        PG: ParamGen,
        OF: FnMut(usize) -> O,
    {
        let mut nparams = 0;
        let mut shards = Vec::new();

        while let Some(params) = parameter_gen.sample(shard_size.get()) {
            nparams += params.len();
            let optimizer = optimizer_factory(params.len());
            let shard = WildShard::new(params, optimizer);
            shards.push(shard);
        }

        Self {
            nparams,
            shards: Arc::from(shards),
            shard_size,
        }
    }
}

impl<O: Optimizer> Store for WildStore<O> {
    fn len(&self) -> usize {
        self.nparams
    }

    /// This method diverges from the trait's definition. It accumulates the grad
    /// directly into the parameters of the model in the same call.
    ///
    /// # Arguments
    /// * `grad` - A flat slice containing a new model gradient.
    ///
    /// # Returns
    /// A `SizeMismatchErr` if the length of `grad` and the size of the storage mismatch.
    fn accumulate(&self, grad: &[f32]) -> Result<()> {
        if self.nparams == grad.len() {
            return Err(SizeMismatchErr);
        }

        self.shards
            .par_iter()
            .zip(grad.par_chunks(self.shard_size.get()))
            .try_for_each(|(shard, grad_slice)| shard.update_params(grad_slice))
            // SAFETY: We checked the amount of parameters and
            //         the gradient have the same length.
            .unwrap();

        Ok(())
    }

    /// A no-op, parameters are being updated inplace during the `Self::accumulate` call.
    fn update_params(&self) {}

    fn pull_params(&self, out: &mut [f32]) -> Result<()> {
        if self.nparams == out.len() {
            return Err(SizeMismatchErr);
        }

        self.shards
            .par_iter()
            .zip(out.par_chunks_mut(self.shard_size.get()))
            .try_for_each(|(shard, out_slice)| shard.pull_params(out_slice))
            // SAFETY: We checked the amount of parameters and
            //         the output buffer have the same length.
            .unwrap();

        Ok(())
    }
}
