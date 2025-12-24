use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use rayon::prelude::*;

use crate::{
    optimization::Optimizer,
    parameter_server::{PSClient, ShardedGradient},
};

/// The central Parameter Server that manages global weights and gradient aggregation.
///
/// It utilizes a "double-buffering" strategy:
/// 1. Workers write to the `active` table.
/// 2. The server processes and resets the `frozen`.
///
/// This lets both server and workers work simultaneously.
#[derive(Debug)]
pub struct PSServer {
    active_idx: Arc<AtomicU8>,
    tables: [Arc<ShardedGradient>; 2],
    weights: Box<[f32]>,
    buf: Box<[f32]>,
}

impl PSServer {
    /// Creates a new `PSServer`.
    ///
    /// # Arguments
    /// * `params` - Total number of parameters in the model.
    /// * `shards_amount` - The amount of shards to partition the gradient.
    pub fn new(params: usize, shards_amount: NonZeroUsize) -> Self {
        Self {
            active_idx: Arc::new(AtomicU8::new(0)),
            tables: [
                Arc::new(ShardedGradient::new(params, shards_amount)),
                Arc::new(ShardedGradient::new(params, shards_amount)),
            ],
            buf: vec![0.; params].into_boxed_slice(),
            weights: vec![0.; params].into_boxed_slice(),
        }
    }

    /// Creates a new `PSClient` associated to this `PSServer`.
    ///
    /// # Returns
    /// `PSClient` - A new parameter server client handler.
    pub fn client_handle(&mut self) -> PSClient {
        let params = self.weights.len();
        let active_idx = Arc::clone(&self.active_idx);
        let tables = self.tables.each_ref().map(|table| Arc::clone(&table));
        PSClient::new(params, active_idx, tables)
    }

    /// Performs the weight update cycle.
    ///
    /// Will swap the underlying active/frozen gradient tables.
    ///
    /// # Arguments
    /// * `optimizer` - The optimization algorithm.
    pub fn update_weights<O>(&mut self, optimizer: &mut O)
    where
        O: Optimizer,
    {
        let frozen_idx = self.active_idx.fetch_xor(1, Ordering::AcqRel) as usize;
        let &ShardedGradient {
            ref shards,
            shard_size,
        } = self.tables[frozen_idx].as_ref();

        self.buf
            .par_chunks_mut(shard_size)
            .zip(shards)
            .for_each(|(grad_chunk, locked_shard)| {
                let mut nums = locked_shard.lock();
                for (acc, g) in grad_chunk.iter_mut().zip(nums.iter_mut()) {
                    *acc = *g;
                    *g = 0.;
                }
            });

        optimizer.update_weights(&mut self.weights, &self.buf);
    }

    pub fn get_weights(&self) -> &[f32] {
        &self.weights
    }
}
