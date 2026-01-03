use std::num::NonZeroUsize;

use crate::{
    data::dataloader::DataLoader,
    model::Model,
};

/// Strategy interface for computing gradients.
///
/// Contract:
/// - must write final gradient into `grads` (flat buffer)
/// - each call corresponds to exactly one *network step* (one gradient send)
pub trait TrainStrategy {
    fn compute_step(&mut self, weights: &[f32], grads: &mut [f32]);
}

/// Supervised training strategy for 1D regression models with microbatch accumulation.
///
/// Semantics:
/// - One worker "step" = consume `microbatch_k` batches locally.
/// - Accumulate gradients over those batches.
/// - Normalize by total number of samples seen (preferred over /k when batch sizes vary).
///
/// The DataLoader is shard-aware and cycles when exhausted.
#[derive(Debug, Clone)]
pub struct SupervisedTrain1D {
    model: Model,
    loader: DataLoader,
    microbatch_k: NonZeroUsize,

    /// Scratch buffer for per-batch gradient, reused each iteration (no allocations).
    scratch: Vec<f32>,

    /// Diagnostics only.
    last_microbatch_batches: usize,
    last_microbatch_samples: usize,
}

impl SupervisedTrain1D {
    pub fn new(model: Model, loader: DataLoader, microbatch_k: NonZeroUsize) -> Self {
        let num_params = model.num_params();
        Self {
            model,
            loader,
            microbatch_k,
            scratch: vec![0.0; num_params],
            last_microbatch_batches: 0,
            last_microbatch_samples: 0,
        }
    }

    #[inline]
    pub fn last_microbatch_batches(&self) -> usize {
        self.last_microbatch_batches
    }

    #[inline]
    pub fn last_microbatch_samples(&self) -> usize {
        self.last_microbatch_samples
    }

    #[inline]
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[inline]
    pub fn loader(&self) -> &DataLoader {
        &self.loader
    }
}

impl TrainStrategy for SupervisedTrain1D {
    fn compute_step(&mut self, weights: &[f32], grads: &mut [f32]) {
        grads.fill(0.0);

        let k = self.microbatch_k.get();
        let mut total_samples = 0usize;

        for _ in 0..k {
            let batch = match self.loader.next_batch() {
                Some(b) => b,
                None => {
                    self.loader.reset();
                    self.loader
                        .next_batch()
                        .expect("dataset/shard must produce at least one batch")
                }
            };

            // Compute batch gradient into scratch (overwrite), then accumulate into grads.
            self.scratch.fill(0.0);
            self.model
                .grad_batch(weights, &mut self.scratch, batch.xs, batch.ys);

            grads
                .iter_mut()
                .zip(self.scratch.iter())
                .for_each(|(g_acc, g)| *g_acc += *g);

            total_samples += batch.len();
        }

        // Normalize by samples (robust even if last batch is smaller).
        let denom = total_samples.max(1) as f32;
        grads.iter_mut().for_each(|g| *g /= denom);

        self.last_microbatch_batches = k;
        self.last_microbatch_samples = total_samples;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use crate::{
        data::{dataset::InMemoryDataset, shard::ShardSpec, dataloader::DataLoader},
        model::{spec::ModelSpec, Model},
    };

    #[test]
    fn microbatch_accumulates_and_normalizes_by_samples() {
        // dataset len 5, batch_size 2 => batches: [2,2,1]
        let ds = InMemoryDataset::new(vec![1., 2., 3., 4., 5.], vec![3., 5., 7., 9., 11.]);
        let shard = ShardSpec::new(0, NonZeroUsize::new(1).unwrap());
        let loader = DataLoader::new(ds, shard, 2);

        let model = Model::new(ModelSpec::LinearRegression1D);

        // microbatch_k = 2 => consumes first two batches => total_samples = 4
        let mut strat = SupervisedTrain1D::new(model.clone(), loader, NonZeroUsize::new(2).unwrap());

        let weights = [0.5_f32, 0.0_f32];
        let mut out = vec![0.0_f32; model.num_params()];

        strat.compute_step(&weights, &mut out);

        assert_eq!(strat.last_microbatch_batches(), 2);
        assert_eq!(strat.last_microbatch_samples(), 4);

        // Validate: expected = (grad(batch1)+grad(batch2)) / total_samples
        let b1_x = vec![1.0, 2.0];
        let b1_y = vec![3.0, 5.0];
        let b2_x = vec![3.0, 4.0];
        let b2_y = vec![7.0, 9.0];

        let mut g1 = vec![0.0_f32; model.num_params()];
        let mut g2 = vec![0.0_f32; model.num_params()];
        model.grad_batch(&weights, &mut g1, &b1_x, &b1_y);
        model.grad_batch(&weights, &mut g2, &b2_x, &b2_y);

        let expected0 = (g1[0] + g2[0]) / 4.0;
        let expected1 = (g1[1] + g2[1]) / 4.0;

        assert!((out[0] - expected0).abs() < 1e-6);
        assert!((out[1] - expected1).abs() < 1e-6);
    }
}
