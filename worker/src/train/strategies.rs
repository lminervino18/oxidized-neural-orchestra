use std::num::NonZeroUsize;

use crate::{
    data::dataloader::{BatchSpec, DataLoader},
    model::Model,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepStats {
    pub microbatches: usize,
    pub samples: usize,
}

pub trait TrainStrategy {
    fn compute_step(&mut self, weights: &[f32], grads: &mut [f32]);
    fn last_step_stats(&self) -> StepStats;
}

/// Supervised training strategy for 1D regression models with microbatch accumulation.
///
/// One worker "step" consumes `microbatch_k` batches locally, accumulates,
/// and normalizes by total samples.
#[derive(Debug, Clone)]
pub struct SupervisedTrain1D {
    model: Model,
    loader: DataLoader,
    microbatch_k: NonZeroUsize,

    /// Scratch buffer for per-batch gradient (reused, no allocations).
    scratch: Vec<f32>,

    /// Scratch specs for this step (reused, no allocations).
    specs: Vec<BatchSpec>,

    last_microbatch_batches: usize,
    last_microbatch_samples: usize,
}

impl SupervisedTrain1D {
    pub fn new(model: Model, loader: DataLoader, microbatch_k: NonZeroUsize) -> Self {
        let num_params = model.num_params();
        let k = microbatch_k.get();
        Self {
            model,
            loader,
            microbatch_k,
            scratch: vec![0.0; num_params],
            specs: Vec::with_capacity(k),
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

        // 1) Plan: collect specs (this mutates the loader).
        self.specs.clear();
        for _ in 0..k {
            let spec = match self.loader.next_spec() {
                Some(s) => s,
                None => {
                    self.loader.reset();
                    self.loader
                        .next_spec()
                        .expect("dataset/shard must produce at least one batch")
                }
            };
            self.specs.push(spec);
        }

        // 2) Execute: now borrow dataset immutably and compute.
        let ds = self.loader.dataset();
        let xs_all = ds.xs();
        let ys_all = ds.ys();

        let mut total_samples = 0usize;

        for spec in self.specs.iter().copied() {
            let xs = &xs_all[spec.start..spec.end];
            let ys = &ys_all[spec.start..spec.end];

            self.scratch.fill(0.0);
            self.model.grad_batch(weights, &mut self.scratch, xs, ys);

            grads
                .iter_mut()
                .zip(self.scratch.iter())
                .for_each(|(g_acc, g)| *g_acc += *g);

            total_samples += spec.len();
        }

        // 3) Normalize by samples (robust even if last batch smaller).
        let denom = total_samples.max(1) as f32;
        grads.iter_mut().for_each(|g| *g /= denom);

        self.last_microbatch_batches = k;
        self.last_microbatch_samples = total_samples;
    }

    #[inline]
    fn last_step_stats(&self) -> StepStats {
        StepStats {
            microbatches: self.last_microbatch_batches,
            samples: self.last_microbatch_samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use crate::{
        data::{dataloader::DataLoader, dataset::InMemoryDataset, shard::ShardSpec},
        model::{spec::ModelSpec, Model},
    };

    #[test]
    fn microbatch_accumulates_and_normalizes_by_samples() {
        // dataset len 5, batch_size 2 => specs: [0..2, 2..4, 4..5]
        let ds = InMemoryDataset::new(vec![1., 2., 3., 4., 5.], vec![3., 5., 7., 9., 11.]);
        let shard = ShardSpec::new(0, NonZeroUsize::new(1).unwrap());
        let loader = DataLoader::new(ds, shard, 2);

        let model = Model::new(ModelSpec::LinearRegression1D);

        // microbatch_k = 2 => consumes first two batches => total_samples = 4
        let mut strat =
            SupervisedTrain1D::new(model.clone(), loader, NonZeroUsize::new(2).unwrap());

        let weights = [0.5_f32, 0.0_f32];
        let mut out = vec![0.0_f32; model.num_params()];

        strat.compute_step(&weights, &mut out);

        assert_eq!(strat.last_microbatch_batches(), 2);
        assert_eq!(strat.last_microbatch_samples(), 4);
        assert_eq!(
            strat.last_step_stats(),
            StepStats {
                microbatches: 2,
                samples: 4
            }
        );

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
