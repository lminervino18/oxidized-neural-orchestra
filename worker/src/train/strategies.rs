use crate::{
    data::dataloader::DataLoader,
    model::Model,
};

/// Strategy interface for computing gradients.
///
/// `compute_step` must:
/// - write grads into the provided `grads` buffer (flat)
/// - treat each call as exactly one "step" (here: one batch)
pub trait TrainStrategy {
    fn compute_step(&mut self, weights: &[f32], grads: &mut [f32]);
}

/// Supervised training strategy for 1D regression models.
///
/// Step semantics: one batch per step.
/// When the shard is exhausted, we reset and continue (cyclic over shard).
#[derive(Debug, Clone)]
pub struct SupervisedTrain1D {
    model: Model,
    loader: DataLoader,
    last_batch_len: usize,
}

impl SupervisedTrain1D {
    pub fn new(model: Model, loader: DataLoader) -> Self {
        Self {
            model,
            loader,
            last_batch_len: 0,
        }
    }

    #[inline]
    pub fn last_batch_len(&self) -> usize {
        self.last_batch_len
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
        let batch = match self.loader.next_batch() {
            Some(b) => b,
            None => {
                self.loader.reset();
                self.loader
                    .next_batch()
                    .expect("dataset/shard must produce at least one batch")
            }
        };

        self.last_batch_len = batch.len();
        self.model
            .grad_batch(weights, grads, &batch.xs, &batch.ys);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use crate::{
        data::{dataset::InMemoryDataset, shard::ShardSpec},
        model::{spec::ModelSpec, Model},
    };

    #[test]
    fn strategy_cycles_over_shard_batches() {
        let ds = InMemoryDataset::new(vec![1., 2., 3., 4., 5.], vec![3., 5., 7., 9., 11.]);
        let shard = ShardSpec::new(0, NonZeroUsize::new(1).unwrap());
        let loader = DataLoader::new(ds, shard, 2);

        let model = Model::new(ModelSpec::LinearRegression1D);
        let mut strat = SupervisedTrain1D::new(model, loader);

        let weights = [0.0_f32, 0.0_f32];
        let mut grads = [0.0_f32, 0.0_f32];

        strat.compute_step(&weights, &mut grads);
        assert_eq!(strat.last_batch_len(), 2);

        strat.compute_step(&weights, &mut grads);
        assert_eq!(strat.last_batch_len(), 2);

        strat.compute_step(&weights, &mut grads);
        assert_eq!(strat.last_batch_len(), 1);

        strat.compute_step(&weights, &mut grads);
        assert_eq!(strat.last_batch_len(), 2);
    }
}
