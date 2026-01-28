use rand::Rng;

use crate::{
    arch::{loss::LossFn, Model},
    dataset::Dataset,
    optimization::Optimizer,
};

pub struct Trainer<M: Model, O: Optimizer, L: LossFn, R: Rng> {
    grad: Vec<f32>,
    dataset: Dataset,
    optimizer: O,
    model: M,
    loss: L,

    offline_iters: usize,
    batch_size: usize,
    rng: R,
}

impl<M: Model, O: Optimizer, L: LossFn, R: Rng> Trainer<M, O, L, R> {
    pub fn new(
        model: M,
        optimizer: O,
        dataset: Dataset,
        offline_iters: usize,
        batch_size: usize,
        loss: L,
        rng: R,
    ) -> Self {
        Self {
            grad: vec![0.0; model.size()],
            model,
            optimizer,
            dataset,
            offline_iters,
            batch_size,
            loss,
            rng,
        }
    }
}

impl<M: Model, O: Optimizer + Send, L: LossFn, R: Rng> Trainer<M, O, L, R> {
    pub fn train(&mut self, params: &mut [f32]) -> &[f32] {
        for i in 0..self.offline_iters + 1 {
            println!("epoch {i}");

            self.dataset.shuffle(&mut self.rng);
            let batches = self.dataset.batches(self.batch_size);

            self.model.backprop(
                params,
                &mut self.grad,
                &self.loss,
                &mut self.optimizer,
                batches,
            );
        }

        &self.grad
    }
}
