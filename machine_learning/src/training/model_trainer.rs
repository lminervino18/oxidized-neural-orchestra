use rand::Rng;

use super::Trainer;
use crate::{
    arch::{loss::LossFn, Model},
    dataset::Dataset,
    optimization::Optimizer,
};

/// A model `Trainer`. Contains the relevant components needed for training a model,
/// including the model itself.
pub struct ModelTrainer<M: Model, O: Optimizer, L: LossFn, R: Rng> {
    grad: Vec<f32>,
    dataset: Dataset,
    optimizer: O,
    model: M,
    loss: L,

    iters: usize,
    batch_size: usize,
    rng: R,
}

impl<M: Model, O: Optimizer, L: LossFn, R: Rng> Trainer for ModelTrainer<M, O, L, R> {}

impl<M: Model, O: Optimizer, L: LossFn, R: Rng> ModelTrainer<M, O, L, R> {
    /// Returns a new `ModelTrainer`.
    ///
    /// # Arguments
    /// * `model` - The model that will be trained.
    /// * `optimizer` - The way of optimizing the model (e.g. stochastic gradient descent).
    /// * `dataset` - The dataset the model will be trained with.
    /// * `iters` - The amont of training iterations performed on each call to `train`.
    /// * `loss` - The loss function used to measure the difference between a model's output and the
    /// expected one.
    /// * `rng` - A random number generator.
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
            iters: offline_iters,
            batch_size,
            loss,
            rng,
        }
    }
}

impl<M: Model, O: Optimizer + Send, L: LossFn, R: Rng> ModelTrainer<M, O, L, R> {
    /// Performs `iters` iterations of training its model, using its optimizer, dataset, loss
    /// function and batch size.
    ///
    /// # Arguments
    /// * `params` - The parameters that will be optimized for the model that's being trained.
    pub fn train(&mut self, params: &mut [f32]) -> &[f32] {
        for i in 0..self.iters + 1 {
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
