use rand::Rng;

use super::Trainer;
use crate::{
    arch::{Model, loss::LossFn},
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

    epochs: usize,
    batch_size: usize,
    rng: R,
}

impl<M: Model, O: Optimizer, L: LossFn, R: Rng> ModelTrainer<M, O, L, R> {
    /// Returns a new `ModelTrainer`.
    ///
    /// # Arguments
    /// * `model` - The model that will be trained.
    /// * `optimizer` - The way of optimizing the model (e.g. stochastic gradient descent).
    /// * `dataset` - The dataset the model will be trained with.
    /// * `epochs` - The amount of training epochs performed on each call to `train`.
    /// * `loss` - The loss function used to measure the difference between a model's output and the expected one.
    /// * `rng` - A random number generator.
    pub fn new(
        model: M,
        optimizer: O,
        dataset: Dataset,
        epochs: usize,
        batch_size: usize,
        loss: L,
        rng: R,
    ) -> Self {
        Self {
            grad: vec![0.0; model.size()],
            model,
            optimizer,
            dataset,
            epochs,
            batch_size,
            loss,
            rng,
        }
    }
}

impl<M: Model, O: Optimizer, L: LossFn, R: Rng> ModelTrainer<M, O, L, R> {
    /// Performs `epochs` epochs of training its model, using its optimizer, dataset, loss
    /// function and batch size.
    ///
    /// # Arguments
    /// * `params` - The parameters that will be optimized for the model that's being trained.
    ///
    /// # Returns
    /// A tuple with the param grads and the epoch loss.
    pub fn train(&mut self, params: &mut [f32]) -> (&[f32], Vec<f32>) {
        let mut losses = Vec::with_capacity(self.epochs);

        for i in 0..self.epochs {
            println!("epoch {i}");

            self.dataset.shuffle(&mut self.rng);
            let batches = self.dataset.batches(self.batch_size);

            losses.push(self.model.backprop(
                params,
                &mut self.grad,
                &self.loss,
                &mut self.optimizer,
                batches,
            ));
        }

        (&self.grad, losses)
    }
}

impl<M, O, L, R> Trainer for ModelTrainer<M, O, L, R>
where
    M: Model,
    O: Optimizer,
    L: LossFn,
    R: Rng,
{
    fn train(&mut self, params: &mut [f32]) -> (&[f32], Vec<f32>) {
        self.train(params)
    }
}
