use std::num::NonZeroUsize;

use rand::Rng;

use super::Trainer;
use crate::{
    arch::{Model, loss::LossFn},
    dataset::Dataset,
    middleware::ParamManager,
    optimization::Optimizer,
};

/// A model `Trainer`. Contains the relevant components needed for training a model,
/// including the model itself.
pub struct ModelTrainer<M, O, L, R>
where
    M: Model,
    O: Optimizer,
    L: LossFn,
    R: Rng,
{
    optimizers: Vec<O>,
    dataset: Dataset,
    loss_fn: L,
    model: M,

    offline_epochs: usize,
    batch_size: NonZeroUsize,
    rng: R,
}

impl<M, O, L, R> ModelTrainer<M, O, L, R>
where
    M: Model,
    O: Optimizer,
    L: LossFn,
    R: Rng,
{
    /// Returns a new `ModelTrainer`.
    ///
    /// # Arguments
    /// * `model` - The model that will be trained.
    /// * `optimizers` - A list of optimizers, one per server.
    /// * `dataset` - The dataset the model will be trained with.
    /// * `offline_epochs` - The amount of extra epochs to run per `train` call.
    /// * `loss` - The loss function used to measure the difference between a model's output and the expected one.
    /// * `rng` - A random number generator.
    pub fn new(
        model: M,
        optimizers: Vec<O>,
        dataset: Dataset,
        offline_epochs: usize,
        batch_size: NonZeroUsize,
        loss_fn: L,
        rng: R,
    ) -> Self {
        Self {
            model,
            dataset,
            optimizers,
            offline_epochs,
            batch_size,
            loss_fn,
            rng,
        }
    }
}

impl<M, O, L, R> ModelTrainer<M, O, L, R>
where
    M: Model,
    O: Optimizer + Send,
    L: LossFn,
    R: Rng,
{
    /// Performs `epochs` epochs of training its model, using its optimizer, dataset, loss
    /// function and batch size.
    ///
    /// # Arguments
    /// * `param_manager` - The manager of parameters for this training.
    ///
    /// # Returns
    /// A tuple with the param grads and the epoch loss.
    pub fn train<'mw>(&mut self, param_manager: &mut ParamManager<'mw>) -> Vec<f32> {
        let epochs = self.offline_epochs + 1;
        let mut losses = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            self.dataset.shuffle(&mut self.rng);
            let batches = self.dataset.batches(self.batch_size);

            let loss =
                self.model
                    .backprop(param_manager, &mut self.optimizers, &self.loss_fn, batches);

            losses.push(loss);
        }

        losses
    }
}

impl<M, O, L, R> Trainer for ModelTrainer<M, O, L, R>
where
    M: Model,
    O: Optimizer + Send,
    L: LossFn,
    R: Rng,
{
    fn train(&mut self, param_manager: &mut ParamManager<'_>) -> Vec<f32> {
        self.train(param_manager)
    }
}
