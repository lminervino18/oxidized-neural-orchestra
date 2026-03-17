use std::num::NonZeroUsize;

use rand::Rng;

use super::{TrainResult, Trainer};
use crate::{
    Result,
    arch::{Sequential, loss::LossFn},
    dataset::Dataset,
    optimization::Optimizer,
    param_provider::ParamProvider,
};

/// A model `Trainer`. Contains the relevant components needed for training a model,
/// including the model itself.
pub struct BackpropTrainer<O, L, R>
where
    O: Optimizer,
    L: LossFn,
    R: Rng,
{
    model: Sequential,
    optimizers: Vec<O>,
    dataset: Dataset,
    loss_fn: L,

    epoch: usize,
    offline_epochs: usize,
    max_epochs: NonZeroUsize,
    batch_size: NonZeroUsize,
    rng: R,

    losses: Vec<f32>,
}

impl<O, L, R> BackpropTrainer<O, L, R>
where
    O: Optimizer,
    L: LossFn,
    R: Rng,
{
    /// Returns a new `BackpropTrainer` model trainer.
    ///
    /// # Arguments
    /// * `model` - A trainable model.
    /// * `optimizers` - A list of optimizers, one per server.
    /// * `dataset` - The dataset the model will be trained with.
    /// * `loss_fn` - The loss function used to measure the difference between a model's output and the expected one.
    /// * `offline_epochs` - The amount of extra epochs to run per `train` call.
    /// * `max_epochs` - The maximum amount of epochs to train.
    /// * `batch_size` - The size of the mini batch.
    /// * `rng` - A random number generator.
    ///
    /// # Returns
    /// A new `BackpropTrainer` instance.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Sequential,
        optimizers: Vec<O>,
        dataset: Dataset,
        loss_fn: L,
        offline_epochs: usize,
        max_epochs: NonZeroUsize,
        batch_size: NonZeroUsize,
        rng: R,
    ) -> Self {
        Self {
            model,
            optimizers,
            dataset,
            loss_fn,
            epoch: 0,
            offline_epochs,
            max_epochs,
            batch_size,
            rng,
            losses: Vec::with_capacity(1 + offline_epochs),
        }
    }
}

impl<O, L, R, P> Trainer<P> for BackpropTrainer<O, L, R>
where
    O: Optimizer + Send,
    L: LossFn,
    R: Rng,
    P: ParamProvider,
{
    /// Performs a training cycle.
    ///
    /// # Arguments
    /// * `param_manager` - The manager of parameters for this training.
    ///
    /// # Returns
    /// A tuple with the param grads and the epoch loss.
    fn train(&mut self, param_provider: &mut P) -> Result<TrainResult<'_>> {
        let remaining = self.max_epochs.get() - self.epoch;
        let epochs = remaining.min(self.offline_epochs + 1);

        self.losses.clear();

        for _ in 0..epochs {
            self.dataset.shuffle(&mut self.rng);
            let batches = self.dataset.batches(self.batch_size);

            let loss = self.model.backprop(
                param_provider,
                &mut self.optimizers,
                &mut self.loss_fn,
                batches,
            )?;

            self.losses.push(loss);
        }

        self.epoch += epochs;
        let res = TrainResult {
            losses: &self.losses,
            was_last: self.epoch == self.max_epochs.get(),
        };

        Ok(res)
    }
}
