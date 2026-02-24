use comms::specs::machine_learning::{
    ActFnSpec, LayerSpec, LossFnSpec, ModelSpec, OptimizerSpec, TrainerSpec,
};
use rand::{SeedableRng, rngs::StdRng};

use super::{ModelTrainer, Trainer};
use crate::{
    arch::{
        Model, Sequential,
        activations::ActFn,
        layers::Layer,
        loss::{LossFn, Mse},
    },
    dataset::Dataset,
    optimization::{GradientDescent, Optimizer},
};

/// Builds `Trainer`s given a specification.
pub struct TrainerBuilder;

impl TrainerBuilder {
    /// Creates a new `TrainerBuilder`.
    pub fn new() -> Self {
        Self
    }

    /// Builds a new `Trainer` following a spec.
    ///
    /// # Arguments
    /// * `spec` - The specification for the trainer.
    /// * `server_sizes` - The mount of parameters per server.
    ///
    /// # Returns
    /// A new `Trainer`.
    pub fn build(&self, spec: TrainerSpec, server_sizes: &[usize]) -> Box<dyn Trainer> {
        self.resolve_optimizers(spec, server_sizes)
    }

    /// Resolves the `Optimizer`s for this trainer.
    ///
    /// # Arguments
    /// * `spec` - The specification of the trainer.
    /// * `server_sizes` - The amount of parameters per server.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_optimizers(&self, spec: TrainerSpec, server_sizes: &[usize]) -> Box<dyn Trainer> {
        match spec.optimizer {
            OptimizerSpec::GradientDescent { learning_rate } => {
                let optimizers: Vec<_> = server_sizes
                    .iter()
                    .map(|_| GradientDescent::new(learning_rate))
                    .collect();

                self.resolve_model(spec, optimizers)
            }
            _ => unimplemented!(),
        }
    }

    /// Resolves the `Model` for this trainer.
    ///
    /// # Arguments
    /// * `spec` - The specification of the trainer.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_model<O>(&self, spec: TrainerSpec, optimizers: Vec<O>) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
    {
        match &spec.model {
            ModelSpec::Sequential {
                layers: layer_specs,
            } => {
                let (layers, sizes): (Vec<_>, Vec<_>) =
                    layer_specs.iter().map(|ls| self.resolve_layer(*ls)).unzip();

                let model = Sequential::new(layers);
                self.resolve_loss_fn(spec, optimizers, model)
            }
        }
    }

    /// Resolves the `Layer`s for a `Sequential` model.
    ///
    /// # Arguments
    /// * `spec` - The specification of a certain layer.
    ///
    /// # Returns
    /// A new `Layer`.
    fn resolve_layer(&self, spec: LayerSpec) -> (Layer, usize) {
        match spec {
            LayerSpec::Dense { dim, act_fn, size } => {
                let factory = |act_fn| Layer::dense(dim, act_fn);
                (self.resolve_act_fn(act_fn, factory), size)
            }
        }
    }

    /// Resolves the `ActFn` for a specific layer.
    ///
    /// # Arguments
    /// * `spec` - An optional specification for an `ActFn`.
    /// * `layer_factory` - A layer factory given an optional activation function.
    ///
    /// # Returns
    /// A new `Layer`.
    fn resolve_act_fn<F>(&self, spec: Option<ActFnSpec>, layer_factory: F) -> Layer
    where
        F: FnOnce(Option<ActFn>) -> Layer,
    {
        let Some(act_fn) = spec else {
            return layer_factory(None);
        };

        let act_fn = match act_fn {
            ActFnSpec::Sigmoid { amp } => Some(ActFn::sigmoid(amp)),
        };

        layer_factory(act_fn)
    }

    /// Resolves the `LossFn` for this trainer.
    ///
    /// # Arguments
    /// * `spec` - The specification for this trainer.
    /// * `optimizers` - A list of optimizers, one per server.
    /// * `model` - A resolved model.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_loss_fn<O, M>(
        &self,
        spec: TrainerSpec,
        optimizers: Vec<O>,
        model: M,
    ) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
        M: Model + 'static,
    {
        match spec.loss_fn {
            LossFnSpec::Mse => {
                let loss_fn = Mse::new();
                self.terminate_build(spec, optimizers, model, loss_fn)
            }
        }
    }

    /// Terminates the entire build for this trainer and instanciates the final entity.
    ///
    /// # Arguments
    /// * `spec` - The specification for this trainer.
    /// * `optimizers` - A list of optimizers, one per server.
    /// * `model` - A resolved model.
    /// * `loss_fn` - A resolved loss function.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn terminate_build<O, M, L>(
        &self,
        spec: TrainerSpec,
        optimizers: Vec<O>,
        model: M,
        loss_fn: L,
    ) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
        M: Model + 'static,
        L: LossFn + 'static,
    {
        let offline_epochs = spec.offline_epochs;
        let batch_size = spec.batch_size;
        let rng = self.generate_rng(spec.seed);
        let dataset = Dataset::new(spec.dataset.data, spec.dataset.x_size, spec.dataset.y_size);
        let trainer = ModelTrainer::new(
            model,
            optimizers,
            dataset,
            offline_epochs,
            batch_size,
            loss_fn,
            rng,
        );

        Box::new(trainer)
    }

    /// Generates a random number generator given (or not) a seed.
    ///
    /// # Arguments
    /// * `seed` - An optional seed for the rng.
    ///
    /// # Returns
    /// A new rng.
    fn generate_rng(&self, seed: Option<u64>) -> StdRng {
        match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        }
    }
}
