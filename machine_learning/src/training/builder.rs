use comms::specs::machine_learning::{
    ActFnSpec, LayerSpec, LossFnSpec, OptimizerSpec, TrainerSpec,
};
use rand::{SeedableRng, rngs::StdRng};

use super::{ModelTrainer, Trainer};
use crate::{
    arch::{
        Sequential,
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

                self.resolve_layers(spec, optimizers)
            }
            _ => unimplemented!(),
        }
    }

    /// Resolves the the `Layer`s for a `Sequential` model.
    ///
    /// # Arguments
    /// * `spec` - The specification of the trainer.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_layers<O>(&self, spec: TrainerSpec, optimizers: Vec<O>) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
    {
        let layers: Vec<_> = spec
            .layers
            .iter()
            .map(|layer_spec| self.resolve_layer(*layer_spec))
            .flatten()
            .collect();

        self.resolve_loss_fn(spec, optimizers, layers)
    }

    /// Resolves a `Layer` for a `Sequential` model.
    ///
    /// # Arguments
    /// * `spec` - The specification of a certain layer.
    ///
    /// # Returns
    /// A new `Layer`.
    fn resolve_layer(&self, spec: LayerSpec) -> Vec<Layer> {
        let mut layers = Vec::with_capacity(2);

        match spec {
            LayerSpec::Dense { dim, act_fn } => {
                layers.push(Layer::dense(dim));

                if let Some(spec) = act_fn {
                    layers.push(self.resolve_act_fn(spec));
                }
            }
        }

        layers
    }

    /// Resolves the `ActFn` for a specific layer.
    ///
    /// # Arguments
    /// * `spec` - An optional specification for an `ActFn`.
    ///
    /// # Returns
    /// A new `Layer`.
    fn resolve_act_fn(&self, spec: ActFnSpec) -> Layer {
        match spec {
            ActFnSpec::Sigmoid { amp } => Layer::sigmoid(amp),
        }
    }

    /// Resolves the `LossFn` for this trainer.
    ///
    /// # Arguments
    /// * `spec` - The specification for this trainer.
    /// * `optimizers` - A list of resolved optimizers, one per server.
    /// * `layers` - A list of resolved layers.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_loss_fn<O>(
        &self,
        spec: TrainerSpec,
        optimizers: Vec<O>,
        layers: Vec<Layer>,
    ) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
    {
        match spec.loss_fn {
            LossFnSpec::Mse => {
                let loss_fn = Mse::new();
                self.terminate_build(spec, optimizers, layers, loss_fn)
            }
        }
    }

    /// Terminates the entire build for this trainer and instanciates the final entity.
    ///
    /// # Arguments
    /// * `spec` - The specification for this trainer.
    /// * `optimizers` - A list of optimizers, one per server.
    /// * `layers` - A list of resolved layers.
    /// * `loss_fn` - A resolved loss function.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn terminate_build<O, L>(
        &self,
        spec: TrainerSpec,
        optimizers: Vec<O>,
        layers: Vec<Layer>,
        loss_fn: L,
    ) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
        L: LossFn + 'static,
    {
        let model = Sequential::new(layers);
        let dataset = Dataset::new(spec.dataset.data, spec.dataset.x_size, spec.dataset.y_size);
        let trainer = ModelTrainer::new(
            model,
            optimizers,
            dataset,
            spec.offline_epochs,
            spec.max_epochs,
            spec.batch_size,
            loss_fn,
            self.generate_rng(spec.seed),
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
