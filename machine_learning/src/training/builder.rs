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
use comms::specs::machine_learning::{
    ActFnSpec, LayerSpec, LossFnSpec, ModelSpec, OptimizerSpec, TrainerSpec,
};
use rand::{SeedableRng, rngs::StdRng};

/// Builds `Trainer`s given a specification.
#[derive(Default)]
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
    ///
    /// # Returns
    /// A new `Trainer`.
    pub fn build(&self, spec: TrainerSpec) -> Box<dyn Trainer> {
        self.resolve_model(spec)
    }

    /// Resolves the `Model` for this trainer.
    ///
    /// # Arguments
    /// * `spec` - The specification of the trainer.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_model(&self, spec: TrainerSpec) -> Box<dyn Trainer> {
        match &spec.model {
            ModelSpec::Sequential {
                layers: layer_specs,
            } => {
                let layers = layer_specs.iter().map(|ls| self.resolve_layer(*ls));
                let model = Sequential::new(layers);
                self.resolve_optimizer(spec, model)
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
    fn resolve_layer(&self, spec: LayerSpec) -> Layer {
        match spec {
            LayerSpec::Dense { dim, act_fn } => {
                let factory = |act_fn| Layer::dense(dim, act_fn);
                self.resolve_act_fn(act_fn, factory)
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

    /// Resolves the `Optimizer` for this trainer.
    ///
    /// # Arguments
    /// * `spec` - The specification for this trainer.
    /// * `model` - A resolved model.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_optimizer<M>(&self, spec: TrainerSpec, model: M) -> Box<dyn Trainer>
    where
        M: Model + 'static,
    {
        match spec.optimizer {
            OptimizerSpec::GradientDescent { learning_rate } => {
                let optimizer = GradientDescent::new(learning_rate);
                self.resolve_loss(spec, model, optimizer)
            }
            _ => unimplemented!(),
        }
    }

    /// Resolves the `LossFn` for this trainer.
    ///
    /// # Arguments
    /// * `spec` - The specification for this trainer.
    /// * `model` - A resolved model.
    /// * `optimizer` - A resolved optimizer.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_loss<M, O>(&self, spec: TrainerSpec, model: M, optimizer: O) -> Box<dyn Trainer>
    where
        M: Model + 'static,
        O: Optimizer + 'static,
    {
        match spec.loss_fn {
            LossFnSpec::Mse => {
                let loss = Mse::new();
                self.terminate_build(spec, model, optimizer, loss)
            }
        }
    }

    /// Terminates the entire build for this trainer and instanciates the final entity.
    ///
    /// # Arguments
    /// * `spec` - The specification for this trainer.
    /// * `model` - A resolved model.
    /// * `optimizer` - A resolved optimizer.
    /// * `loss` - A resolved loss function.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn terminate_build<M, O, L>(
        &self,
        spec: TrainerSpec,
        model: M,
        optimizer: O,
        loss: L,
    ) -> Box<dyn Trainer>
    where
        M: Model + 'static,
        O: Optimizer + 'static,
        L: LossFn + 'static,
    {
        let offline_iters = spec.offline_epochs;
        let batch_size = spec.batch_size;
        let rng = self.generate_rng(spec.seed);
        let dataset = Dataset::new(spec.dataset.data, spec.dataset.x_size, spec.dataset.y_size);

        let trainer = ModelTrainer::new(
            model,
            optimizer,
            dataset,
            offline_iters,
            batch_size,
            loss,
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
