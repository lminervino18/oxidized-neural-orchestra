use super::{ModelTrainer, Trainer};
use crate::{
    arch::{
        activations::ActFn,
        layers::Layer,
        loss::{LossFn, Mse},
        Model, Sequential,
    },
    dataset::Dataset,
    optimization::{GradientDescent, Optimizer},
};
use comms::specs::machine_learning::{
    ActFnSpec, LayerSpec, LossFnSpec, ModelSpec, OptimizerSpec, TrainerSpec,
};
use rand::{rngs::StdRng, SeedableRng};

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
    pub fn build(&self, spec: &TrainerSpec) -> Box<dyn Trainer> {
        self.resolve_model(spec)
    }

    fn resolve_model(&self, spec: &TrainerSpec) -> Box<dyn Trainer> {
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

    fn resolve_layer(&self, spec: LayerSpec) -> Layer {
        match spec {
            LayerSpec::Dense { dim, act_fn } => {
                let factory = |act_fn| Layer::dense(dim, act_fn);
                self.resolve_act_fn(act_fn, factory)
            }
        }
    }

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

    fn resolve_optimizer<M>(&self, spec: &TrainerSpec, model: M) -> Box<dyn Trainer>
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

    fn resolve_loss<M, O>(&self, spec: &TrainerSpec, model: M, optimizer: O) -> Box<dyn Trainer>
    where
        M: Model + 'static,
        O: Optimizer + 'static,
    {
        match spec.loss {
            LossFnSpec::Mse => {
                let loss = Mse::new();
                self.terminate_build(spec, model, optimizer, loss)
            }
        }
    }

    fn terminate_build<M, O, L>(
        &self,
        spec: &TrainerSpec,
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
        let dataset = Dataset::new(vec![], 0, 0); // TODO

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

    fn generate_rng(&self, seed: Option<u64>) -> StdRng {
        match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        }
    }
}
