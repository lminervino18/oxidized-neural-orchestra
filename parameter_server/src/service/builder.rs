use std::{cell::RefCell, rc::Rc};

use comms::specs::server::{
    DistributionSpec, OptimizerSpec, ServerSpec, TrainerSpec, WeightGenSpec,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokio::io::{AsyncRead, AsyncWrite};

use super::{ParameterServer, Server};
use crate::{
    initialization::{ChainedWeightGen, ConstWeightGen, RandWeightGen, Result, WeightGen},
    optimization::{Adam, GradientDescent, GradientDescentWithMomentum, Optimizer},
    storage::ParameterStore,
    training::{BarrierSyncTrainer, NonBlockingTrainer, Trainer},
};

/// Makes `callback`'s return type generic, when trying to resolve a concrete `RandWeightGen` it avoids boxing all
/// weight generators variants.
///
/// When trying to resolve for a `ChainedWeightGen`, a sub `RandWeightGen` must be returned as a boxed
/// `WeightGen`, so it must duplicate the distribution's static dispatch match.
///
/// To avoid this duplication, this macro generalizes what it's immediately done with the concrete weight generator,
/// and thus avoids boxing weight generators unnecessarily.
///
/// # Arguments
/// * `rng` - A random number generator.
/// * `dist_spec` - A specification for a distribution.
/// * `limit` - The limit of weights that the `RandWeightGen` can generate.
/// * `callback` - The closure to call passing in the created weigth gen.
macro_rules! with_distribution {
    ($rng:expr, $dist_spec:expr, $limit:expr, $callback:expr) => {
        match $dist_spec {
            DistributionSpec::Uniform { low, high } => {
                let weight_gen = RandWeightGen::uniform($rng, $limit, low, high)?;
                ($callback)(weight_gen)
            }
            DistributionSpec::UniformInclusive { low, high } => {
                let weight_gen = RandWeightGen::uniform_inclusive($rng, $limit, low, high)?;
                ($callback)(weight_gen)
            }
            DistributionSpec::XavierUniform { fan_in, fan_out } => {
                let weight_gen = RandWeightGen::xavier_uniform($rng, $limit, fan_in, fan_out)?;
                ($callback)(weight_gen)
            }
            DistributionSpec::LecunUniform { fan_in } => {
                let weight_gen = RandWeightGen::lecun_uniform($rng, $limit, fan_in)?;
                ($callback)(weight_gen)
            }
            DistributionSpec::Normal { mean, std_dev } => {
                let weight_gen = RandWeightGen::normal($rng, $limit, mean, std_dev)?;
                ($callback)(weight_gen)
            }
            DistributionSpec::Kaiming { fan_in } => {
                let weight_gen = RandWeightGen::kaiming($rng, $limit, fan_in)?;
                ($callback)(weight_gen)
            }
            DistributionSpec::Xavier { fan_in, fan_out } => {
                let weight_gen = RandWeightGen::xavier($rng, $limit, fan_in, fan_out)?;
                ($callback)(weight_gen)
            }
            DistributionSpec::Lecun { fan_in } => {
                let weight_gen = RandWeightGen::lecun($rng, $limit, fan_in)?;
                ($callback)(weight_gen)
            }
        }
    };
}

/// Builds `Server`s given a specification.
#[derive(Default)]
pub struct ServerBuilder {}

impl ServerBuilder {
    /// Creates a new `ServerBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a new `Server` following a spec.
    ///
    /// # Arguments
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandWeightGen` construction values.
    pub fn build<R, W>(&self, spec: ServerSpec) -> Result<Box<dyn Server<R, W>>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        self.resolve_weight_gen(spec)
    }

    /// Generates a random number generator given (or not) a seed.
    ///
    /// # Arguments
    /// * `seed` - An optional seed for the rng.
    ///
    /// # Returns
    /// An clonable random number generator with interior mutability.
    fn generate_rng(&self, seed: Option<u64>) -> Rc<RefCell<StdRng>> {
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        Rc::new(RefCell::new(rng))
    }

    /// Resolves the `WeightGen` for this server.
    ///
    /// # Arguments
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandWeightGen` construction values.
    fn resolve_weight_gen<R, W>(&self, spec: ServerSpec) -> Result<Box<dyn Server<R, W>>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        match spec.weight_gen {
            WeightGenSpec::Const { value, limit } => {
                let weight_gen = ConstWeightGen::new(value, limit);
                Ok(self.resolve_optimizer(spec, weight_gen))
            }
            WeightGenSpec::Rand {
                distribution,
                limit,
            } => {
                let rng = self.generate_rng(spec.seed);
                with_distribution!(rng, distribution, limit, |weight_gen| {
                    Ok(self.resolve_optimizer(spec, weight_gen))
                })
            }
            WeightGenSpec::Chained { ref specs } => {
                let rng = self.generate_rng(spec.seed);
                let weight_gen = self.resolve_chained(rng, specs)?;
                Ok(self.resolve_optimizer(spec, weight_gen))
            }
        }
    }

    /// Resolves the `ChainedWeightGen` weight generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `specs` - A list of weight gen specifications.
    ///
    /// # Returns
    /// A resolved `ChainedWeightGen`.
    fn resolve_chained<R>(
        &self,
        rng: Rc<RefCell<R>>,
        specs: &[WeightGenSpec],
    ) -> Result<ChainedWeightGen>
    where
        R: Rng + 'static,
    {
        let mut weight_gens: Vec<Box<dyn WeightGen>> = Vec::with_capacity(specs.len());

        for spec in specs {
            match spec {
                WeightGenSpec::Const { value, limit } => {
                    let weight_gen = ConstWeightGen::new(*value, *limit);
                    weight_gens.push(Box::new(weight_gen));
                }
                WeightGenSpec::Rand {
                    distribution,
                    limit,
                } => {
                    with_distribution!(rng.clone(), *distribution, *limit, |weight_gen| {
                        weight_gens.push(Box::new(weight_gen))
                    });
                }
                WeightGenSpec::Chained { specs } => {
                    let weight_gen = self.resolve_chained(rng.clone(), specs)?;
                    weight_gens.push(Box::new(weight_gen))
                }
            }
        }

        Ok(ChainedWeightGen::new(weight_gens))
    }

    /// Resolves the `Optimizer` for this server.
    ///
    /// # Arguments
    /// * `spec` - The specification for the parameter server.
    /// * `weight_gen` - A resolved weight generator.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandWeightGen` construction values.
    fn resolve_optimizer<R, W, WG>(&self, spec: ServerSpec, weight_gen: WG) -> Box<dyn Server<R, W>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        WG: WeightGen,
    {
        match spec.optimizer {
            OptimizerSpec::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                let factory = |len| Adam::new(len, learning_rate, beta1, beta2, epsilon);
                self.resolve_trainer(spec, weight_gen, factory)
            }
            OptimizerSpec::GradientDescent { learning_rate } => {
                let factory = |_| GradientDescent::new(learning_rate);
                self.resolve_trainer(spec, weight_gen, factory)
            }
            OptimizerSpec::GradientDescentWithMomentum {
                learning_rate,
                momentum,
            } => {
                let factory = |len| GradientDescentWithMomentum::new(len, learning_rate, momentum);
                self.resolve_trainer(spec, weight_gen, factory)
            }
        }
    }

    /// Resolves the `Trainer` for this server.
    ///
    /// # Arguments
    /// * `spec` - The specification for the parameter server.
    /// * `weight_gen` - A resolved weight generator.
    /// * `optimizer_factory` - A factory of optimizers.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandWeightGen` construction values.
    fn resolve_trainer<R, W, WG, O, OF>(
        &self,
        spec: ServerSpec,
        weight_gen: WG,
        optimizer_factory: OF,
    ) -> Box<dyn Server<R, W>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        WG: WeightGen,
        O: Optimizer + Send + 'static,
        OF: FnMut(usize) -> O,
    {
        match spec.trainer {
            TrainerSpec::BarrierSync { barrier_size } => {
                let trainer = BarrierSyncTrainer::new(barrier_size);
                self.terminate_build(spec, weight_gen, optimizer_factory, trainer)
            }
            TrainerSpec::NonBlocking => {
                let trainer = NonBlockingTrainer::new();
                self.terminate_build(spec, weight_gen, optimizer_factory, trainer)
            }
        }
    }

    /// Terminates the entire build for this session and finally instanciates all the entities.
    ///
    /// # Arguments
    /// * `spec` - The specification for the parameter server.
    /// * `weight_gen` - A resolved weight generator.
    /// * `optimizer_factory` - A factory of optimzers.
    /// * `trainer_factory` - A factory of a trainer.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandWeightGen` construction values.
    fn terminate_build<R, W, WG, O, OF, T>(
        &self,
        spec: ServerSpec,
        weight_gen: WG,
        optimizer_factory: OF,
        trainer: T,
    ) -> Box<dyn Server<R, W>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        WG: WeightGen,
        O: Optimizer + Send + 'static,
        OF: FnMut(usize) -> O,
        T: Trainer + Send + Sync + 'static,
    {
        let store = ParameterStore::new(spec.shard_size, weight_gen, optimizer_factory);
        let pserver = ParameterServer::new(store, trainer);
        Box::new(pserver)
    }
}
