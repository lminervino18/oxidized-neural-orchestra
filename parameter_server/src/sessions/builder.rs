use std::{cell::RefCell, error::Error, num::NonZeroUsize, rc::Rc};

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    parameters::{
        ParameterStore,
        optimization::{Adam, GradientDescent, GradientDescentWithMomentum, Optimizer},
        weight_gen::{ChainedWeightGen, ConstWeightGen, RandWeightGen, WeightGen},
    },
    server::ParameterServer,
    sessions::{
        Session,
        session::TrainingSession,
        specs::{DistributionSpec, OptimizerSpec, ParameterServerSpec, TrainerSpec, WeightGenSpec},
    },
    training::{BarrierSyncTrainer, NonBlockingTrainer, Trainer},
};

/// Makes `callback`'s return type genric, when trying to resolve a concrete `RandWeightGen` it avoids boxing all
/// weight generators variants.
///
/// When trying to resolve for a `ChainedWeightGen`, a sub `RandWeightGen` must be returned as a boxed
/// `WeightGen`, so it must duplicate the distribution's static dispatch match.
///
/// To avoid this duplication, this macro generalizes what it's immediately done with the concrete weight generator,
/// and thus avoids boxing weight generators unnecessarily.
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

/// Builds new `Session`s given a specification.
pub struct SessionBuilder {
    built_count: usize,
}

impl SessionBuilder {
    /// Creates a new `SessionBuilder`.
    pub fn new() -> Self {
        Self { built_count: 0 }
    }

    /// Builds a parameter server `Session` following a spec.
    ///
    /// # Args
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new session or an error if encountered.
    pub fn build(&mut self, spec: ParameterServerSpec) -> Result<Box<dyn Session>, Box<dyn Error>> {
        self.resolve_weight_gen(spec)
    }

    /// Generates a random number generator given (or not) a seed.
    ///
    /// # Args
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

    /// Generates a new incremental id for the session being created.
    ///
    /// # Returns
    /// A new id.
    fn generate_id(&mut self) -> usize {
        self.built_count += 1;
        self.built_count - 1
    }

    /// Resolves the `WeightGen` for this session.
    ///
    /// # Args
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new session or an error if encountered.
    fn resolve_weight_gen(
        &mut self,
        spec: ParameterServerSpec,
    ) -> Result<Box<dyn Session>, Box<dyn Error>> {
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
                let weight_gens = self.resolve_chained(rng, specs)?;
                let weight_gen = ChainedWeightGen::new(weight_gens);
                Ok(self.resolve_optimizer(spec, weight_gen))
            }
        }
    }

    /// Resolves the `ChainedWeightGen` weight generator.
    ///
    /// # Args
    /// * `rng` - A random number generator.
    /// * `specs` - A list of weight gen specifications.
    ///
    /// # Returns
    /// A list of dynamic weight generators or an error if encountered.
    fn resolve_chained<R>(
        &self,
        rng: Rc<RefCell<R>>,
        specs: &[WeightGenSpec],
    ) -> Result<Vec<Box<dyn WeightGen>>, Box<dyn Error>>
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
                    let sub_weight_gens = self.resolve_chained(rng.clone(), specs)?;
                    let weight_gen = ChainedWeightGen::new(sub_weight_gens);
                    weight_gens.push(Box::new(weight_gen))
                }
            }
        }

        Ok(weight_gens)
    }

    /// Resolves the `Optimizer` for this session.
    ///
    /// # Args
    /// * `spec` - The specification for the parameter server.
    /// * `weight_gen` - A resolved weight generator.
    ///
    /// # Returns
    /// A new session.
    fn resolve_optimizer<W>(&mut self, spec: ParameterServerSpec, weight_gen: W) -> Box<dyn Session>
    where
        W: WeightGen,
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

    /// Resolves the `Trainer` for this session.
    ///
    /// # Args
    /// * `spec` - The specification for the parameter server.
    /// * `weight_gen` - A resolved weight generator.
    /// * `optimizer_factory` - A factory of optimizers.
    ///
    /// # Returns
    /// A new session.
    fn resolve_trainer<W, O, OF>(
        &mut self,
        spec: ParameterServerSpec,
        weight_gen: W,
        optimizer_factory: OF,
    ) -> Box<dyn Session>
    where
        W: WeightGen,
        O: Optimizer + Send + 'static,
        OF: FnMut(usize) -> O,
    {
        match spec.trainer {
            TrainerSpec::BarrierSync { barrier_size } => {
                let factory = |store| BarrierSyncTrainer::new(barrier_size, store);
                self.terminate_build(spec, weight_gen, optimizer_factory, factory)
            }
            TrainerSpec::NonBlocking => {
                let factory = NonBlockingTrainer::new;
                self.terminate_build(spec, weight_gen, optimizer_factory, factory)
            }
        }
    }

    /// Terminates the entire build for this session and finally instanciates all the entities.
    ///
    /// # Args
    /// * `spec` - The specification for the parameter server.
    /// * `weight_gen` - A resolved weight generator.
    /// * `optimizer_factory` - A factory of optimzers.
    /// * `trainer_factory` - A factory of a trainer.
    ///
    /// # Returns
    /// A new session.
    fn terminate_build<W, O, OF, T, TF>(
        &mut self,
        spec: ParameterServerSpec,
        weight_gen: W,
        optimizer_factory: OF,
        trainer_factory: TF,
    ) -> Box<dyn Session>
    where
        W: WeightGen,
        O: Optimizer,
        OF: FnMut(usize) -> O,
        T: Trainer + 'static,
        TF: FnOnce(ParameterStore<O>) -> T,
    {
        let ParameterServerSpec {
            params,
            shard_amount,
            epochs,
            ..
        } = spec;

        let shard_amount = shard_amount.unwrap_or_else(|| {
            let div = NonZeroUsize::new(10_000).unwrap();
            params.div_ceil(div)
        });

        let store = ParameterStore::new(params.get(), shard_amount, weight_gen, optimizer_factory);
        let trainer = trainer_factory(store);
        let pserver = ParameterServer::new(params.get(), epochs, trainer);
        let session = TrainingSession::new(self.generate_id(), pserver);
        Box::new(session)
    }
}
