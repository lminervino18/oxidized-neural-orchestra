use std::{cell::RefCell, rc::Rc};

use comms::specs::{
    machine_learning::OptimizerSpec,
    server::{DistributionSpec, ParamGenSpec, ServerSpec, StoreSpec, SynchronizerSpec},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokio::io::{AsyncRead, AsyncWrite};

use super::{ParameterServer, Server};
use crate::{
    initialization::{ChainedParamGen, ConstParamGen, ParamGen, RandParamGen, Result},
    optimization::{Adam, GradientDescent, GradientDescentWithMomentum, Optimizer},
    storage::{BlockingStore, Store, StoreHandle, WildStore},
    synchronization::{BarrierSync, NoBlockingSync, Synchronizer},
};

/// Makes `callback`'s return type generic, when trying to resolve a concrete `RandParamGen` it avoids boxing all
/// parameter generators variants.
///
/// When trying to resolve for a `ChainedParamGen`, a sub `RandParamGen` must be returned as a boxed
/// `ParamGen`, so it must duplicate the distribution's static dispatch match.
///
/// To avoid this duplication, this macro generalizes what it's immediately done with the concrete parameter generator,
/// and thus avoids boxing parameter generators unnecessarily.
///
/// # Arguments
/// * `rng` - A random number generator.
/// * `dist_spec` - A specification for a distribution.
/// * `limit` - The limit of parameters that the `RandParamGen` can generate.
/// * `callback` - The closure to call passing in the created weigth gen.
macro_rules! with_distribution {
    ($rng:expr, $dist_spec:expr, $limit:expr, $callback:expr) => {
        match $dist_spec {
            DistributionSpec::Uniform { low, high } => {
                let param_gen = RandParamGen::uniform($rng, $limit, low, high)?;
                ($callback)(param_gen)
            }
            DistributionSpec::UniformInclusive { low, high } => {
                let param_gen = RandParamGen::uniform_inclusive($rng, $limit, low, high)?;
                ($callback)(param_gen)
            }
            DistributionSpec::XavierUniform { fan_in, fan_out } => {
                let param_gen = RandParamGen::xavier_uniform($rng, $limit, fan_in, fan_out)?;
                ($callback)(param_gen)
            }
            DistributionSpec::LecunUniform { fan_in } => {
                let param_gen = RandParamGen::lecun_uniform($rng, $limit, fan_in)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Normal { mean, std_dev } => {
                let param_gen = RandParamGen::normal($rng, $limit, mean, std_dev)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Kaiming { fan_in } => {
                let param_gen = RandParamGen::kaiming($rng, $limit, fan_in)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Xavier { fan_in, fan_out } => {
                let param_gen = RandParamGen::xavier($rng, $limit, fan_in, fan_out)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Lecun { fan_in } => {
                let param_gen = RandParamGen::lecun($rng, $limit, fan_in)?;
                ($callback)(param_gen)
            }
        }
    };
}

/// Builds `Server`s given a specification.
pub struct ServerBuilder;

impl ServerBuilder {
    /// Creates a new `ServerBuilder`.
    ///
    /// # Returns
    /// A new `ServerBuilder` instance.
    pub fn new() -> Self {
        Self
    }

    /// Builds a new `Server` following a spec.
    ///
    /// # Arguments
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandParamGen` construction values.
    pub fn build<R, W>(&self, spec: ServerSpec) -> Result<Box<dyn Server<R, W>>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        self.resolve_param_gen(spec)
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

    /// Resolves the `ParamGen` for this server.
    ///
    /// # Arguments
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandParamGen` construction values.
    fn resolve_param_gen<R, W>(&self, spec: ServerSpec) -> Result<Box<dyn Server<R, W>>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        match spec.param_gen {
            ParamGenSpec::Const { value, limit } => {
                let param_gen = ConstParamGen::new(value, limit);
                Ok(self.resolve_optimizer(spec, param_gen))
            }
            ParamGenSpec::Rand {
                distribution,
                limit,
            } => {
                let rng = self.generate_rng(spec.seed);
                with_distribution!(rng, distribution, limit, |param_gen| {
                    Ok(self.resolve_optimizer(spec, param_gen))
                })
            }
            ParamGenSpec::Chained { ref specs } => {
                let rng = self.generate_rng(spec.seed);
                let param_gen = self.resolve_chained(rng, specs)?;
                Ok(self.resolve_optimizer(spec, param_gen))
            }
        }
    }

    /// Resolves the `ChainedParamGen` parameter generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `specs` - A list of parameter generator specifications.
    ///
    /// # Returns
    /// A resolved `ChainedParamGen`.
    fn resolve_chained<R>(
        &self,
        rng: Rc<RefCell<R>>,
        specs: &[ParamGenSpec],
    ) -> Result<ChainedParamGen>
    where
        R: Rng + 'static,
    {
        let mut param_gens: Vec<Box<dyn ParamGen>> = Vec::with_capacity(specs.len());

        for spec in specs {
            match spec {
                ParamGenSpec::Const { value, limit } => {
                    let param_gen = ConstParamGen::new(*value, *limit);
                    param_gens.push(Box::new(param_gen));
                }
                ParamGenSpec::Rand {
                    distribution,
                    limit,
                } => {
                    with_distribution!(rng.clone(), *distribution, *limit, |param_gen| {
                        param_gens.push(Box::new(param_gen))
                    });
                }
                ParamGenSpec::Chained { specs } => {
                    let param_gen = self.resolve_chained(rng.clone(), specs)?;
                    param_gens.push(Box::new(param_gen))
                }
            }
        }

        Ok(ChainedParamGen::new(param_gens))
    }

    /// Resolves the `Optimizer` for this server.
    ///
    /// # Arguments
    /// * `spec` - The specification for the parameter server.
    /// * `param_gen` - A resolved parameter generator.
    ///
    /// # Returns
    /// A new server.
    fn resolve_optimizer<R, W, PG>(&self, spec: ServerSpec, param_gen: PG) -> Box<dyn Server<R, W>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        PG: ParamGen,
    {
        match spec.optimizer {
            OptimizerSpec::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                let factory = |len| Adam::new(len, learning_rate, beta1, beta2, epsilon);
                self.resolve_synchronizer(spec, param_gen, factory)
            }
            OptimizerSpec::GradientDescent { learning_rate } => {
                let factory = |_| GradientDescent::new(learning_rate);
                self.resolve_synchronizer(spec, param_gen, factory)
            }
            OptimizerSpec::GradientDescentWithMomentum {
                learning_rate,
                momentum,
            } => {
                let factory = |len| GradientDescentWithMomentum::new(len, learning_rate, momentum);
                self.resolve_synchronizer(spec, param_gen, factory)
            }
        }
    }

    /// Resolves the `Synchronizer` for this server.
    ///
    /// # Arguments
    /// * `spec` - The specification for the parameter server.
    /// * `param_gen` - A resolved parameter generator.
    /// * `optimizer_factory` - A factory of optimizers.
    ///
    /// # Returns
    /// A new server.
    fn resolve_synchronizer<R, W, PG, O, OF>(
        &self,
        spec: ServerSpec,
        param_gen: PG,
        optimizer_factory: OF,
    ) -> Box<dyn Server<R, W>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        PG: ParamGen,
        O: Optimizer + Send + 'static,
        OF: FnMut(usize) -> O,
    {
        match spec.synchronizer {
            SynchronizerSpec::Barrier { barrier_size } => {
                let synchronizer = BarrierSync::new(barrier_size);
                self.resolve_store(spec, param_gen, optimizer_factory, synchronizer)
            }
            SynchronizerSpec::NonBlocking => {
                let synchronizer = NoBlockingSync::new();
                self.resolve_store(spec, param_gen, optimizer_factory, synchronizer)
            }
        }
    }

    /// Resolves the `Store` for this server.
    ///
    /// # Arguments
    /// * `spec` - The specification for the parameter server.
    /// * `param_gen` - A resolved parameter generator.
    /// * `optimizer_factory` - A factory of optimizers.
    /// * `synchronizer` - A resolved synchronizer.
    ///
    /// # Returns
    /// A new server.
    fn resolve_store<R, W, PG, O, OF, S>(
        &self,
        spec: ServerSpec,
        param_gen: PG,
        optimizer_factory: OF,
        synchronizer: S,
    ) -> Box<dyn Server<R, W>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        PG: ParamGen,
        O: Optimizer + Send + 'static,
        OF: FnMut(usize) -> O,
        S: Synchronizer + Send + Sync + 'static,
    {
        match spec.store {
            StoreSpec::Blocking { shard_size } => {
                let store = BlockingStore::new(shard_size, param_gen, optimizer_factory);
                self.terminate_build(store, synchronizer)
            }
            StoreSpec::Wild { shard_size } => {
                let store = WildStore::new(shard_size, param_gen, optimizer_factory);
                self.terminate_build(store, synchronizer)
            }
        }
    }

    /// Terminates the entire build for this session and finally instanciates all the entities.
    ///
    /// # Arguments
    /// * `store` - A resolved store.
    /// * `synchronizer` - A resolved synchronizer.
    ///
    /// # Returns
    /// A new server.
    fn terminate_build<R, W, PS, Sy>(&self, store: PS, synchronizer: Sy) -> Box<dyn Server<R, W>>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        PS: Store + Send + Sync + 'static,
        Sy: Synchronizer + Send + Sync + 'static,
    {
        let handle = StoreHandle::new(store);
        let pserver = ParameterServer::new(handle, synchronizer);
        Box::new(pserver)
    }
}
