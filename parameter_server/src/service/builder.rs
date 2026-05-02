use std::{io, num::NonZeroUsize, thread};

use comms::{
    Acceptor, Connection, TransportLayer,
    specs::{
        machine_learning::OptimizerSpec,
        server::{ServerSpec, StoreSpec, SynchronizerSpec},
    },
};
use machine_learning::{
    initialization::{ParamGenBuilder, Result},
    optimization::{Adam, GradientDescent, GradientDescentWithMomentum, Optimizer},
};

use super::{ParameterServer, Server};
use crate::{
    storage::{BlockingStore, Store, StoreHandle, WildStore},
    synchronization::{BarrierSync, NoBlockingSync, Synchronizer},
};

/// The amount of cores to use if `std::thread::available_parallelism` fails.
const DEFAULT_CORE_COUNT: NonZeroUsize = NonZeroUsize::new(8).unwrap();

/// The factor to multiply the amount of cores to obtain the shard amount.
const SHARD_AMOUNT_FACTOR: NonZeroUsize = NonZeroUsize::new(2).unwrap();

/// Builds `Server`s given a specification.
pub struct ServerBuilder<'a, T, F>
where
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
{
    acceptor: &'a mut Acceptor<T, F>,
}

impl<'a, T, F> ServerBuilder<'a, T, F>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
{
    /// Creates a new `ServerBuilder`.
    ///
    /// # Returns
    /// A new `ServerBuilder` instance.
    pub fn new(acceptor: &'a mut Acceptor<T, F>) -> Self {
        Self { acceptor }
    }

    /// Builds a new `Server` following a spec.
    ///
    /// # Args
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandParamGen` construction values.
    pub async fn build(&mut self, spec: ServerSpec) -> io::Result<Box<dyn Server<T>>> {
        let nworkers = spec.nworkers;
        let mut server = self.resolve_optimizer(spec).map_err(io::Error::other)?;

        for _ in 0..nworkers {
            let Connection::Worker(worker_handle) = self.acceptor.accept().await? else {
                return Err(io::Error::other("Unexpected non worker connection"));
            };

            server.spawn(worker_handle);
        }

        Ok(server)
    }

    /// Resolves the `Optimizer` for this server.
    ///
    /// # Args
    /// * `spec` - The specification for the parameter server.
    /// * `param_gen` - A resolved parameter generator.
    ///
    /// # Returns
    /// A new server.
    fn resolve_optimizer(&self, spec: ServerSpec) -> Result<Box<dyn Server<T>>> {
        match spec.optimizer {
            OptimizerSpec::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                let factory = |len| Adam::new(len, learning_rate, beta1, beta2, epsilon);
                self.resolve_store(spec, factory)
            }
            OptimizerSpec::GradientDescent { learning_rate } => {
                let factory = |_| GradientDescent::new(learning_rate);
                self.resolve_store(spec, factory)
            }
            OptimizerSpec::GradientDescentWithMomentum {
                learning_rate,
                momentum,
            } => {
                let factory = |len| GradientDescentWithMomentum::new(len, learning_rate, momentum);
                self.resolve_store(spec, factory)
            }
        }
    }

    /// Resolves the `Store` for this server.
    ///
    /// # Args
    /// * `spec` - The specification for the parameter server.
    /// * `param_gen` - A resolved parameter generator.
    /// * `optimizer_factory` - A factory of optimizers.
    /// * `synchronizer` - A resolved synchronizer.
    ///
    /// # Returns
    /// A new server.
    fn resolve_store<O, OF>(
        &self,
        spec: ServerSpec,
        optimizer_factory: OF,
    ) -> Result<Box<dyn Server<T>>>
    where
        O: Optimizer + Send + 'static,
        OF: Fn(usize) -> O,
    {
        let param_gen_builder = ParamGenBuilder::new();
        let mut param_gen = param_gen_builder.build(spec.param_gen.clone(), spec.seed)?;

        // SAFETY: The argument is at least 1.
        let nparams = unsafe { NonZeroUsize::new_unchecked(param_gen.size().max(1)) };
        let cores = thread::available_parallelism().unwrap_or(DEFAULT_CORE_COUNT);
        let max_shard_amount = cores.saturating_mul(SHARD_AMOUNT_FACTOR);
        let shard_amount = nparams.min(max_shard_amount);
        let shard_size = NonZeroUsize::new(shard_amount.get().div_ceil(nparams.get())).unwrap();

        match spec.store {
            StoreSpec::Blocking => {
                let store = BlockingStore::new(shard_size, param_gen.as_mut(), optimizer_factory);
                Ok(self.resolve_synchronizer(spec, store))
            }
            StoreSpec::Wild => {
                let store = WildStore::new(shard_size, param_gen.as_mut(), optimizer_factory);
                Ok(self.resolve_synchronizer(spec, store))
            }
        }
    }

    /// Resolves the `Synchronizer` for this server.
    ///
    /// # Args
    /// * `spec` - The specification for the parameter server.
    /// * `param_gen` - A resolved parameter generator.
    /// * `optimizer_factory` - A factory of optimizers.
    ///
    /// # Returns
    /// A new server.
    fn resolve_synchronizer<PS>(&self, spec: ServerSpec, store: PS) -> Box<dyn Server<T>>
    where
        PS: Store + Send + Sync + 'static,
    {
        match spec.synchronizer {
            SynchronizerSpec::Barrier { barrier_size } => {
                let synchronizer = BarrierSync::new(barrier_size);
                self.terminate_build(store, synchronizer)
            }
            SynchronizerSpec::NonBlocking => {
                let synchronizer = NoBlockingSync::new();
                self.terminate_build(store, synchronizer)
            }
        }
    }

    /// Terminates the entire build for this session and finally instanciates all the entities.
    ///
    /// # Args
    /// * `store` - A resolved store.
    /// * `synchronizer` - A resolved synchronizer.
    ///
    /// # Returns
    /// A new server.
    fn terminate_build<PS, Sy>(&self, store: PS, synchronizer: Sy) -> Box<dyn Server<T>>
    where
        PS: Store + Send + Sync + 'static,
        Sy: Synchronizer + Send + Sync + 'static,
    {
        let handle = StoreHandle::new(store);
        let pserver = ParameterServer::new(handle, synchronizer);
        Box::new(pserver)
    }
}
