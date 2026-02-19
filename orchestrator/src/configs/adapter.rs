use std::{
    io,
    net::{SocketAddr, ToSocketAddrs},
};

use comms::specs::{
    machine_learning::{
        ActFnSpec, DatasetSpec, LayerSpec, LossFnSpec, ModelSpec, OptimizerSpec, TrainerSpec,
    },
    server::{DistributionSpec, ParamGenSpec, ServerSpec, StoreSpec, SynchronizerSpec},
    worker::{AlgorithmSpec, WorkerSpec},
};

use super::{ModelConfig, TrainingConfig};
use crate::configs::{
    ActFnConfig, AlgorithmConfig, DatasetConfig, LayerConfig, LossFnConfig, OptimizerConfig,
    ParamGenConfig, StoreConfig, SynchronizerConfig,
};

/// It adapts the `ModelConfig` and `TrainingConfig` into the respective `WorkerSpec` and `ServerSpec`.
pub struct Adapter;

impl Adapter {
    /// Creates a new `Adapter`.
    ///
    /// # Returns
    /// A new `Adapter` instance.
    pub fn new() -> Self {
        Self
    }

    /// Adapts the model and training configs into the entities' specifications.
    ///
    /// # Arguments
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    ///
    /// # Returns
    /// The worker's configuration and network address and an optional network address and server
    /// specification pair if the algorithm requires it. If there's an error resolving the given
    /// addresses it will return an io error.
    pub fn adapt_configs<A: ToSocketAddrs>(
        &self,
        model: ModelConfig,
        training: TrainingConfig<A>,
    ) -> io::Result<(Vec<(SocketAddr, WorkerSpec)>, Vec<(SocketAddr, ServerSpec)>)> {
        let workers = self.adapt_workers(&model, &training)?;
        let servers = self.adapt_servers(&model, &training)?;
        Ok((workers, servers))
    }

    /// Adapts the configurations to `WorkerSpec`.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    ///
    /// # Returns
    /// The worker's specification or an io error if occurred.
    fn adapt_workers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> io::Result<Vec<(SocketAddr, WorkerSpec)>> {
        let algorithm = self.adapt_algorithm(&training.algorithm)?;
        let trainer = self.adapt_trainer(model, training)?;

        let workers = training
            .worker_addrs
            .iter()
            .enumerate()
            .map(|(i, addressable)| {
                let addr = addressable
                    .to_socket_addrs()?
                    .next()
                    .ok_or_else(|| io::Error::other("failed to resolve worker address"))?;

                let worker = WorkerSpec {
                    worker_id: i,
                    max_epochs: training.max_epochs,
                    trainer: trainer.clone(),
                    algorithm: algorithm.clone(),
                };

                Ok((addr, worker))
            })
            .collect::<io::Result<Vec<_>>>()?;

        Ok(workers)
    }

    /// Adapts the configurations to `ServerSpec`.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    ///
    /// # Returns
    /// The server's specification or an io error if occurred.
    fn adapt_servers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> io::Result<Vec<(SocketAddr, ServerSpec)>> {
        let AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            store,
            ..
        } = &training.algorithm;

        let (_, param_gen) = self.adapt_model_param_gen(model);

        let servers = server_addrs
            .iter()
            .enumerate()
            .map(|(i, addressable)| {
                let addr = addressable
                    .to_socket_addrs()?
                    .next()
                    .ok_or_else(|| io::Error::other("failed to resolve server address"))?;

                let server = ServerSpec {
                    id: i,
                    nworkers: training.worker_addrs.len(),
                    param_gen: param_gen.clone(),
                    optimizer: self.adapt_optimizer(training.optimizer),
                    synchronizer: self.adapt_synchronizer(synchronizer),
                    store: self.adapt_store(store),
                    seed: training.seed,
                };

                Ok((addr, server))
            })
            .collect::<io::Result<Vec<_>>>()?;

        Ok(servers)
    }

    /// Adapts the configurations to `AlgorithmSpec`.
    ///
    /// # Args
    /// * `algorithm` - The algorithm's configuration.
    ///
    /// # Returns
    /// The algorithm's specification or an io error if occurred.
    fn adapt_algorithm<A: ToSocketAddrs>(
        &self,
        algorithm: &AlgorithmConfig<A>,
    ) -> io::Result<AlgorithmSpec> {
        let spec = match algorithm {
            AlgorithmConfig::ParameterServer {
                server_addrs,
                server_ordering,
                ..
            } => {
                let resolved = server_addrs
                    .iter()
                    .map(|addr| {
                        addr.to_socket_addrs()?.next().ok_or_else(|| {
                            io::Error::other("failed to resolve the address for a server")
                        })
                    })
                    .collect::<io::Result<Vec<_>>>()?;

                AlgorithmSpec::ParameterServer {
                    addrs: resolved,
                    ordering: server_ordering.to_vec(),
                }
            }
        };

        Ok(spec)
    }

    /// Adapts the configuration to `SynchronizerSpec`.
    ///
    /// # Args
    /// * `synchronizer` - The synchronizer's configuration.
    ///
    /// # Returns
    /// The synchronizer's specification or an io error if occurred.
    fn adapt_synchronizer(&self, synchronizer: &SynchronizerConfig) -> SynchronizerSpec {
        match *synchronizer {
            SynchronizerConfig::Barrier { barrier_size } => {
                SynchronizerSpec::Barrier { barrier_size }
            }
            SynchronizerConfig::NonBlocking => SynchronizerSpec::NonBlocking,
        }
    }

    /// Adapts the configuration to `StoreSpec`.
    ///
    /// # Args
    /// * `store` - The store's configuration.
    ///
    /// # Returns
    /// The store's specification or an io error if occurred.
    fn adapt_store(&self, store: &StoreConfig) -> StoreSpec {
        match *store {
            StoreConfig::Blocking { shard_size } => StoreSpec::Blocking { shard_size },
            StoreConfig::Wild { shard_size } => StoreSpec::Wild { shard_size },
        }
    }

    /// Adapts the configuration to `TrainerSpec`.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    ///
    /// # Returns
    /// The trainer's specification or an io error if occurred.
    fn adapt_trainer<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> io::Result<TrainerSpec> {
        let (model_spec, _) = self.adapt_model_param_gen(model);
        let optimizer = self.adapt_optimizer(training.optimizer);
        let dataset = self.adapt_dataset(&training.dataset)?;
        let loss_fn = self.adapt_loss_fn(training.loss_fn);

        let trainer = TrainerSpec {
            model: model_spec,
            optimizer,
            dataset,
            loss_fn,
            offline_epochs: training.offline_epochs,
            batch_size: training.batch_size,
            seed: training.seed,
        };

        Ok(trainer)
    }

    /// Adapts the configuration to `LossFnSpec`.
    ///
    /// # Args
    /// * `loss_fn` - The loss function's configuration.
    ///
    /// # Returns
    /// The loss function's specification or an io error if occurred.
    fn adapt_loss_fn(&self, loss_fn: LossFnConfig) -> LossFnSpec {
        match loss_fn {
            LossFnConfig::Mse => LossFnSpec::Mse,
        }
    }

    /// Adapts the configuration to `DatasetSpec`.
    ///
    /// # Args
    /// * `dataset` - The dataset's configuration.
    ///
    /// # Returns
    /// The dataset's specification or an io error if occurred.
    fn adapt_dataset(&self, dataset: &DatasetConfig) -> io::Result<DatasetSpec> {
        let dataset_spec = match dataset {
            DatasetConfig::Local { path: _path } => todo!(),
            &DatasetConfig::Inline {
                ref data,
                x_size,
                y_size,
            } => DatasetSpec {
                data: data.to_vec(),
                x_size,
                y_size,
            },
        };

        Ok(dataset_spec)
    }

    /// Adapts the configuration to `OptimizerSpec`.
    ///
    /// # Args
    /// * `optimizer` - The optimizer's configuration.
    ///
    /// # Returns
    /// The optimizer's specification or an io error if occurred.
    fn adapt_optimizer(&self, optimizer: OptimizerConfig) -> OptimizerSpec {
        match optimizer {
            OptimizerConfig::Adam { lr, b1, b2, eps } => OptimizerSpec::Adam {
                learning_rate: lr,
                beta1: b1,
                beta2: b2,
                epsilon: eps,
            },
            OptimizerConfig::GradientDescent { lr } => {
                OptimizerSpec::GradientDescent { learning_rate: lr }
            }
            OptimizerConfig::GradientDescentWithMomentum { lr, mu } => {
                OptimizerSpec::GradientDescentWithMomentum {
                    learning_rate: lr,
                    momentum: mu,
                }
            }
        }
    }

    /// Adapts the configuration to `ModelSpec` and `ParamGenSpec`.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    ///
    /// # Returns
    /// The model's and parameter generator's specification or an io error if occurred.
    fn adapt_model_param_gen(&self, model: &ModelConfig) -> (ModelSpec, ParamGenSpec) {
        match model {
            ModelConfig::Sequential { layers } => {
                let (layer_specs, param_gen_specs): (Vec<_>, Vec<_>) =
                    layers.iter().map(|layer| self.adapt_layer(layer)).unzip();

                let model_spec = ModelSpec::Sequential {
                    layers: layer_specs,
                };

                let param_gen_spec = ParamGenSpec::Chained {
                    specs: param_gen_specs,
                };

                (model_spec, param_gen_spec)
            }
        }
    }

    /// Adapts the configuration to `ParamGenSpec`.
    ///
    /// # Args
    /// * `param_gen` - The parameter generator's configuration.
    ///
    /// # Returns
    /// The parameter generator's specification or an io error if occurred.
    fn adapt_param_gen(
        &self,
        param_gen: ParamGenConfig,
        (fan_in, limit, fan_out): (usize, usize, usize),
    ) -> ParamGenSpec {
        match param_gen {
            ParamGenConfig::Const { value } => ParamGenSpec::Const { value, limit },
            ParamGenConfig::Uniform { low, high } => ParamGenSpec::Rand {
                distribution: DistributionSpec::Uniform { low, high },
                limit,
            },
            ParamGenConfig::UniformInclusive { low, high } => ParamGenSpec::Rand {
                distribution: DistributionSpec::UniformInclusive { low, high },
                limit,
            },
            ParamGenConfig::XavierUniform {} => ParamGenSpec::Rand {
                distribution: DistributionSpec::XavierUniform { fan_in, fan_out },
                limit,
            },
            ParamGenConfig::LecunUniform => ParamGenSpec::Rand {
                distribution: DistributionSpec::LecunUniform { fan_in },
                limit,
            },
            ParamGenConfig::Normal { mean, std_dev } => ParamGenSpec::Rand {
                distribution: DistributionSpec::Normal { mean, std_dev },
                limit,
            },
            ParamGenConfig::Kaiming => ParamGenSpec::Rand {
                distribution: DistributionSpec::Kaiming { fan_in },
                limit,
            },
            ParamGenConfig::Xavier => ParamGenSpec::Rand {
                distribution: DistributionSpec::Xavier { fan_in, fan_out },
                limit,
            },
            ParamGenConfig::Lecun => ParamGenSpec::Rand {
                distribution: DistributionSpec::Lecun { fan_in },
                limit,
            },
        }
    }

    /// Adapts the configuration to `LayerSpec`.
    ///
    /// # Args
    /// * `layer` - The layer's configuration.
    ///
    /// # Returns
    /// The layer's specification or an io error if occurred.
    fn adapt_layer(&self, layer: &LayerConfig) -> (LayerSpec, ParamGenSpec) {
        match *layer {
            LayerConfig::Dense {
                dim: (n, m),
                init,
                act_fn,
            } => {
                let act_fn = self.adapt_act_fn(act_fn.as_ref());
                (
                    LayerSpec::Dense {
                        dim: (n, m),
                        act_fn,
                    },
                    self.adapt_param_gen(init, layer.sizes()),
                )
            }
        }
    }

    /// Adapts the configuration to `ActFnSpec`.
    ///
    /// # Args
    /// * `act_fn` - The layer's configuration.
    ///
    /// # Returns
    /// The activation function's specification or an io error if occurred.
    fn adapt_act_fn(&self, act_fn: Option<&ActFnConfig>) -> Option<ActFnSpec> {
        let act_fn_spec = match *act_fn? {
            ActFnConfig::Sigmoid { amp } => ActFnSpec::Sigmoid { amp },
        };

        Some(act_fn_spec)
    }
}
