use std::net::{SocketAddr, ToSocketAddrs};

use comms::specs::{
    machine_learning::{
        ActFnSpec, DatasetSpec, LayerSpec, LossFnSpec, ModelSpec, OptimizerSpec, TrainerSpec,
    },
    server::{DistributionSpec, ParamGenSpec, ServerSpec, StoreSpec, SynchronizerSpec},
    worker::{AlgorithmSpec, WorkerSpec},
};

use super::{ModelConfig, TrainingConfig};
use crate::{
    configs::{
        ActFnConfig, AlgorithmConfig, DatasetConfig, LayerConfig, LossFnConfig, OptimizerConfig,
        ParamGenConfig, StoreConfig, SynchronizerConfig,
    },
    error::{OrchestratorError, Result},
};

use super::validator::Validator;

/// Converts orchestrator configs into wire-level specs ready to be sent to workers and servers.
pub struct Adapter;

impl Adapter {
    /// Creates a new `Adapter` instance.
    pub fn new() -> Self {
        Self
    }

    /// Validates and adapts model and training configs into worker and server specs.
    ///
    /// # Args
    /// * `model` - The model architecture configuration.
    /// * `training` - The training configuration.
    ///
    /// # Returns
    /// A pair of worker spec lists and server spec lists with their resolved addresses.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if validation fails or any address cannot be resolved.
    pub fn adapt_configs<A: ToSocketAddrs>(
        &self,
        model: ModelConfig,
        training: TrainingConfig<A>,
    ) -> Result<(Vec<(SocketAddr, WorkerSpec)>, Vec<(SocketAddr, ServerSpec)>)> {
        Validator::new().validate(&model, &training)?;

        let workers = self.adapt_workers(&model, &training)?;
        let servers = self.adapt_servers(&model, &training)?;
        Ok((workers, servers))
    }

    // -------------------------------------------------------------------------
    // Adaptation
    // -------------------------------------------------------------------------

    /// Adapts worker addresses and model/training configs into `WorkerSpec`s.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any address cannot be resolved.
    fn adapt_workers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<Vec<(SocketAddr, WorkerSpec)>> {
        let algorithm = self.adapt_algorithm(&training.algorithm)?;
        let trainer = self.adapt_trainer(model, training)?;

        training
            .worker_addrs
            .iter()
            .enumerate()
            .map(|(i, addressable)| {
                let addr = addressable
                    .to_socket_addrs()
                    .map_err(|e| OrchestratorError::ConnectionFailed {
                        addr: format!("worker[{i}]"),
                        source: e,
                    })?
                    .next()
                    .ok_or_else(|| {
                        OrchestratorError::InvalidConfig(format!(
                            "worker[{i}]: could not resolve address"
                        ))
                    })?;

                let spec = WorkerSpec {
                    worker_id: i,
                    max_epochs: training.max_epochs,
                    trainer: trainer.clone(),
                    algorithm,
                };
                Ok((addr, spec))
            })
            .collect()
    }

    /// Adapts server addresses and model/training configs into `ServerSpec`s.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any address cannot be resolved.
    fn adapt_servers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<Vec<(SocketAddr, ServerSpec)>> {
        let AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            store,
        } = &training.algorithm;

        let (_, param_gen) = self.adapt_model_param_gen(model);

        server_addrs
            .iter()
            .enumerate()
            .map(|(i, addressable)| {
                let addr = addressable
                    .to_socket_addrs()
                    .map_err(|e| OrchestratorError::ConnectionFailed {
                        addr: format!("server[{i}]"),
                        source: e,
                    })?
                    .next()
                    .ok_or_else(|| {
                        OrchestratorError::InvalidConfig(format!(
                            "server[{i}]: could not resolve address"
                        ))
                    })?;

                let spec = ServerSpec {
                    id: i,
                    nworkers: training.worker_addrs.len(),
                    param_gen: param_gen.clone(),
                    optimizer: self.adapt_optimizer(training.optimizer),
                    synchronizer: self.adapt_synchronizer(synchronizer),
                    store: self.adapt_store(store),
                    seed: training.seed,
                };
                Ok((addr, spec))
            })
            .collect()
    }

    /// Resolves the primary server address for the algorithm spec.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if the address cannot be resolved.
    fn adapt_algorithm<A: ToSocketAddrs>(
        &self,
        algorithm: &AlgorithmConfig<A>,
    ) -> Result<AlgorithmSpec> {
        match algorithm {
            AlgorithmConfig::ParameterServer { server_addrs, .. } => {
                let server_addr = server_addrs[0]
                    .to_socket_addrs()
                    .map_err(|e| OrchestratorError::ConnectionFailed {
                        addr: server_addrs[0]
                            .to_socket_addrs()
                            .ok()
                            .and_then(|mut it| it.next())
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| "server[0]".into()),
                        source: e,
                    })?
                    .next()
                    .ok_or_else(|| {
                        OrchestratorError::InvalidConfig("no server addresses provided".into())
                    })?;

                Ok(AlgorithmSpec::ParameterServer { server_addr })
            }
        }
    }

    /// Converts a `SynchronizerConfig` into a `SynchronizerSpec`.
    fn adapt_synchronizer(&self, synchronizer: &SynchronizerConfig) -> SynchronizerSpec {
        match *synchronizer {
            SynchronizerConfig::Barrier { barrier_size } => {
                SynchronizerSpec::Barrier { barrier_size }
            }
            SynchronizerConfig::NonBlocking => SynchronizerSpec::NonBlocking,
        }
    }

    /// Converts a `StoreConfig` into a `StoreSpec`.
    fn adapt_store(&self, store: &StoreConfig) -> StoreSpec {
        match *store {
            StoreConfig::Blocking { shard_size } => StoreSpec::Blocking { shard_size },
            StoreConfig::Wild { shard_size } => StoreSpec::Wild { shard_size },
        }
    }

    /// Builds a `TrainerSpec` from the model and training configs.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if the dataset config is invalid.
    fn adapt_trainer<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<TrainerSpec> {
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

    /// Converts a `LossFnConfig` into a `LossFnSpec`.
    fn adapt_loss_fn(&self, loss_fn: LossFnConfig) -> LossFnSpec {
        match loss_fn {
            LossFnConfig::Mse => LossFnSpec::Mse,
        }
    }

    /// Converts a `DatasetConfig` into a `DatasetSpec`.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if the dataset variant is not yet supported.
    fn adapt_dataset(&self, dataset: &DatasetConfig) -> Result<DatasetSpec> {
        match dataset {
            DatasetConfig::Local { path } => Err(OrchestratorError::InvalidConfig(format!(
                "local dataset loading not yet implemented: {}",
                path.display()
            ))),
            DatasetConfig::Inline { data, x_size, y_size } => Ok(DatasetSpec {
                data: data.to_vec(),
                x_size: *x_size,
                y_size: *y_size,
            }),
        }
    }

    /// Converts an `OptimizerConfig` into an `OptimizerSpec`.
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

    /// Converts a `ModelConfig` into a `ModelSpec` and its associated `ParamGenSpec`.
    fn adapt_model_param_gen(&self, model: &ModelConfig) -> (ModelSpec, ParamGenSpec) {
        match model {
            ModelConfig::Sequential { layers } => {
                let (layer_specs, param_gen_specs): (Vec<_>, Vec<_>) =
                    layers.iter().map(|layer| self.adapt_layer(layer)).unzip();

                (
                    ModelSpec::Sequential { layers: layer_specs },
                    ParamGenSpec::Chained { specs: param_gen_specs },
                )
            }
        }
    }

    /// Converts a `ParamGenConfig` into a `ParamGenSpec` given the layer dimensions.
    ///
    /// # Args
    /// * `param_gen` - The parameter generator configuration.
    /// * `(fan_in, limit, fan_out)` - Layer sizing info used by distribution-based generators.
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

    /// Converts a `LayerConfig` into a `LayerSpec` and its associated `ParamGenSpec`.
    fn adapt_layer(&self, layer: &LayerConfig) -> (LayerSpec, ParamGenSpec) {
        match *layer {
            LayerConfig::Dense {
                dim: (n, m),
                init,
                act_fn,
            } => {
                let act_fn = self.adapt_act_fn(act_fn.as_ref());
                (
                    LayerSpec::Dense { dim: (n, m), act_fn },
                    self.adapt_param_gen(init, layer.sizes()),
                )
            }
        }
    }

    /// Converts an optional `ActFnConfig` reference into an optional `ActFnSpec`.
    fn adapt_act_fn(&self, act_fn: Option<&ActFnConfig>) -> Option<ActFnSpec> {
        Some(match *act_fn? {
            ActFnConfig::Sigmoid { amp } => ActFnSpec::Sigmoid { amp },
        })
    }
}