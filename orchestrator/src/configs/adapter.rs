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
    error::OrchestratorError,
};

pub struct Adapter;

impl Adapter {
    pub fn new() -> Self {
        Self
    }

    pub fn adapt_configs<A: ToSocketAddrs>(
        &self,
        model: ModelConfig,
        training: TrainingConfig<A>,
    ) -> Result<(Vec<(SocketAddr, WorkerSpec)>, Vec<(SocketAddr, ServerSpec)>), OrchestratorError>
    {
        let workers = self.adapt_workers(&model, &training)?;
        let servers = self.adapt_servers(&model, &training)?;
        Ok((workers, servers))
    }

    fn adapt_workers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<Vec<(SocketAddr, WorkerSpec)>, OrchestratorError> {
        let algorithm = self.adapt_algorithm(&training.algorithm)?;
        let trainer = self.adapt_trainer(model, training)?;

        let workers = training
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

                let worker = WorkerSpec {
                    worker_id: i,
                    max_epochs: training.max_epochs,
                    trainer: trainer.clone(),
                    algorithm,
                };

                Ok((addr, worker))
            })
            .collect::<Result<Vec<_>, OrchestratorError>>()?;

        Ok(workers)
    }

    fn adapt_servers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<Vec<(SocketAddr, ServerSpec)>, OrchestratorError> {
        let AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            store,
        } = &training.algorithm;

        let (_, param_gen) = self.adapt_model_param_gen(model);

        let servers = server_addrs
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
            .collect::<Result<Vec<_>, OrchestratorError>>()?;

        Ok(servers)
    }

    fn adapt_algorithm<A: ToSocketAddrs>(
        &self,
        algorithm: &AlgorithmConfig<A>,
    ) -> Result<AlgorithmSpec, OrchestratorError> {
        let spec = match algorithm {
            AlgorithmConfig::ParameterServer { server_addrs, .. } => {
                let server_addr = server_addrs[0]
                    .to_socket_addrs()
                    .map_err(|e| OrchestratorError::ConnectionFailed {
                        addr: "server[0]".into(),
                        source: e,
                    })?
                    .next()
                    .ok_or_else(|| {
                        OrchestratorError::InvalidConfig("no server addresses provided".into())
                    })?;

                AlgorithmSpec::ParameterServer { server_addr }
            }
        };

        Ok(spec)
    }

    fn adapt_synchronizer(&self, synchronizer: &SynchronizerConfig) -> SynchronizerSpec {
        match *synchronizer {
            SynchronizerConfig::Barrier { barrier_size } => {
                SynchronizerSpec::Barrier { barrier_size }
            }
            SynchronizerConfig::NonBlocking => SynchronizerSpec::NonBlocking,
        }
    }

    fn adapt_store(&self, store: &StoreConfig) -> StoreSpec {
        match *store {
            StoreConfig::Blocking { shard_size } => StoreSpec::Blocking { shard_size },
            StoreConfig::Wild { shard_size } => StoreSpec::Wild { shard_size },
        }
    }

    fn adapt_trainer<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<TrainerSpec, OrchestratorError> {
        let (model_spec, _) = self.adapt_model_param_gen(model);
        let optimizer = self.adapt_optimizer(training.optimizer);
        let dataset = self.adapt_dataset(&training.dataset)?;
        let loss_fn = self.adapt_loss_fn(training.loss_fn);

        Ok(TrainerSpec {
            model: model_spec,
            optimizer,
            dataset,
            loss_fn,
            offline_epochs: training.offline_epochs,
            batch_size: training.batch_size,
            seed: training.seed,
        })
    }

    fn adapt_loss_fn(&self, loss_fn: LossFnConfig) -> LossFnSpec {
        match loss_fn {
            LossFnConfig::Mse => LossFnSpec::Mse,
        }
    }

    fn adapt_dataset(&self, dataset: &DatasetConfig) -> Result<DatasetSpec, OrchestratorError> {
        match dataset {
            DatasetConfig::Local { path } => Err(OrchestratorError::InvalidConfig(format!(
                "local dataset loading not yet implemented: {}",
                path.display()
            ))),
            DatasetConfig::Inline {
                data,
                x_size,
                y_size,
            } => Ok(DatasetSpec {
                data: data.to_vec(),
                x_size: *x_size,
                y_size: *y_size,
            }),
        }
    }

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

    fn adapt_model_param_gen(&self, model: &ModelConfig) -> (ModelSpec, ParamGenSpec) {
        match model {
            ModelConfig::Sequential { layers } => {
                let (layer_specs, param_gen_specs): (Vec<_>, Vec<_>) =
                    layers.iter().map(|layer| self.adapt_layer(layer)).unzip();

                (
                    ModelSpec::Sequential {
                        layers: layer_specs,
                    },
                    ParamGenSpec::Chained {
                        specs: param_gen_specs,
                    },
                )
            }
        }
    }

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

    fn adapt_act_fn(&self, act_fn: Option<&ActFnConfig>) -> Option<ActFnSpec> {
        match *act_fn? {
            ActFnConfig::Sigmoid { amp } => Some(ActFnSpec::Sigmoid { amp }),
        }
    }
}
