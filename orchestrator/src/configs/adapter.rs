use std::{
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
        self.validate_model(&model)?;
        self.validate_training(&training)?;

        let workers = self.adapt_workers(&model, &training)?;
        let servers = self.adapt_servers(&model, &training)?;
        Ok((workers, servers))
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    fn validate_model(&self, model: &ModelConfig) -> Result<(), OrchestratorError> {
        match model {
            ModelConfig::Sequential { layers } => {
                if layers.is_empty() {
                    return Err(OrchestratorError::InvalidConfig(
                        "model must have at least one layer".into(),
                    ));
                }

                // Adjacent layers must have compatible dimensions: prev.m == next.n
                for i in 1..layers.len() {
                    let prev_m = match layers[i - 1] {
                        LayerConfig::Dense { dim: (_, m), .. } => m,
                    };
                    let curr_n = match layers[i] {
                        LayerConfig::Dense { dim: (n, _), .. } => n,
                    };
                    if prev_m != curr_n {
                        return Err(OrchestratorError::InvalidConfig(format!(
                            "layer {i}: input size ({curr_n}) does not match \
                             previous layer output size ({prev_m})"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_training<A: ToSocketAddrs>(
        &self,
        training: &TrainingConfig<A>,
    ) -> Result<(), OrchestratorError> {
        // At least one worker
        if training.worker_addrs.is_empty() {
            return Err(OrchestratorError::InvalidConfig(
                "at least one worker address is required".into(),
            ));
        }

        // At least one server
        let AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            ..
        } = &training.algorithm;

        if server_addrs.is_empty() {
            return Err(OrchestratorError::InvalidConfig(
                "at least one server address is required".into(),
            ));
        }

        // barrier_size must not exceed number of workers
        if let SynchronizerConfig::Barrier { barrier_size } = synchronizer {
            if *barrier_size > training.worker_addrs.len() {
                return Err(OrchestratorError::InvalidConfig(format!(
                    "barrier_size ({barrier_size}) cannot exceed number of workers ({})",
                    training.worker_addrs.len()
                )));
            }
            if *barrier_size == 0 {
                return Err(OrchestratorError::InvalidConfig(
                    "barrier_size must be greater than 0".into(),
                ));
            }
        }

        // Dataset must have at least one sample
        let dataset_samples = match &training.dataset {
            DatasetConfig::Inline { data, x_size, y_size } => {
                let row_size = x_size + y_size;
                if row_size == 0 {
                    return Err(OrchestratorError::InvalidConfig(
                        "x_size + y_size must be greater than 0".into(),
                    ));
                }
                if data.len() % row_size != 0 {
                    return Err(OrchestratorError::InvalidConfig(format!(
                        "dataset length ({}) is not divisible by x_size + y_size ({row_size})",
                        data.len()
                    )));
                }
                data.len() / row_size
            }
            DatasetConfig::Local { path } => {
                return Err(OrchestratorError::InvalidConfig(format!(
                    "local dataset loading not yet implemented: {}",
                    path.display()
                )));
            }
        };

        if dataset_samples == 0 {
            return Err(OrchestratorError::InvalidConfig(
                "dataset must have at least one sample".into(),
            ));
        }

        // batch_size must not exceed dataset size
        if training.batch_size.get() > dataset_samples {
            return Err(OrchestratorError::InvalidConfig(format!(
                "batch_size ({}) exceeds dataset size ({dataset_samples} samples)",
                training.batch_size
            )));
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Adaptation
    // -------------------------------------------------------------------------

    fn adapt_workers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<Vec<(SocketAddr, WorkerSpec)>, OrchestratorError> {
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
                    .ok_or_else(|| OrchestratorError::InvalidConfig(
                        format!("worker[{i}]: could not resolve address")
                    ))?;

                Ok((addr, WorkerSpec {
                    worker_id: i,
                    max_epochs: training.max_epochs,
                    trainer: trainer.clone(),
                    algorithm,
                }))
            })
            .collect()
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
                    .ok_or_else(|| OrchestratorError::InvalidConfig(
                        format!("server[{i}]: could not resolve address")
                    ))?;

                Ok((addr, ServerSpec {
                    id: i,
                    nworkers: training.worker_addrs.len(),
                    param_gen: param_gen.clone(),
                    optimizer: self.adapt_optimizer(training.optimizer),
                    synchronizer: self.adapt_synchronizer(synchronizer),
                    store: self.adapt_store(store),
                    seed: training.seed,
                }))
            })
            .collect()
    }

    fn adapt_algorithm<A: ToSocketAddrs>(
        &self,
        algorithm: &AlgorithmConfig<A>,
    ) -> Result<AlgorithmSpec, OrchestratorError> {
        match algorithm {
            AlgorithmConfig::ParameterServer { server_addrs, .. } => {
                let server_addr = server_addrs[0]
                    .to_socket_addrs()
                    .map_err(|e| OrchestratorError::ConnectionFailed {
                        addr: "server[0]".into(),
                        source: e,
                    })?
                    .next()
                    .ok_or_else(|| OrchestratorError::InvalidConfig(
                        "no server addresses provided".into()
                    ))?;

                Ok(AlgorithmSpec::ParameterServer { server_addr })
            }
        }
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
            DatasetConfig::Inline { data, x_size, y_size } => Ok(DatasetSpec {
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
                    ModelSpec::Sequential { layers: layer_specs },
                    ParamGenSpec::Chained { specs: param_gen_specs },
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
            LayerConfig::Dense { dim: (n, m), init, act_fn } => {
                let act_fn = self.adapt_act_fn(act_fn.as_ref());
                (
                    LayerSpec::Dense { dim: (n, m), act_fn },
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

