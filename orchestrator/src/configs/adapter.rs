use std::{
    cmp::Reverse,
    collections::BinaryHeap,
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

        let (servers, server_addrs, server_sizes, server_ordering) =
            self.adapt_servers(&model, &training)?;
        let workers = self.adapt_workers(
            &model,
            &training,
            server_addrs,
            server_sizes,
            server_ordering,
        )?;
        Ok((workers, servers))
    }

    /// Adapts worker addresses and model/training configs into `WorkerSpec`s.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    /// * `server_addrs` - Already-resolved server socket addresses.
    /// * `server_sizes` - The amounts of parameters per server.
    /// * `server_ordering` - The ordering of the layer owners.
    ///
    /// # Returns
    /// A list of resolved worker addresses paired with their specs.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any address cannot be resolved.
    fn adapt_workers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
        server_addrs: Vec<SocketAddr>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    ) -> Result<Vec<(SocketAddr, WorkerSpec)>> {
        let trainer = self.adapt_trainer(model, training)?;
        let algorithm = self.adapt_algorithm(server_addrs, server_sizes, server_ordering);

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
                    trainer: trainer.clone(),
                    algorithm: algorithm.clone(),
                };
                Ok((addr, spec))
            })
            .collect()
    }

    /// Adapts server addresses and model/training configs into `ServerSpec`s.
    ///
    /// # Args
    /// * `model` - The model architecture configuration.
    /// * `training` - The training configuration.
    ///
    /// # Returns
    /// The server specifications, their resolved addresses, sizes and ordering.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any address cannot be resolved.
    fn adapt_servers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<(
        Vec<(SocketAddr, ServerSpec)>,
        Vec<SocketAddr>,
        Vec<usize>,
        Vec<usize>,
    )> {
        let AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            store,
            ..
        } = &training.algorithm;

        let (_, param_gens) = self.adapt_model_param_gen(model);
        let nlayers = param_gens.len();

        let items: Vec<_> = param_gens
            .into_iter()
            .enumerate()
            .map(|(i, param_gen)| {
                let size = param_gen.size();
                ((i, param_gen), size)
            })
            .collect();

        let param_gen_bins = balanced_partitions(items, server_addrs.len());
        let mut server_ordering = vec![0; nlayers];

        for (server_i, bin) in param_gen_bins.iter().enumerate() {
            for &(layer_i, _) in bin {
                server_ordering[layer_i] = server_i;
            }
        }

        let chained_param_gens = param_gen_bins.into_iter().map(|bin| {
            let (specs, sizes): (Vec<_>, Vec<_>) = bin
                .into_iter()
                .map(|(_, spec)| {
                    let size = spec.size();
                    (spec, size)
                })
                .unzip();

            let chained = ParamGenSpec::Chained { specs };
            (chained, sizes.into_iter().sum::<usize>())
        });

        let mut resolved_addrs: Vec<SocketAddr> = Vec::with_capacity(server_addrs.len());
        let mut servers: Vec<(SocketAddr, ServerSpec)> = Vec::with_capacity(server_addrs.len());
        let mut server_sizes: Vec<usize> = Vec::with_capacity(server_addrs.len());

        for (i, (addressable, (param_gen, size))) in
            server_addrs.iter().zip(chained_param_gens).enumerate()
        {
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
                param_gen,
                optimizer: self.adapt_optimizer(training.optimizer),
                synchronizer: self.adapt_synchronizer(synchronizer),
                store: self.adapt_store(store),
                seed: training.seed,
            };

            resolved_addrs.push(addr);
            servers.push((addr, spec));
            server_sizes.push(size);
        }

        Ok((servers, resolved_addrs, server_sizes, server_ordering))
    }

    /// Builds an `AlgorithmSpec` from already-resolved server addresses.
    ///
    /// # Args
    /// * `server_addrs` - Already-resolved server socket addresses.
    /// * `server_sizes` - The amount of parameters per server.
    /// * `server_ordering` - The ordering of the layer owners.
    ///
    /// # Returns
    /// The resolved `AlgorithmSpec`.
    fn adapt_algorithm(
        &self,
        server_addrs: Vec<SocketAddr>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    ) -> AlgorithmSpec {
        AlgorithmSpec::ParameterServer {
            server_addrs,
            server_sizes,
            server_ordering,
        }
    }

    /// Converts a `SynchronizerConfig` into a `SynchronizerSpec`.
    ///
    /// # Args
    /// * `synchronizer` - The synchronizer configuration.
    ///
    /// # Returns
    /// The resolved `SynchronizerSpec`.
    fn adapt_synchronizer(&self, synchronizer: &SynchronizerConfig) -> SynchronizerSpec {
        match *synchronizer {
            SynchronizerConfig::Barrier { barrier_size } => {
                SynchronizerSpec::Barrier { barrier_size }
            }
            SynchronizerConfig::NonBlocking => SynchronizerSpec::NonBlocking,
        }
    }

    /// Converts a `StoreConfig` into a `StoreSpec`.
    ///
    /// # Args
    /// * `store` - The store configuration.
    ///
    /// # Returns
    /// The resolved `StoreSpec`.
    fn adapt_store(&self, store: &StoreConfig) -> StoreSpec {
        match *store {
            StoreConfig::Blocking => StoreSpec::Blocking,
            StoreConfig::Wild => StoreSpec::Wild,
        }
    }

    /// Builds a `TrainerSpec` from the model and training configs.
    ///
    /// # Args
    /// * `model` - The model architecture configuration.
    /// * `training` - The training configuration.
    ///
    /// # Returns
    /// The resolved `TrainerSpec`.
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

        Ok(TrainerSpec {
            model: model_spec,
            optimizer,
            dataset,
            loss_fn,
            offline_epochs: training.offline_epochs,
            max_epochs: training.max_epochs,
            batch_size: training.batch_size,
            seed: training.seed,
        })
    }

    /// Converts a `LossFnConfig` into a `LossFnSpec`.
    ///
    /// # Args
    /// * `loss_fn` - The loss function configuration.
    ///
    /// # Returns
    /// The resolved `LossFnSpec`.
    fn adapt_loss_fn(&self, loss_fn: LossFnConfig) -> LossFnSpec {
        match loss_fn {
            LossFnConfig::Mse => LossFnSpec::Mse,
        }
    }

    /// Converts a `DatasetConfig` into a `DatasetSpec`.
    ///
    /// # Args
    /// * `dataset` - The dataset configuration.
    ///
    /// # Returns
    /// The resolved `DatasetSpec`.
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
    ///
    /// # Args
    /// * `optimizer` - The optimizer configuration.
    ///
    /// # Returns
    /// The resolved `OptimizerSpec`.
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

    /// Converts a `ModelConfig` into a `ModelSpec` and its associated per-layer `ParamGenSpec`s.
    ///
    /// # Args
    /// * `model` - The model architecture configuration.
    ///
    /// # Returns
    /// A tuple of the resolved `ModelSpec` and a `ParamGenSpec` per layer.
    fn adapt_model_param_gen(&self, model: &ModelConfig) -> (ModelSpec, Vec<ParamGenSpec>) {
        match model {
            ModelConfig::Sequential { layers } => {
                let (layer_specs, param_gen_specs): (Vec<_>, Vec<_>) =
                    layers.iter().map(|layer| self.adapt_layer(layer)).unzip();

                (
                    ModelSpec::Sequential { layers: layer_specs },
                    param_gen_specs,
                )
            }
        }
    }

    /// Converts a `ParamGenConfig` into a `ParamGenSpec` given the layer dimensions.
    ///
    /// # Args
    /// * `param_gen` - The parameter generator configuration.
    /// * `(fan_in, limit, fan_out)` - Layer sizing info used by distribution-based generators.
    ///
    /// # Returns
    /// The resolved `ParamGenSpec`.
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
    ///
    /// # Args
    /// * `layer` - The layer configuration.
    ///
    /// # Returns
    /// A tuple of the resolved `LayerSpec` and its `ParamGenSpec`.
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

    /// Converts an optional `ActFnConfig` reference into an optional `ActFnSpec`.
    ///
    /// # Args
    /// * `act_fn` - The optional activation function configuration.
    ///
    /// # Returns
    /// The resolved `ActFnSpec`, or `None` if no activation function is configured.
    fn adapt_act_fn(&self, act_fn: Option<&ActFnConfig>) -> Option<ActFnSpec> {
        Some(match *act_fn? {
            ActFnConfig::Sigmoid { amp } => ActFnSpec::Sigmoid { amp },
        })
    }
}

fn balanced_partitions<T>(mut items: Vec<(T, usize)>, k: usize) -> Vec<Vec<T>> {
    let mut sizes: BinaryHeap<_> = (0..k).map(|i| Reverse((0, i))).collect();
    let mut bins: Vec<_> = (0..k).map(|_| Vec::new()).collect();

    items.sort_unstable_by_key(|(_, size)| Reverse(*size));

    for (item, size) in items {
        let Reverse((bin_size, i)) = sizes.pop().unwrap();
        bins[i].push(item);
        sizes.push(Reverse((bin_size + size, i)));
    }

    bins
}