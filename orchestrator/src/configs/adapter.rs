use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs,
    net::{SocketAddr, ToSocketAddrs},
    num::NonZeroUsize,
    path::PathBuf,
};

use comms::specs::{
    machine_learning::{
        ActFnSpec, DatasetSpec, LayerSpec, LossFnSpec, ModelSpec, OptimizerSpec, TrainerSpec,
    },
    server::{DistributionSpec, ParamGenSpec, ServerSpec, StoreSpec, SynchronizerSpec},
    worker::{AlgorithmSpec, WorkerSpec},
};

use super::{ModelConfig, TrainingConfig, partition::Partition};
use crate::{
    configs::{
        ActFnConfig, AlgorithmConfig, DatasetConfig, DatasetSrc, LayerConfig, LossFnConfig,
        OptimizerConfig, ParamGenConfig, StoreConfig, SynchronizerConfig,
    },
    error::{OrchErr, Result},
};

/// Converts user model and training configurations into worker and server specifications.
pub struct Adapter;

impl Adapter {
    /// Creates a new `Adapter`.
    ///
    /// # Returns
    /// A new `Adapter` instance.
    pub fn new() -> Self {
        Self
    }

    /// Adapts both `ModelConfig` and `TrainingConfig` into `WorkerSpec`, `ServerSpec` and their network addresses.
    ///
    /// # Args
    /// * `model` - The model's architecture and initialization configuration.
    /// * `training` - The training's configuration.
    ///
    /// # Returns
    /// The workers' and servers' specifications and network addresses.
    ///
    /// # Errors
    /// An `OrchErr` if the configs fail to be adapted.
    pub fn adapt_configs<A: ToSocketAddrs>(
        &self,
        model: ModelConfig,
        training: TrainingConfig<A>,
    ) -> Result<(
        Vec<(SocketAddr, WorkerSpec)>,
        Vec<Partition>,
        Vec<(SocketAddr, ServerSpec)>,
    )> {
        let (servers, server_addrs, server_sizes, server_ordering) =
            self.adapt_servers(&model, &training)?;

        let (dataset_specs, partitions) =
            self.adapt_dataset(&training.dataset, training.worker_addrs.len())?;

        let workers = self.adapt_workers(
            &model,
            &training,
            dataset_specs,
            server_addrs,
            server_sizes,
            server_ordering,
        )?;

        Ok((workers, partitions, servers))
    }

    /// Adapts both `ModelConfig` and `TrainingConfig` into a `WorkerSpec`.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    /// * `server_addrs` - Already-resolved server socket addresses.
    /// * `server_sizes` - The amounts of parameters per server.
    /// * `server_ordering` - The ordering of the layer owners.
    ///
    /// # Returns
    /// Worker addresses and specifications.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any address cannot be resolved.
    fn adapt_workers<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
        dataset_specs: Vec<DatasetSpec>,
        server_addrs: Vec<SocketAddr>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    ) -> Result<Vec<(SocketAddr, WorkerSpec)>> {
        let trainer_spec = self.adapt_trainer(model, training);
        let algorithm_spec = AlgorithmSpec::ParameterServer {
            server_addrs,
            server_sizes,
            server_ordering,
        };

        let worker_specs = training
            .worker_addrs
            .iter()
            .enumerate()
            .zip(dataset_specs)
            .map(|((i, addressable), dataset)| {
                let addr = addressable.to_socket_addrs()?.next().ok_or_else(|| {
                    OrchErr::InvalidConfig(format!(
                        "failted to resolve {i}'th worker's network address"
                    ))
                })?;

                let worker_spec = WorkerSpec {
                    worker_id: i,
                    trainer: trainer_spec.clone(),
                    dataset,
                    algorithm: algorithm_spec.clone(),
                };

                Ok((addr, worker_spec))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(worker_specs)
    }

    /// Adapts both model's and trianing's configurations into a `ServerSpec`.
    ///
    /// # Args
    /// * `model` - The model's architecture and initialization configuration.
    /// * `training` - The training's configuration.
    ///
    /// # Returns
    /// The servers' specifications, resolved addresses, sizes and layers' ordering.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any address cannot be resolved.
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

        let (_, param_gens) = self.adapt_model_param_gen(model, training.dataset.x_size);
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

        let nworkers = training.worker_addrs.len();
        let (servers, server_sizes): (Vec<_>, Vec<_>) = server_addrs
            .iter()
            .zip(chained_param_gens)
            .enumerate()
            .map(|(i, (addressable, (param_gen_spec, size)))| {
                let addr = addressable.to_socket_addrs()?.next().ok_or_else(|| {
                    OrchErr::InvalidConfig(format!(
                        "failed to resolve {i}'th server's network address"
                    ))
                })?;

                let server_spec = ServerSpec {
                    id: i,
                    nworkers: training.worker_addrs.len(),
                    param_gen: param_gen_spec,
                    optimizer: self.adapt_optimizer(training.optimizer),
                    synchronizer: self.adapt_synchronizer(synchronizer, nworkers),
                    store: self.adapt_store(store),
                    seed: training.seed,
                };

                Ok(((addr, server_spec), size))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();

        let server_addrs = servers.iter().map(|&(addr, _)| addr).collect();
        Ok((servers, server_addrs, server_sizes, server_ordering))
    }

    /// Adapts a `SynchronizerConfig` into a `SynchronizerSpec`.
    ///
    /// # Args
    /// * `synchronizer` - A synchronizer's configuration.
    ///
    /// # Returns
    /// THe synchronizer's specification.
    fn adapt_synchronizer(
        &self,
        synchronizer: &SynchronizerConfig,
        worker_amount: usize,
    ) -> SynchronizerSpec {
        match *synchronizer {
            SynchronizerConfig::Barrier => SynchronizerSpec::Barrier {
                barrier_size: worker_amount,
            },
            SynchronizerConfig::NonBlocking => SynchronizerSpec::NonBlocking,
        }
    }

    /// Adapts a `StoreConfig` into a `StoreSpec`.
    ///
    /// # Args
    /// * `store` - A store's configuration.
    ///
    /// # Returns
    /// The store's specification.
    fn adapt_store(&self, store: &StoreConfig) -> StoreSpec {
        match *store {
            StoreConfig::Blocking => StoreSpec::Blocking,
            StoreConfig::Wild => StoreSpec::Wild,
        }
    }

    /// Adapts a `ModelConfig` and a `TrainingConfig` into a `TrainerSpec`.
    ///
    /// # Args
    /// * `model` - A model architecture configuration.
    /// * `training` - A training configuration.
    ///
    /// # Returns
    /// The trainer's specification.
    fn adapt_trainer<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> TrainerSpec {
        let (model_spec, _) = self.adapt_model_param_gen(model, training.dataset.x_size);
        let optimizer_spec = self.adapt_optimizer(training.optimizer);
        let loss_fn_spec = self.adapt_loss_fn(training.loss_fn);

        TrainerSpec {
            model: model_spec,
            optimizer: optimizer_spec,
            loss_fn: loss_fn_spec,
            offline_epochs: training.offline_epochs,
            max_epochs: training.max_epochs,
            batch_size: training.batch_size,
            seed: training.seed,
        }
    }

    /// Adapts a `LossFnConfig` into a `LossFnSpec`.
    ///
    /// # Args
    /// * `loss_fn` - A loss function's configuration.
    ///
    /// # Returns
    /// The loss function's specification.
    fn adapt_loss_fn(&self, loss_fn: LossFnConfig) -> LossFnSpec {
        match loss_fn {
            LossFnConfig::Mse => LossFnSpec::Mse,
        }
    }

    fn adapt_local_dataset(
        &self,
        path: &PathBuf,
        x_size: usize,
        y_size: usize,
        npartitions: usize,
    ) -> Result<(Vec<DatasetSpec>, Vec<Partition>)> {
        let size = fs::metadata(path)?.len();
        let row_size = (x_size + y_size) as u64;
        let nrows = (size / row_size) as usize;
        let base_rows = (nrows / npartitions) as u64;
        let remainder = nrows % npartitions;

        let mut specs = vec![];
        let mut partitions = vec![];
        let mut offset = 0;

        for i in 0..npartitions {
            let size = if i < remainder {
                base_rows + 1
            } else {
                base_rows
            } * row_size;

            specs.push(DatasetSpec {
                size,
                x_size,
                y_size,
            });
            partitions.push(Partition {
                // TODO: avoid cloning path
                path: path.clone(),
                offset,
                size,
            });

            offset += size;
        }

        Ok((vec![], vec![]))
    }

    /// Converts a `DatasetConfig` into `DatasetSpec`s and `Partition`s.
    ///
    /// # Args
    /// * `dataset` - A dataset's configuration.
    /// * `npartitions` - The amount of partitions.
    ///
    /// # Returns
    /// A list of resolved dataset specs and a list with partition metadata.
    ///
    /// # Errors
    /// Returns an `OrchErr` if the dataset cannot be resolved.
    fn adapt_dataset(
        &self,
        dataset: &DatasetConfig,
        npartitions: usize,
    ) -> Result<(Vec<DatasetSpec>, Vec<Partition>)> {
        let DatasetConfig {
            src,
            x_size,
            y_size,
        } = dataset;

        let (x_size, y_size) = (x_size.get(), y_size.get());

        match src {
            DatasetSrc::Local { path } => {
                self.adapt_local_dataset(path, x_size, y_size, npartitions)
            }
            _ => unimplemented!(),
        }
    }

    /// Adapts an `OptimizerConfig` into an `OptimizerSpec`.
    ///
    /// # Args
    /// * `optimizer` - A optimizer's configuration.
    ///
    /// # Returns
    /// The optimizer specification.
    fn adapt_optimizer(&self, optimizer: OptimizerConfig) -> OptimizerSpec {
        match optimizer {
            OptimizerConfig::GradientDescent { lr } => {
                OptimizerSpec::GradientDescent { learning_rate: lr }
            }
        }
    }

    /// Adapts a `ModelConfig` into both `ModelSpec` and per layer `ParamGenSpec`s.
    ///
    /// # Args
    /// * `model` - A model's architecture and initialization configuration.
    /// * `input_size` - The model's input size.
    ///
    /// # Returns
    /// The model's specification and it's layers' parameter generators specifications.
    fn adapt_model_param_gen(
        &self,
        model: &ModelConfig,
        input_size: NonZeroUsize,
    ) -> (ModelSpec, Vec<ParamGenSpec>) {
        match model {
            ModelConfig::Sequential { layers } => {
                let (layer_specs, param_gen_specs) = layers
                    .iter()
                    .scan(input_size, |input_size, config| {
                        let (layer_spec, param_gen_spec, output_size) =
                            self.adapt_layer(config, *input_size);
                        *input_size = output_size;
                        Some((layer_spec, param_gen_spec))
                    })
                    .unzip();

                let model_spec = ModelSpec::Sequential {
                    layers: layer_specs,
                };

                (model_spec, param_gen_specs)
            }
        }
    }

    /// Adapts a `ParamGenConfig` into a `ParamGenSpec`.
    ///
    /// # Args
    /// * `param_gen` - A parameter generator configuration.
    /// * `(fan_in, limit, fan_out)` - It's layer sizing information.
    ///
    /// # Returns
    /// The parameter generator's specification.
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

    /// Adapts a `LayerConfig` into both `LayerSpec` and `ParamGenSpec`.
    ///
    /// # Args
    /// * `layer` - The layer configuration.
    /// * `input_size` - The input size to this layer.
    ///
    /// # Returns
    /// The layer's specification, it's parameter generator specifications and it's output size.
    fn adapt_layer(
        &self,
        layer: &LayerConfig,
        input_size: NonZeroUsize,
    ) -> (LayerSpec, ParamGenSpec, NonZeroUsize) {
        match *layer {
            LayerConfig::Dense {
                output_size,
                init,
                act_fn,
            } => {
                let act_fn_spec = act_fn.map(|act_fn| self.adapt_act_fn(act_fn));
                let layer_size = input_size.saturating_add(1).saturating_mul(output_size);
                let sizes = (input_size.get(), layer_size.get(), output_size.get());

                (
                    LayerSpec::Dense {
                        dim: (sizes.0, sizes.2),
                        act_fn: act_fn_spec,
                    },
                    self.adapt_param_gen(init, sizes),
                    output_size,
                )
            }
        }
    }

    /// Adapts an `ActFnConfig` into an `ActFnSpec`.
    ///
    /// # Args
    /// * `act_fn` - An activation function configuration.
    ///
    /// # Returns
    /// The activation function's specification.
    fn adapt_act_fn(&self, act_fn: ActFnConfig) -> ActFnSpec {
        match act_fn {
            ActFnConfig::Sigmoid { amp } => ActFnSpec::Sigmoid { amp },
        }
    }
}

/// Approximates the balanced partitions problem.
///
/// # Args
/// * `items` - The items to distribute in bins.
/// * `k` - The amount of bins.
///
/// # Returns
/// k vecs with an approximate distribution which minimizes the difference
/// in size between the minimum and maximum sizes.
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
