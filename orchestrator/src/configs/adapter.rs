use std::{
    cmp::Reverse, collections::BinaryHeap, fs, net::ToSocketAddrs, num::NonZeroUsize, path::PathBuf,
};

use comms::specs::{
    machine_learning::{
        ActFnSpec, DatasetSpec, DistributionSpec, LayerSpec, LossFnSpec, OptimizerSpec,
        ParamGenSpec, TrainerSpec,
    },
    server::{ServerSpec, StoreSpec, SynchronizerSpec},
    worker::{AlgorithmSpec, SerializerSpec, WorkerSpec},
};

use super::{ModelConfig, SerializerConfig, TrainingConfig, partition::Partition};
use crate::{
    configs::{
        ActFnConfig, AlgorithmConfig, DatasetConfig, DatasetSrc, LayerConfig, LossFnConfig,
        OptimizerConfig, ParamGenConfig, StoreConfig, SynchronizerConfig,
    },
    error::{OrchErr, Result},
};

/// Converts user model and training configurations into worker and server specifications.
#[derive(Default)]
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
    #[allow(clippy::type_complexity)]
    pub fn adapt_configs<'a>(
        &self,
        model: ModelConfig,
        training: &'a TrainingConfig,
    ) -> Result<(
        Vec<(String, WorkerSpec)>,
        Vec<Partition<'a>>,
        Vec<(String, ServerSpec)>,
    )> {
        let (dataset_specs, partitions) =
            self.adapt_dataset(&training.dataset, training.worker_addrs.len())?;

        match &training.algorithm {
            AlgorithmConfig::ParameterServer { .. } => {
                let (servers, server_addrs, server_sizes, server_ordering) =
                    self.adapt_servers(&model, training)?;

                let workers = self.adapt_parameter_server_workers(
                    &model,
                    training,
                    dataset_specs,
                    server_addrs,
                    server_sizes,
                    server_ordering,
                )?;

                Ok((workers, partitions, servers))
            }
            AlgorithmConfig::AllReduce => {
                let workers = self.adapt_all_reduce_workers(&model, training, dataset_specs)?;

                Ok((workers, partitions, Vec::new()))
            }
        }
    }

    /// Adapts both `ModelConfig` and `TrainingConfig` into a `WorkerSpec`.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    /// * `dataset_specs` - Already-resolved dataset specs per worker.
    /// * `server_addrs` - Already-resolved server socket addresses.
    /// * `server_sizes` - The amounts of parameters per server.
    /// * `server_ordering` - The ordering of the layer owners.
    ///
    /// # Returns
    /// Worker addresses and specifications.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any address cannot be resolved.
    fn adapt_parameter_server_workers(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig,
        dataset_specs: Vec<DatasetSpec>,
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    ) -> Result<Vec<(String, WorkerSpec)>> {
        let trainer_spec = self.adapt_trainer(model, training);
        let algorithm_spec = AlgorithmSpec::ParameterServer {
            server_addrs,
            server_sizes,
            server_ordering,
        };
        let serializer_spec = self.adapt_serializer(training);

        let worker_specs = training
            .worker_addrs
            .iter()
            .enumerate()
            .zip(dataset_specs)
            .map(|((i, addr), dataset)| {
                if let Err(..) | Ok(None) = addr.to_socket_addrs().map(|mut addrs| addrs.next()) {
                    let text = format!("failed to resolve {i}'th worker's network address: {addr}");
                    return Err(OrchErr::InvalidConfig(text));
                }

                let worker_spec = WorkerSpec {
                    worker_id: i,
                    trainer: trainer_spec.clone(),
                    dataset,
                    algorithm: algorithm_spec.clone(),
                    serializer: serializer_spec.clone(),
                    seed: training.seed,
                };

                Ok((addr.clone(), worker_spec))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(worker_specs)
    }

    /// Adapts both `ModelConfig` and `TrainingConfig` into a `WorkerSpec`.
    ///
    /// # Args
    /// * `model` - The model's configuration.
    /// * `training` - The training's configuration.
    /// * `dataset_specs` - Already-resolved dataset specs per worker.
    ///
    /// # Returns
    /// Worker addresses and specifications.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any address cannot be resolved.
    fn adapt_all_reduce_workers(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig,
        dataset_specs: Vec<DatasetSpec>,
    ) -> Result<Vec<(String, WorkerSpec)>> {
        let trainer_spec = self.adapt_trainer(model, training);
        let (_, param_gen_specs) = self.adapt_layers(model, training.dataset.x_size);
        let algorithm_spec = AlgorithmSpec::AllReduce {
            worker_addrs: training.worker_addrs.clone(),
            param_gen: ParamGenSpec::Chained {
                specs: param_gen_specs,
            },
            amount_of_layers: model.layers.len(),
        };
        let serializer_spec = self.adapt_serializer(training);

        let worker_specs = training
            .worker_addrs
            .iter()
            .enumerate()
            .zip(dataset_specs)
            .map(|((i, addr), dataset)| {
                if let Err(..) | Ok(None) = addr.to_socket_addrs().map(|mut addrs| addrs.next()) {
                    let text = format!("failed to resolve {i}'th worker's network address: {addr}");
                    return Err(OrchErr::InvalidConfig(text));
                }

                let worker_spec = WorkerSpec {
                    worker_id: i,
                    trainer: trainer_spec.clone(),
                    dataset,
                    algorithm: algorithm_spec.clone(),
                    serializer: serializer_spec.clone(),
                    seed: training.seed,
                };

                Ok((addr.clone(), worker_spec))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(worker_specs)
    }

    /// Adapts the training's configurations into a `SerializerSpec`.
    ///
    /// # Args
    /// * `training` - The training's configuration.
    ///
    /// # Returns
    /// The serializer's specification.
    fn adapt_serializer(&self, training: &TrainingConfig) -> SerializerSpec {
        match training.serializer {
            SerializerConfig::Base => SerializerSpec::Base,
            SerializerConfig::SparseCapable { r } => SerializerSpec::SparseCapable { r },
        }
    }

    /// Adapts both model's and training's configurations into a `ServerSpec`.
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
    #[allow(clippy::type_complexity)]
    fn adapt_servers(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig,
    ) -> Result<(
        Vec<(String, ServerSpec)>,
        Vec<String>,
        Vec<usize>,
        Vec<usize>,
    )> {
        let AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            store,
            ..
        } = &training.algorithm
        else {
            let text = "all-reduce does not use parameter servers".into();
            return Err(OrchErr::Unsupported(text));
        };

        let (_, param_gens) = self.adapt_layers(model, training.dataset.x_size);
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
            for &(layer_i, ..) in bin {
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
            .map(|(i, (addr, (param_gen_spec, size)))| {
                if let Err(..) | Ok(None) = addr.to_socket_addrs().map(|mut addrs| addrs.next()) {
                    let text = format!("failed to resolve {i}'th server's network address: {addr}");
                    return Err(OrchErr::InvalidConfig(text));
                }

                let server_spec = ServerSpec {
                    id: i,
                    nworkers: training.worker_addrs.len(),
                    param_gen: param_gen_spec,
                    optimizer: self.adapt_optimizer(training.optimizer),
                    synchronizer: self.adapt_synchronizer(synchronizer, nworkers),
                    store: self.adapt_store(store),
                    seed: training.seed,
                };

                Ok(((addr.clone(), server_spec), size))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();

        let server_addrs = servers.iter().map(|(addr, _)| addr.clone()).collect();
        Ok((servers, server_addrs, server_sizes, server_ordering))
    }

    /// Adapts a `SynchronizerConfig` into a `SynchronizerSpec`.
    ///
    /// # Args
    /// * `synchronizer` - A synchronizer's configuration.
    /// * `worker_amount` - The total number of workers.
    ///
    /// # Returns
    /// The synchronizer's specification.
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
    fn adapt_trainer(&self, model: &ModelConfig, training: &TrainingConfig) -> TrainerSpec {
        let (layers, _) = self.adapt_layers(model, training.dataset.x_size);
        let loss_fn_spec = self.adapt_loss_fn(training.loss_fn);
        let optimizer_spec =
            if matches!(training.algorithm, AlgorithmConfig::ParameterServer { .. }) {
                self.adapt_optimizer_to_gradient_descent(training.optimizer)
            } else {
                self.adapt_optimizer(training.optimizer)
            };

        TrainerSpec {
            layers,
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
            LossFnConfig::CrossEntropy => LossFnSpec::CrossEntropy,
        }
    }

    /// Helper method for `adapt_dataset` — partitions an inline dataset slice.
    ///
    /// Partition sizes must contemplate the rows of data that will go in each partition.
    // TODO: return an error if this condition is not met, otherwise this will result in undefined
    // behavior
    ///
    /// # Args
    /// * `samples` - The full dataset samples as a slice of `f32`.
    /// * `labels` - The full dataset labels as a slice of `f32`.
    /// * `partition_sizes` - An iterator of partition sample and label sizes in **bytes**.
    /// * `x_size` - The number of input features per sample.
    /// * `y_size` - The number of output values per sample.
    ///
    /// # Returns
    /// Paired lists of `DatasetSpec`s and `Partition::Inline` variants.
    fn adapt_inline_dataset<'a, T>(
        &self,
        samples: &'a [f32],
        labels: &'a [f32],
        partition_sizes: T,
        x_size: NonZeroUsize,
        y_size: NonZeroUsize,
    ) -> (Vec<DatasetSpec>, Vec<Partition<'a>>)
    where
        T: Iterator<Item = (u64, u64)>,
    {
        let mut samples_rest = samples;
        let mut labels_rest = labels;
        partition_sizes
            .map(|(x_size_bytes, y_size_bytes)| {
                let samples_curr;
                let labels_curr;
                (samples_curr, samples_rest) =
                    samples_rest.split_at((x_size_bytes / size_of::<f32>() as u64) as usize);
                (labels_curr, labels_rest) =
                    labels_rest.split_at((y_size_bytes / size_of::<f32>() as u64) as usize);
                let spec = DatasetSpec {
                    x_size_bytes,
                    y_size_bytes,
                    x_size,
                    y_size,
                };
                let partition = Partition::Inline {
                    samples: samples_curr,
                    labels: labels_curr,
                };

                (spec, partition)
            })
            .collect()
    }

    /// Helper method for `adapt_dataset` — partitions a local dataset file.
    ///
    /// # Args
    /// * `samples_path` - Path to the local dataset samples file.
    /// * `labels_path` - Path to the local dataset labels file.
    /// * `partition_sizes` - An iterator of partition sample and label sizes in **bytes**.
    /// * `x_size` - The number of input features per sample.
    /// * `y_size` - The number of output values per sample.
    ///
    /// # Returns
    /// Paired lists of `DatasetSpec`s and `Partition::Local` variants.
    fn adapt_local_dataset<'a, T>(
        &self,
        samples_path: &'a PathBuf,
        labels_path: &'a PathBuf,
        partition_sizes: T,
        x_size: NonZeroUsize,
        y_size: NonZeroUsize,
    ) -> (Vec<DatasetSpec>, Vec<Partition<'a>>)
    where
        T: Iterator<Item = (u64, u64)>,
    {
        let mut samples_offset = 0;
        let mut labels_offset = 0;
        partition_sizes
            .map(|(x_size_bytes, y_size_bytes)| {
                let spec = DatasetSpec {
                    x_size_bytes,
                    y_size_bytes,
                    x_size,
                    y_size,
                };
                let partition = Partition::Local {
                    samples_path,
                    labels_path,
                    samples_offset,
                    labels_offset,
                    samples_size: x_size_bytes,
                    labels_size: y_size_bytes,
                };

                samples_offset += x_size_bytes;
                labels_offset += y_size_bytes;

                (spec, partition)
            })
            .collect()
    }

    /// Converts a `DatasetConfig` into `DatasetSpec`s and `Partition`s.
    ///
    /// Partition sizes are computed in **bytes** so that they align correctly
    /// with what `send_dataset` reads off the wire. Previously `row_size` was
    /// expressed in number-of-f32s while `size` was in bytes, causing each
    /// partition to be 4× larger than intended.
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
    fn adapt_dataset<'a>(
        &self,
        dataset: &'a DatasetConfig,
        npartitions: usize,
    ) -> Result<(Vec<DatasetSpec>, Vec<Partition<'a>>)> {
        let DatasetConfig {
            src,
            x_size,
            y_size,
        } = dataset;

        let size_bytes = match src {
            DatasetSrc::Local {
                samples_path,
                labels_path,
            } => {
                let samples_size_bytes = fs::metadata(samples_path)?.len();
                let labels_size_bytes = fs::metadata(labels_path)?.len();
                samples_size_bytes + labels_size_bytes
            }
            DatasetSrc::Inline { samples, labels } => {
                let data_len = samples.len() + labels.len();
                (data_len * size_of::<f32>()) as u64
            }
        };

        let npartitions = npartitions as u64;

        let x_size_bytes = (x_size.get() * size_of::<f32>()) as u64;
        let y_size_bytes = (y_size.get() * size_of::<f32>()) as u64;
        let row_size_bytes = x_size_bytes + y_size_bytes;
        let nrows = size_bytes / row_size_bytes;
        let base_rows = nrows / npartitions;
        let remainder = nrows % npartitions;

        let partition_sizes = (0..npartitions).map(|i| {
            let rows = if i < remainder {
                base_rows + 1
            } else {
                base_rows
            };

            (rows * x_size_bytes, rows * y_size_bytes)
        });

        let (specs, partitions) = match src {
            DatasetSrc::Inline { samples, labels } => {
                self.adapt_inline_dataset(samples, labels, partition_sizes, *x_size, *y_size)
            }
            DatasetSrc::Local {
                samples_path,
                labels_path,
            } => self.adapt_local_dataset(
                samples_path,
                labels_path,
                partition_sizes,
                *x_size,
                *y_size,
            ),
        };

        Ok((specs, partitions))
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
            OptimizerConfig::GradientDescentWithMomentum { lr, mu } => {
                OptimizerSpec::GradientDescentWithMomentum {
                    learning_rate: lr,
                    momentum: mu,
                }
            }
            OptimizerConfig::Adam { lr, b1, b2, eps } => OptimizerSpec::Adam {
                learning_rate: lr,
                beta1: b1,
                beta2: b2,
                epsilon: eps,
            },
        }
    }

    /// Adapts an `OptimizerConfig` into an `OptimizerSpec::GradientDescent`.
    ///
    /// # Args
    /// * `optimizer` - A optimizer's configuration.
    ///
    /// # Returns
    /// The optimizer specification.
    fn adapt_optimizer_to_gradient_descent(&self, optimizer: OptimizerConfig) -> OptimizerSpec {
        match optimizer {
            OptimizerConfig::GradientDescent { lr } => {
                OptimizerSpec::GradientDescent { learning_rate: lr }
            }
            OptimizerConfig::GradientDescentWithMomentum { lr, .. } => {
                OptimizerSpec::GradientDescent { learning_rate: lr }
            }
            OptimizerConfig::Adam { lr, .. } => {
                OptimizerSpec::GradientDescent { learning_rate: lr }
            }
        }
    }

    /// Adapts a `ModelConfig` into the model's layers and the per layer `ParamGenSpec`s.
    ///
    /// # Args
    /// * `model` - A model's architecture and initialization configuration.
    /// * `input_size` - The model's input size.
    ///
    /// # Returns
    /// The layers' specifications and their parameter generators' specifications.
    fn adapt_layers(
        &self,
        model: &ModelConfig,
        input_size: NonZeroUsize,
    ) -> (Vec<LayerSpec>, Vec<ParamGenSpec>) {
        let (layer_specs, param_gen_specs) = model
            .layers
            .iter()
            .scan(input_size, |input_size, config| {
                let (layer_spec, param_gen_spec, output_size) =
                    self.adapt_layer(config, *input_size);
                *input_size = output_size;
                Some((layer_spec, param_gen_spec))
            })
            .unzip();

        (layer_specs, param_gen_specs)
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
            ParamGenConfig::XavierUniform => ParamGenSpec::Rand {
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
            LayerConfig::Conv {
                input_dim,
                kernel_dim,
                stride,
                padding,
                init,
                act_fn,
            } => {
                let act_fn_spec = act_fn.map(|act_fn| self.adapt_act_fn(act_fn));

                let (filters, channels, kernel_size) =
                    (kernel_dim.0.get(), kernel_dim.1.get(), kernel_dim.2.get());

                let layer_size = filters * channels * kernel_size * kernel_size + filters;
                let input_dim = (input_dim.0.get(), input_dim.1.get(), input_dim.2.get());
                let output_height = (input_dim.1 + 2 * padding - kernel_size) / stride + 1;
                let output_width = (input_dim.2 + 2 * padding - kernel_size) / stride + 1;
                let output_size = output_height * output_width * filters;
                let sizes = (input_size.get(), layer_size, output_size);

                (
                    LayerSpec::Conv {
                        input_dim,
                        kernel_dim: (filters, channels, kernel_size),
                        stride: stride.get(),
                        padding,
                        act_fn: act_fn_spec,
                    },
                    self.adapt_param_gen(init, sizes),
                    // SAFETY: input config has already been validated at this point.
                    NonZeroUsize::new(output_size).unwrap(),
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
            ActFnConfig::Softmax => ActFnSpec::Softmax,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_adapt_inline_dataset() {
        let samples = [1., 3., 5.];
        let labels = [2., 4., 6.];
        let x_size = NonZeroUsize::new(1).unwrap();
        let y_size = NonZeroUsize::new(1).unwrap();
        let npartitions = 3;
        let x_size_bytes = ((samples.len() / npartitions) * size_of::<f32>()) as u64;
        let y_size_bytes = ((labels.len() / npartitions) * size_of::<f32>()) as u64;
        let config = DatasetConfig {
            src: DatasetSrc::Inline {
                samples: samples.into(),
                labels: labels.into(),
            },
            x_size,
            y_size,
        };

        let expected_specs = [
            DatasetSpec {
                x_size_bytes,
                y_size_bytes,
                x_size,
                y_size,
            },
            DatasetSpec {
                x_size_bytes,
                y_size_bytes,
                x_size,
                y_size,
            },
            DatasetSpec {
                x_size_bytes,
                y_size_bytes,
                x_size,
                y_size,
            },
        ];
        let expected_partitions = [
            Partition::Inline {
                samples: &samples[..1],
                labels: &labels[..1],
            },
            Partition::Inline {
                samples: &samples[1..2],
                labels: &labels[1..2],
            },
            Partition::Inline {
                samples: &samples[2..],
                labels: &labels[2..],
            },
        ];

        let adapter = Adapter::new();
        let (specs, partitions) = adapter.adapt_dataset(&config, npartitions).unwrap();

        assert_eq!(specs, expected_specs);
        assert_eq!(partitions, expected_partitions);
    }

    #[test]
    fn test_adapter_adapt_inline_xor2_3partitions() {
        let samples = [0., 0., 0., 1., 1., 0., 1., 1.];
        let labels = [0., 1., 1., 0.];
        let x_size = NonZeroUsize::new(2).unwrap();
        let y_size = NonZeroUsize::new(1).unwrap();
        let npartitions = 3;
        let config = DatasetConfig {
            src: DatasetSrc::Inline {
                samples: samples.into(),
                labels: labels.into(),
            },
            x_size,
            y_size,
        };

        let expected_specs = [
            DatasetSpec {
                x_size_bytes: (2 * x_size.get() * size_of::<f32>()) as u64,
                y_size_bytes: (2 * y_size.get() * size_of::<f32>()) as u64,
                x_size,
                y_size,
            },
            DatasetSpec {
                x_size_bytes: (x_size.get() * size_of::<f32>()) as u64,
                y_size_bytes: (y_size.get() * size_of::<f32>()) as u64,
                x_size,
                y_size,
            },
            DatasetSpec {
                x_size_bytes: (x_size.get() * size_of::<f32>()) as u64,
                y_size_bytes: (y_size.get() * size_of::<f32>()) as u64,
                x_size,
                y_size,
            },
        ];
        let expected_partitions = [
            Partition::Inline {
                samples: &samples[..2 * x_size.get()],
                labels: &labels[..2 * y_size.get()],
            },
            Partition::Inline {
                samples: &samples[4..6],
                labels: &labels[2..3],
            },
            Partition::Inline {
                samples: &samples[6..],
                labels: &labels[3..],
            },
        ];

        let adapter = Adapter::new();
        let (specs, partitions) = adapter.adapt_dataset(&config, npartitions).unwrap();

        assert_eq!(specs, expected_specs);
        assert_eq!(partitions, expected_partitions);
    }

    #[test]
    fn test_adapter_adapt_layers_returns_exepcted_layers() {
        let cfg = ModelConfig {
            layers: vec![
                LayerConfig::Conv {
                    input_dim: (
                        NonZeroUsize::new(1).unwrap(),
                        NonZeroUsize::new(3).unwrap(),
                        NonZeroUsize::new(3).unwrap(),
                    ),
                    kernel_dim: (
                        NonZeroUsize::new(1).unwrap(),
                        NonZeroUsize::new(1).unwrap(),
                        NonZeroUsize::new(2).unwrap(),
                    ),
                    stride: NonZeroUsize::new(1).unwrap(),
                    padding: 0,
                    init: ParamGenConfig::Kaiming,
                    act_fn: None,
                },
                LayerConfig::Dense {
                    output_size: NonZeroUsize::new(4).unwrap(),
                    init: ParamGenConfig::Kaiming,
                    act_fn: Some(ActFnConfig::Sigmoid { amp: 1.0 }),
                },
            ],
        };
        let input_size = NonZeroUsize::new(9).unwrap();

        let expected_specs = vec![
            LayerSpec::Conv {
                input_dim: (1, 3, 3),
                kernel_dim: (1, 1, 2),
                stride: 1,
                padding: 0,
                act_fn: None,
            },
            LayerSpec::Dense {
                dim: (4, 4),
                act_fn: Some(ActFnSpec::Sigmoid { amp: 1.0 }),
            },
        ];

        let adapter = Adapter::new();
        let (got_specs, _) = adapter.adapt_layers(&cfg, input_size);

        assert_eq!(got_specs, expected_specs);
    }
}
