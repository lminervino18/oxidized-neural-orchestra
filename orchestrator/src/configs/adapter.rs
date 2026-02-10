use std::{io, net::SocketAddr};

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

pub fn to_specs_adapter(
    model: ModelConfig,
    training: TrainingConfig,
) -> io::Result<(
    Vec<SocketAddr>,
    WorkerSpec,
    Option<(SocketAddr, ServerSpec)>,
)> {
    let worker_spec = resolve_worker(&model, &training)?;
    let server_spec = resolve_server(&model, &training);
    Ok((training.worker_ips, worker_spec, server_spec))
}

fn resolve_worker(model: &ModelConfig, training: &TrainingConfig) -> io::Result<WorkerSpec> {
    let trainer = resolve_trainer(model, training)?;
    let (algorithm, _) = resolve_algorithm_synchronizer_store(&training.algorithm);

    let worker = WorkerSpec {
        worker_id: 0,
        trainer,
        algorithm,
    };

    Ok(worker)
}

fn resolve_server(
    model: &ModelConfig,
    training: &TrainingConfig,
) -> Option<(SocketAddr, ServerSpec)> {
    let (_, Some((server_ip, synchronizer, store))) =
        resolve_algorithm_synchronizer_store(&training.algorithm)
    else {
        return None;
    };

    let (_, param_gen) = resolve_model_param_gen(model);
    let optimizer = resolve_optimizer(training.optimizer);

    let server = ServerSpec {
        nworkers: training.worker_ips.len(),
        param_gen,
        optimizer,
        synchronizer,
        store,
        seed: training.seed,
    };

    Some((server_ip, server))
}

fn resolve_algorithm_synchronizer_store(
    algorithm: &AlgorithmConfig,
) -> (
    AlgorithmSpec,
    Option<(SocketAddr, SynchronizerSpec, StoreSpec)>,
) {
    match algorithm {
        AlgorithmConfig::ParameterServer {
            server_ips,
            synchronizer,
            store,
        } => {
            let algorithm_spec = AlgorithmSpec::ParameterServer {
                server_ip: server_ips[0], // TODO: eventualmente esto tiene que pasar todas las ips
            };
            let synchronizer_spec = resolve_synchronizer(*synchronizer);
            let store_spec = resolve_store(*store);

            (
                algorithm_spec,
                Some((server_ips[0], synchronizer_spec, store_spec)),
            )
        }
    }
}

fn resolve_store(store: StoreConfig) -> StoreSpec {
    match store {
        StoreConfig::Blocking { shard_size } => StoreSpec::Blocking { shard_size },
        StoreConfig::Wild { shard_size } => StoreSpec::Wild { shard_size },
    }
}

fn resolve_synchronizer(synchronizer: SynchronizerConfig) -> SynchronizerSpec {
    match synchronizer {
        SynchronizerConfig::Barrier { barrier_size } => SynchronizerSpec::Barrier { barrier_size },
        SynchronizerConfig::NonBlocking => SynchronizerSpec::NonBlocking,
    }
}

fn resolve_trainer(model: &ModelConfig, training: &TrainingConfig) -> io::Result<TrainerSpec> {
    let (model_spec, _) = resolve_model_param_gen(model);
    let optimizer = resolve_optimizer(training.optimizer);
    let dataset = resolve_dataset(&training.dataset)?;
    let loss_fn = resolve_loss_fn(training.loss_fn);

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

fn resolve_loss_fn(loss_fn: LossFnConfig) -> LossFnSpec {
    match loss_fn {
        LossFnConfig::Mse => LossFnSpec::Mse,
    }
}

fn resolve_dataset(dataset: &DatasetConfig) -> io::Result<DatasetSpec> {
    // deberia ir a buscar el archivo / bajarlo de internet / etc
    todo!()
}

fn resolve_optimizer(optimizer: OptimizerConfig) -> OptimizerSpec {
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

fn resolve_model_param_gen(model: &ModelConfig) -> (ModelSpec, ParamGenSpec) {
    match model {
        ModelConfig::Sequential { layers } => {
            let (layers, param_gens): (Vec<_>, Vec<ParamGenConfig>) =
                layers.into_iter().map(|layer| resolve_layer(layer)).unzip();

            let param_gen = resolve_param_gens(param_gens);
            (ModelSpec::Sequential { layers }, param_gen)
        }
    }
}

fn resolve_param_gens(param_gens: Vec<ParamGenConfig>) -> ParamGenSpec {
    let inner_param_gens = param_gens
        .into_iter()
        .map(|param_gen| match param_gen {
            ParamGenConfig::Const { value, limit } => ParamGenSpec::Const { value, limit },
            ParamGenConfig::Uniform { low, high, limit } => ParamGenSpec::Rand {
                distribution: DistributionSpec::Uniform { low, high },
                limit,
            },
            ParamGenConfig::UniformInclusive { low, high, limit } => ParamGenSpec::Rand {
                distribution: DistributionSpec::UniformInclusive { low, high },
                limit,
            },
            ParamGenConfig::XavierUniform {
                fan_in,
                fan_out,
                limit,
            } => ParamGenSpec::Rand {
                distribution: DistributionSpec::XavierUniform { fan_in, fan_out },
                limit,
            },
            ParamGenConfig::LecunUniform { fan_in, limit } => ParamGenSpec::Rand {
                distribution: DistributionSpec::LecunUniform { fan_in },
                limit,
            },
            ParamGenConfig::Normal {
                mean,
                std_dev,
                limit,
            } => ParamGenSpec::Rand {
                distribution: DistributionSpec::Normal { mean, std_dev },
                limit,
            },
            ParamGenConfig::Kaiming { fan_in, limit } => ParamGenSpec::Rand {
                distribution: DistributionSpec::Kaiming { fan_in },
                limit,
            },
            ParamGenConfig::Xavier {
                fan_in,
                fan_out,
                limit,
            } => ParamGenSpec::Rand {
                distribution: DistributionSpec::Xavier { fan_in, fan_out },
                limit,
            },
            ParamGenConfig::Lecun { fan_in, limit } => ParamGenSpec::Rand {
                distribution: DistributionSpec::Lecun { fan_in },
                limit,
            },
        })
        .collect();

    ParamGenSpec::Chained {
        specs: inner_param_gens,
    }
}

fn resolve_layer(layer: &LayerConfig) -> (LayerSpec, ParamGenConfig) {
    match layer {
        LayerConfig::Dense { dim, init, act_fn } => {
            let act_fn = resolve_act_fn(act_fn.as_ref());
            (LayerSpec::Dense { dim: *dim, act_fn }, *init)
        }
    }
}

fn resolve_act_fn(act_fn: Option<&ActFnConfig>) -> Option<ActFnSpec> {
    let act_fn_spec = match act_fn? {
        ActFnConfig::Sigmoid { amp } => ActFnSpec::Sigmoid { amp: *amp },
    };

    Some(act_fn_spec)
}
