use std::num::NonZeroUsize;

use orchestrator::configs::{
    ActFnConfig, AlgorithmConfig, DatasetConfig, LayerConfig, LossFnConfig, ModelConfig,
    OptimizerConfig, ParamGenConfig, StoreConfig, SynchronizerConfig, TrainingConfig,
};

use super::model::{
    ActFnKind, InitKind, ModelDraft, OptimizerKind, StoreKind, SynchronizerKind, TrainingDraft,
};

/// Converts [`ModelDraft`] and [`TrainingDraft`] into orchestrator config types.
///
/// # Errors
/// Returns a human-readable error if any value is invalid.
pub fn build(
    model: &ModelDraft,
    training: &TrainingDraft,
) -> Result<(ModelConfig, TrainingConfig<String>), String> {
    Ok((build_model(model)?, build_training(training)?))
}

fn build_model(d: &ModelDraft) -> Result<ModelConfig, String> {
    if d.layers.is_empty() {
        return Err("model must have at least one layer".into());
    }

    let layers = d
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let init = match l.init {
                InitKind::Const => ParamGenConfig::Const { value: l.init_value },
                InitKind::Uniform => ParamGenConfig::Uniform {
                    low: l.init_low,
                    high: l.init_high,
                },
                InitKind::UniformInclusive => ParamGenConfig::UniformInclusive {
                    low: l.init_low,
                    high: l.init_high,
                },
                InitKind::XavierUniform => ParamGenConfig::XavierUniform {},
                InitKind::LecunUniform => ParamGenConfig::LecunUniform,
                InitKind::Normal => ParamGenConfig::Normal {
                    mean: l.init_mean,
                    std_dev: l.init_std,
                },
                InitKind::Kaiming => ParamGenConfig::Kaiming,
                InitKind::Xavier => ParamGenConfig::Xavier,
                InitKind::Lecun => ParamGenConfig::Lecun,
            };

            let act_fn = match l.act_fn {
                ActFnKind::None => None,
                ActFnKind::Sigmoid => Some(ActFnConfig::Sigmoid { amp: l.act_amp }),
            };

            Ok(LayerConfig::Dense {
                dim: (l.n, l.m),
                init,
                act_fn,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(ModelConfig::Sequential { layers })
}

fn build_training(d: &TrainingDraft) -> Result<TrainingConfig<String>, String> {
    let nz = |v: usize, name: &str| {
        NonZeroUsize::new(v).ok_or_else(|| format!("{name} must be greater than zero"))
    };

    let synchronizer = match d.synchronizer {
        SynchronizerKind::Barrier => SynchronizerConfig::Barrier {
            barrier_size: d.barrier_size,
        },
        SynchronizerKind::NonBlocking => SynchronizerConfig::NonBlocking,
    };

    let store = match d.store {
        StoreKind::Blocking => StoreConfig::Blocking {
            shard_size: nz(d.shard_size, "shard_size")?,
        },
        StoreKind::Wild => StoreConfig::Wild {
            shard_size: nz(d.shard_size, "shard_size")?,
        },
    };

    let optimizer = match d.optimizer {
        OptimizerKind::GradientDescent => OptimizerConfig::GradientDescent { lr: d.lr },
        OptimizerKind::Adam => OptimizerConfig::Adam {
            lr: d.lr,
            b1: d.b1,
            b2: d.b2,
            eps: d.eps,
        },
        OptimizerKind::GradientDescentWithMomentum => {
            OptimizerConfig::GradientDescentWithMomentum { lr: d.lr, mu: d.mu }
        }
    };

    Ok(TrainingConfig {
        worker_addrs: d.worker_addrs.clone(),
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs: d.server_addrs.clone(),
            synchronizer,
            store,
        },
        dataset: DatasetConfig::Inline {
            data: vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
            x_size: 1,
            y_size: 1,
        },
        optimizer,
        loss_fn: LossFnConfig::Mse,
        batch_size: nz(d.batch_size, "batch_size")?,
        max_epochs: nz(d.max_epochs, "max_epochs")?,
        offline_epochs: d.offline_epochs,
        seed: d.seed,
    })
}