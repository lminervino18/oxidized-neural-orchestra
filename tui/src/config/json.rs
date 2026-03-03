use std::num::NonZeroUsize;

use orchestrator::configs::{
    ActFnConfig, AlgorithmConfig, DatasetConfig, LayerConfig, LossFnConfig, ModelConfig,
    OptimizerConfig, ParamGenConfig, StoreConfig, SynchronizerConfig, TrainingConfig,
};

/// Parsed model architecture from `model.json`.
#[derive(Debug)]
pub struct ModelJson {
    pub config: ModelConfig,
}

/// Parsed training configuration from `training.json`.
#[derive(Debug)]
pub struct TrainingJson {
    pub config: TrainingConfig<String>,
    pub worker_count: usize,
}

/// Loads and parses a [`ModelConfig`] from a JSON file.
///
/// # Args
/// * `path` - Path to the model JSON file.
///
/// # Returns
/// A parsed `ModelJson` on success.
///
/// # Errors
/// Returns a human-readable string if the file cannot be read or parsed.
pub fn load_model(path: &str) -> Result<ModelJson, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let val: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("invalid JSON: {e}"))?;

    let layers = val["layers"]
        .as_array()
        .ok_or("missing field: layers")?
        .iter()
        .enumerate()
        .map(|(i, l)| parse_layer(l, i))
        .collect::<Result<Vec<_>, _>>()?;

    if layers.is_empty() {
        return Err("layers must not be empty".into());
    }

    Ok(ModelJson {
        config: ModelConfig::Sequential { layers },
    })
}

/// Loads and parses a [`TrainingConfig`] from a JSON file.
///
/// # Args
/// * `path` - Path to the training JSON file.
///
/// # Returns
/// A parsed `TrainingJson` on success.
///
/// # Errors
/// Returns a human-readable string if the file cannot be read, parsed,
/// or any required field is missing or invalid.
pub fn load_training(path: &str) -> Result<TrainingJson, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let val: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("invalid JSON: {e}"))?;

    let str_arr = |key: &str| -> Result<Vec<String>, String> {
        val[key]
            .as_array()
            .ok_or_else(|| format!("missing field: {key}"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("{key} must contain strings"))
            })
            .collect()
    };

    let req_usize = |key: &str| -> Result<usize, String> {
        val[key]
            .as_u64()
            .ok_or_else(|| format!("missing field: {key}"))
            .map(|v| v as usize)
    };

    let req_nz = |key: &str| -> Result<NonZeroUsize, String> {
        let v = req_usize(key)?;
        NonZeroUsize::new(v).ok_or_else(|| format!("{key} must be greater than zero"))
    };

    let worker_addrs = str_arr("worker_addrs")?;
    let server_addrs = str_arr("server_addrs")?;

    if worker_addrs.is_empty() {
        return Err("worker_addrs must not be empty".into());
    }
    if server_addrs.is_empty() {
        return Err("server_addrs must not be empty".into());
    }

    let synchronizer = parse_synchronizer(&val)?;
    let store = parse_store(&val)?;
    let optimizer = parse_optimizer(&val)?;
    let dataset = parse_dataset(&val["dataset"])?;
    let worker_count = worker_addrs.len();

    Ok(TrainingJson {
        worker_count,
        config: TrainingConfig {
            worker_addrs,
            algorithm: AlgorithmConfig::ParameterServer {
                server_addrs,
                synchronizer,
                store,
            },
            dataset,
            optimizer,
            loss_fn: LossFnConfig::Mse,
            batch_size: req_nz("batch_size")?,
            max_epochs: req_nz("max_epochs")?,
            offline_epochs: val["offline_epochs"]
                .as_u64()
                .ok_or("missing field: offline_epochs")? as usize,
            seed: val["seed"].as_u64(),
        },
    })
}

fn parse_synchronizer(val: &serde_json::Value) -> Result<SynchronizerConfig, String> {
    match val["synchronizer"]
        .as_str()
        .ok_or("missing field: synchronizer")?
    {
        "barrier" => {
            let barrier_size = val["barrier_size"]
                .as_u64()
                .ok_or("missing field: barrier_size")? as usize;
            Ok(SynchronizerConfig::Barrier { barrier_size })
        }
        "non_blocking" => Ok(SynchronizerConfig::NonBlocking),
        other => Err(format!("unknown synchronizer: {other}")),
    }
}

fn parse_store(val: &serde_json::Value) -> Result<StoreConfig, String> {
    match val["store"].as_str().ok_or("missing field: store")? {
        "blocking" => Ok(StoreConfig::Blocking),
        "wild" => Ok(StoreConfig::Wild),
        other => Err(format!("unknown store: {other}")),
    }
}

fn parse_optimizer(val: &serde_json::Value) -> Result<OptimizerConfig, String> {
    let req_f32 = |key: &str| -> Result<f32, String> {
        val[key]
            .as_f64()
            .ok_or_else(|| format!("missing field: {key}"))
            .map(|v| v as f32)
    };

    match val["optimizer"]
        .as_str()
        .ok_or("missing field: optimizer")?
    {
        "gradient_descent" => Ok(OptimizerConfig::GradientDescent { lr: req_f32("lr")? }),
        "adam" => Ok(OptimizerConfig::Adam {
            lr: req_f32("lr")?,
            b1: req_f32("b1")?,
            b2: req_f32("b2")?,
            eps: req_f32("eps")?,
        }),
        "gradient_descent_with_momentum" => Ok(OptimizerConfig::GradientDescentWithMomentum {
            lr: req_f32("lr")?,
            mu: req_f32("mu")?,
        }),
        other => Err(format!("unknown optimizer: {other}")),
    }
}

fn parse_dataset(val: &serde_json::Value) -> Result<DatasetConfig, String> {
    let csv_path = val["path"].as_str().ok_or("missing field: dataset.path")?;
    let x_size = val["x_size"]
        .as_u64()
        .ok_or("missing field: dataset.x_size")? as usize;
    let y_size = val["y_size"]
        .as_u64()
        .ok_or("missing field: dataset.y_size")? as usize;

    let row_size = x_size + y_size;
    if row_size == 0 {
        return Err("dataset.x_size + dataset.y_size must be greater than 0".into());
    }

    let content = std::fs::read_to_string(csv_path)
        .map_err(|e| format!("cannot read dataset '{csv_path}': {e}"))?;

    let mut data = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let values: Vec<f32> = line
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<f32>()
                    .map_err(|_| format!("dataset line {i}: cannot parse '{v}' as f32"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if values.len() != row_size {
            return Err(format!(
                "dataset line {i}: expected {row_size} values (x_size={x_size} + y_size={y_size}), got {}",
                values.len()
            ));
        }
        data.extend(values);
    }

    if data.is_empty() {
        return Err("dataset is empty".into());
    }

    Ok(DatasetConfig::Inline {
        data,
        x_size,
        y_size,
    })
}

fn parse_layer(l: &serde_json::Value, idx: usize) -> Result<LayerConfig, String> {
    let ctx = |f: &str| format!("layer {idx}: {f}");

    let n = l["n"].as_u64().ok_or_else(|| ctx("missing field: n"))? as usize;
    let m = l["m"].as_u64().ok_or_else(|| ctx("missing field: m"))? as usize;

    let init = parse_param_gen(l, idx)?;
    let act_fn = parse_act_fn(l, idx)?;

    Ok(LayerConfig::Dense {
        dim: (n, m),
        init,
        act_fn,
    })
}

fn parse_param_gen(l: &serde_json::Value, idx: usize) -> Result<ParamGenConfig, String> {
    let ctx = |f: &str| format!("layer {idx}: {f}");

    match l["init"]
        .as_str()
        .ok_or_else(|| ctx("missing field: init"))?
    {
        "const" => {
            let value = l["init_value"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: init_value"))? as f32;
            Ok(ParamGenConfig::Const { value })
        }
        "uniform" => Ok(ParamGenConfig::Uniform {
            low: l["init_low"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: init_low"))? as f32,
            high: l["init_high"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: init_high"))? as f32,
        }),
        "uniform_inclusive" => Ok(ParamGenConfig::UniformInclusive {
            low: l["init_low"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: init_low"))? as f32,
            high: l["init_high"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: init_high"))? as f32,
        }),
        "xavier_uniform" => Ok(ParamGenConfig::XavierUniform),
        "lecun_uniform" => Ok(ParamGenConfig::LecunUniform),
        "normal" => Ok(ParamGenConfig::Normal {
            mean: l["init_mean"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: init_mean"))? as f32,
            std_dev: l["init_std"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: init_std"))? as f32,
        }),
        "kaiming" => Ok(ParamGenConfig::Kaiming),
        "xavier" => Ok(ParamGenConfig::Xavier),
        "lecun" => Ok(ParamGenConfig::Lecun),
        other => Err(ctx(&format!("unknown init: {other}"))),
    }
}

fn parse_act_fn(l: &serde_json::Value, idx: usize) -> Result<Option<ActFnConfig>, String> {
    let ctx = |f: &str| format!("layer {idx}: {f}");

    match l["act_fn"].as_str() {
        None => Ok(None),
        Some("sigmoid") => {
            let amp = l["act_amp"]
                .as_f64()
                .ok_or_else(|| ctx("missing field: act_amp"))? as f32;
            Ok(Some(ActFnConfig::Sigmoid { amp }))
        }
        Some(other) => Err(ctx(&format!("unknown act_fn: {other}"))),
    }
}
