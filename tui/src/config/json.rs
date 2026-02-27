use super::model::{
    ActFnKind, DatasetDraft, InitKind, LayerDraft, ModelDraft, OptimizerKind, StoreKind,
    SynchronizerKind, TrainingDraft,
};

/// Loads a [`ModelDraft`] from a JSON file.
///
/// # Errors
/// Returns a human-readable string if the file cannot be read or parsed.
pub fn load_model(path: &str) -> Result<ModelDraft, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;

    let val: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("invalid JSON: {e}"))?;

    let layers = val["layers"]
        .as_array()
        .ok_or("missing layers array")?
        .iter()
        .enumerate()
        .map(|(i, l)| parse_layer(l, i))
        .collect::<Result<Vec<_>, _>>()?;

    if layers.is_empty() {
        return Err("layers must not be empty".into());
    }

    Ok(ModelDraft { layers })
}

/// Loads a [`TrainingDraft`] from a JSON file, reading the dataset from the CSV path inside.
///
/// # Errors
/// Returns a human-readable string if the file cannot be read or parsed.
pub fn load_training(path: &str) -> Result<TrainingDraft, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;

    let val: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("invalid JSON: {e}"))?;

    let str_arr = |key: &str| -> Result<Vec<String>, String> {
        val[key]
            .as_array()
            .ok_or_else(|| format!("missing array: {key}"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("{key} must contain strings"))
            })
            .collect()
    };

    let worker_addrs = str_arr("worker_addrs")?;
    let server_addrs = str_arr("server_addrs")?;

    if worker_addrs.is_empty() {
        return Err("worker_addrs must not be empty".into());
    }
    if server_addrs.is_empty() {
        return Err("server_addrs must not be empty".into());
    }

    let synchronizer = match val["synchronizer"].as_str().unwrap_or("barrier") {
        "barrier" => SynchronizerKind::Barrier,
        "non_blocking" => SynchronizerKind::NonBlocking,
        other => return Err(format!("unknown synchronizer: {other}")),
    };

    let store = match val["store"].as_str().unwrap_or("blocking") {
        "blocking" => StoreKind::Blocking,
        "wild" => StoreKind::Wild,
        other => return Err(format!("unknown store: {other}")),
    };

    let optimizer = match val["optimizer"].as_str().ok_or("missing field: optimizer")? {
        "gradient_descent" => OptimizerKind::GradientDescent,
        "adam" => OptimizerKind::Adam,
        "gradient_descent_with_momentum" => OptimizerKind::GradientDescentWithMomentum,
        other => return Err(format!("unknown optimizer: {other}")),
    };

    let dataset = load_dataset(&val["dataset"])?;

    Ok(TrainingDraft {
        worker_addrs,
        server_addrs,
        synchronizer,
        barrier_size: val["barrier_size"].as_u64().unwrap_or(1) as usize,
        store,
        shard_size: val["shard_size"].as_u64().unwrap_or(128) as usize,
        max_epochs: val["max_epochs"].as_u64().unwrap_or(100) as usize,
        offline_epochs: val["offline_epochs"].as_u64().unwrap_or(0) as usize,
        batch_size: val["batch_size"].as_u64().unwrap_or(32) as usize,
        seed: val["seed"].as_u64(),
        optimizer,
        lr: val["lr"].as_f64().unwrap_or(0.01) as f32,
        b1: val["b1"].as_f64().unwrap_or(0.9) as f32,
        b2: val["b2"].as_f64().unwrap_or(0.999) as f32,
        eps: val["eps"].as_f64().unwrap_or(1e-8) as f32,
        mu: val["mu"].as_f64().unwrap_or(0.9) as f32,
        dataset,
    })
}

fn load_dataset(val: &serde_json::Value) -> Result<DatasetDraft, String> {
    let csv_path = val["path"]
        .as_str()
        .ok_or("missing dataset.path")?;

    let x_size = val["x_size"].as_u64().ok_or("missing dataset.x_size")? as usize;
    let y_size = val["y_size"].as_u64().ok_or("missing dataset.y_size")? as usize;

    let row_size = x_size + y_size;
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

    Ok(DatasetDraft { data, x_size, y_size })
}

fn parse_layer(l: &serde_json::Value, idx: usize) -> Result<LayerDraft, String> {
    let ctx = |f: &str| format!("layer {idx}: {f}");

    let n = l["n"].as_u64().ok_or_else(|| ctx("missing n"))? as usize;
    let m = l["m"].as_u64().ok_or_else(|| ctx("missing m"))? as usize;

    let init = match l["init"].as_str().ok_or_else(|| ctx("missing init"))? {
        "const" => InitKind::Const,
        "uniform" => InitKind::Uniform,
        "uniform_inclusive" => InitKind::UniformInclusive,
        "xavier_uniform" => InitKind::XavierUniform,
        "lecun_uniform" => InitKind::LecunUniform,
        "normal" => InitKind::Normal,
        "kaiming" => InitKind::Kaiming,
        "xavier" => InitKind::Xavier,
        "lecun" => InitKind::Lecun,
        other => return Err(ctx(&format!("unknown init: {other}"))),
    };

    let act_fn = match l["act_fn"].as_str() {
        Some("sigmoid") => ActFnKind::Sigmoid,
        _ => ActFnKind::None,
    };

    Ok(LayerDraft {
        n,
        m,
        init,
        init_value: l["init_value"].as_f64().unwrap_or(0.0) as f32,
        init_low: l["init_low"].as_f64().unwrap_or(0.0) as f32,
        init_high: l["init_high"].as_f64().unwrap_or(1.0) as f32,
        init_mean: l["init_mean"].as_f64().unwrap_or(0.0) as f32,
        init_std: l["init_std"].as_f64().unwrap_or(1.0) as f32,
        act_fn,
        act_amp: l["act_amp"].as_f64().unwrap_or(1.0) as f32,
    })
}