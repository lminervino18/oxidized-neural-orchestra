/// A single Dense layer parsed from JSON.
#[derive(Debug, Clone)]
pub struct LayerDraft {
    pub n: usize,
    pub m: usize,
    pub init: InitKind,
    pub init_value: f32,
    pub init_low: f32,
    pub init_high: f32,
    pub init_mean: f32,
    pub init_std: f32,
    pub act_fn: ActFnKind,
    pub act_amp: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitKind {
    Const,
    Uniform,
    UniformInclusive,
    XavierUniform,
    LecunUniform,
    Normal,
    Kaiming,
    Xavier,
    Lecun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActFnKind {
    None,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerKind {
    GradientDescent,
    Adam,
    GradientDescentWithMomentum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynchronizerKind {
    Barrier,
    NonBlocking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    Blocking,
    Wild,
}

/// Dataset loaded from a CSV file.
#[derive(Debug, Clone)]
pub struct DatasetDraft {
    pub data: Vec<f32>,
    pub x_size: usize,
    pub y_size: usize,
}

/// Model architecture parsed from model.json.
#[derive(Debug, Clone)]
pub struct ModelDraft {
    pub layers: Vec<LayerDraft>,
}

/// Training config parsed from training.json.
#[derive(Debug, Clone)]
pub struct TrainingDraft {
    pub worker_addrs: Vec<String>,
    pub server_addrs: Vec<String>,
    pub synchronizer: SynchronizerKind,
    pub barrier_size: usize,
    pub store: StoreKind,
    pub shard_size: usize,
    pub max_epochs: usize,
    pub offline_epochs: usize,
    pub batch_size: usize,
    pub seed: Option<u64>,
    pub optimizer: OptimizerKind,
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub mu: f32,
    pub dataset: DatasetDraft,
}
