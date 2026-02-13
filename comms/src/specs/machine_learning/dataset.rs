use serde::{Deserialize, Serialize};

/// The specification for the `Dataset`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetSpec {
    pub size: usize,
    pub x_size: usize,
    pub y_size: usize,

    pub first: ChunkSpec,
}

/// The specification for a `Dataset` chunk.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChunkSpec {
    pub offset: usize,
    pub last: bool,

    pub data: Vec<f32>,
}
