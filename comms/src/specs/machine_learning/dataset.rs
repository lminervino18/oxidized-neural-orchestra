use serde::{Deserialize, Serialize};

/// The specification for the `Dataset`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetSpec {
    pub data: Vec<f32>,
    pub x_size: usize,
    pub y_size: usize,
}
