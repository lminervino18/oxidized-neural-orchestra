use serde::{Deserialize, Serialize};

/// The specification for the `Dataset`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetSpec<'a> {
    pub size: usize,
    pub x_size: usize,
    pub y_size: usize,

    #[serde(borrow)]
    pub first: ChunkSpec<'a>,
}

/// The specification for a `Dataset` chunk.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChunkSpec<'a> {
    pub offset: usize,
    pub last: bool,

    pub data: &'a [u8],
}
