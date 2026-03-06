use std::path::PathBuf;

/// The metadata of a dataset partition.
pub struct Partition {
    pub path: PathBuf,
    pub offset: u64,
    pub size: u64,
}
