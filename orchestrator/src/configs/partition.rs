use std::path::PathBuf;

/// The metadata of a dataset partition.
pub enum Partition<'a> {
    Inline {
        data: &'a [f32],
    },
    Local {
        path: PathBuf,
        offset: u64,
        size: u64,
    },
}
