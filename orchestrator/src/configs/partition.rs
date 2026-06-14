use std::path::PathBuf;

/// The metadata of a dataset partition.
#[derive(PartialEq, Debug, Clone)]
pub enum Partition<'a> {
    Inline {
        samples: &'a [f32],
        labels: &'a [f32],
    },
    Local {
        samples_path: PathBuf,
        labels_path: PathBuf,
        samples_offset: u64,
        labels_offset: u64,
        samples_size: u64,
        labels_size: u64,
    },
}
