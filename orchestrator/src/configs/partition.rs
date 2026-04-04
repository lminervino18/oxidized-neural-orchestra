use std::path::PathBuf;

/// The metadata of a dataset partition.
#[derive(PartialEq, Debug)]
pub enum Partition<'a> {
    Inline {
        samples: &'a [f32],
        labels: &'a [f32],
    },
    Local {
        path: &'a PathBuf,
        offset: u64,
        size: u64,
    },
}
