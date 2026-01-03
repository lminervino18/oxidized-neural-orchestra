//! DataLoader (placeholder for now).
//! Next step: batching, shard iteration, optional prefetch.

use super::dataset::{InMemoryDataset, Sample};

#[derive(Debug, Clone)]
pub struct DataLoader {
    ds: InMemoryDataset,
    idx: usize,
    end: usize,
}

impl DataLoader {
    pub fn new(ds: InMemoryDataset, start: usize, end: usize) -> Self {
        Self { ds, idx: start, end }
    }

    pub fn next(&mut self) -> Option<Sample> {
        if self.idx >= self.end {
            return None;
        }
        let s = self.ds.get(self.idx);
        self.idx += 1;
        Some(s)
    }
}
