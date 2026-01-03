use std::ops::Range;

use super::dataset::{Batch, InMemoryDataset};
use super::shard::ShardSpec;

/// Shard-aware DataLoader producing owned batches.
///
/// Step semantics (for current protocol): **one batch per step**.
/// This loader is intentionally minimal and deterministic.
/// Later:
/// - prefetch
/// - shuffle
/// - borrowed/zero-copy batches
#[derive(Debug, Clone)]
pub struct DataLoader {
    dataset: InMemoryDataset,
    shard: ShardSpec,
    shard_range: Range<usize>,
    batch_size: usize,
    cursor: usize, // absolute index in dataset
}

impl DataLoader {
    /// Creates a new DataLoader for a worker shard.
    ///
    /// # Arguments
    /// * `dataset` - In-memory dataset
    /// * `shard` - which slice of the dataset belongs to this worker
    /// * `batch_size` - number of samples per step (must be > 0)
    pub fn new(dataset: InMemoryDataset, shard: ShardSpec, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");

        let total = dataset.len();
        let shard_range = shard.range(total);
        let cursor = shard_range.start;

        Self {
            dataset,
            shard,
            shard_range,
            batch_size,
            cursor,
        }
    }

    #[inline]
    pub fn shard(&self) -> ShardSpec {
        self.shard
    }

    #[inline]
    pub fn shard_range(&self) -> Range<usize> {
        self.shard_range.clone()
    }

    /// Resets cursor to the beginning of this worker shard.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = self.shard_range.start;
    }

    /// Returns the next batch for this worker shard, or None if exhausted.
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.cursor >= self.shard_range.end {
            return None;
        }

        let end = (self.cursor + self.batch_size).min(self.shard_range.end);

        let mut xs = Vec::with_capacity(end - self.cursor);
        let mut ys = Vec::with_capacity(end - self.cursor);

        for i in self.cursor..end {
            let s = self.dataset.get(i);
            xs.push(s.x);
            ys.push(s.y);
        }

        self.cursor = end;
        Some(Batch::new(xs, ys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn dataloader_respects_shard_range_and_batch_size() {
        // total=10, workers=3 -> ranges: [0..4], [4..7], [7..10]
        let ds = InMemoryDataset::new(
            (0..10).map(|i| i as f32).collect(),
            (0..10).map(|i| (i as f32) + 100.0).collect(),
        );

        let shard = ShardSpec::new(1, NonZeroUsize::new(3).unwrap()); // 4..7
        let mut dl = DataLoader::new(ds, shard, 2);

        assert_eq!(dl.shard_range(), 4..7);

        let b1 = dl.next_batch().unwrap();
        assert_eq!(b1.xs, vec![4.0, 5.0]);
        assert_eq!(b1.ys, vec![104.0, 105.0]);

        let b2 = dl.next_batch().unwrap();
        assert_eq!(b2.xs, vec![6.0]);
        assert_eq!(b2.ys, vec![106.0]);

        assert!(dl.next_batch().is_none());

        dl.reset();
        let b3 = dl.next_batch().unwrap();
        assert_eq!(b3.xs, vec![4.0, 5.0]);
    }
}
