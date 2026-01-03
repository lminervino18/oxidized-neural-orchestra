use std::ops::Range;

use super::dataset::{BatchRef, InMemoryDataset};
use super::shard::ShardSpec;

/// Shard-aware DataLoader producing borrowed batches (zero-copy).
#[derive(Debug, Clone)]
pub struct DataLoader {
    dataset: InMemoryDataset,
    shard: ShardSpec,
    shard_range: Range<usize>,
    batch_size: usize,
    cursor: usize, // absolute index in dataset
}

impl DataLoader {
    pub fn new(dataset: InMemoryDataset, shard: ShardSpec, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");

        let total = dataset.len();
        let shard_range = shard.range(total);
        let cursor = shard_range.start;

        Self { dataset, shard, shard_range, batch_size, cursor }
    }

    #[inline]
    pub fn shard(&self) -> ShardSpec {
        self.shard
    }

    #[inline]
    pub fn shard_range(&self) -> Range<usize> {
        self.shard_range.clone()
    }

    #[inline]
    pub fn reset(&mut self) {
        self.cursor = self.shard_range.start;
    }

    /// Returns the next borrowed batch for this shard, or None if exhausted.
    pub fn next_batch(&mut self) -> Option<BatchRef<'_>> {
        if self.cursor >= self.shard_range.end {
            return None;
        }

        let end = (self.cursor + self.batch_size).min(self.shard_range.end);

        let xs = &self.dataset.xs()[self.cursor..end];
        let ys = &self.dataset.ys()[self.cursor..end];

        self.cursor = end;
        Some(BatchRef { xs, ys })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn dataloader_borrowed_batches_respect_shard_and_batch_size() {
        let ds = InMemoryDataset::new(
            (0..10).map(|i| i as f32).collect(),
            (0..10).map(|i| (i as f32) + 100.0).collect(),
        );

        let shard = ShardSpec::new(1, NonZeroUsize::new(3).unwrap()); // 4..7
        let mut dl = DataLoader::new(ds, shard, 2);

        assert_eq!(dl.shard_range(), 4..7);

        let b1 = dl.next_batch().unwrap();
        assert_eq!(b1.xs, &[4.0, 5.0]);
        assert_eq!(b1.ys, &[104.0, 105.0]);

        let b2 = dl.next_batch().unwrap();
        assert_eq!(b2.xs, &[6.0]);
        assert_eq!(b2.ys, &[106.0]);

        assert!(dl.next_batch().is_none());

        dl.reset();
        let b3 = dl.next_batch().unwrap();
        assert_eq!(b3.xs, &[4.0, 5.0]);
    }
}
