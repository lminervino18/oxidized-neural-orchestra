use std::ops::Range;

use super::dataset::{BatchRef, InMemoryDataset};
use super::shard::ShardSpec;

/// Batch described only by indices into the dataset (no slices, no copies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchSpec {
    pub start: usize,
    pub end: usize,
}

impl BatchSpec {
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Shard-aware DataLoader producing borrowed batches (zero-copy) or index specs.
#[derive(Debug, Clone)]
pub struct DataLoader {
    dataset: InMemoryDataset,
    shard: ShardSpec,
    shard_range: Range<usize>,
    batch_size: usize,
    cursor: usize, // absolute index
}

impl DataLoader {
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
    pub fn dataset(&self) -> &InMemoryDataset {
        &self.dataset
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
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[inline]
    pub fn reset(&mut self) {
        self.cursor = self.shard_range.start;
    }

    /// Returns the next batch as a pure index range (no slices).
    pub fn next_spec(&mut self) -> Option<BatchSpec> {
        if self.cursor >= self.shard_range.end {
            return None;
        }

        let end = (self.cursor + self.batch_size).min(self.shard_range.end);
        let spec = BatchSpec {
            start: self.cursor,
            end,
        };
        self.cursor = end;
        Some(spec)
    }

    /// Returns the next borrowed batch view (zero-copy slices).
    ///
    /// Note: This is a convenience wrapper over `next_spec()`.
    pub fn next_batch(&mut self) -> Option<BatchRef<'_>> {
        let spec = self.next_spec()?;
        let xs = &self.dataset.xs()[spec.start..spec.end];
        let ys = &self.dataset.ys()[spec.start..spec.end];
        Some(BatchRef { xs, ys })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use crate::data::{dataset::InMemoryDataset, shard::ShardSpec};

    #[test]
    fn dataloader_next_spec_respects_shard_and_batch_size() {
        let ds = InMemoryDataset::new(
            (0..10).map(|i| i as f32).collect(),
            (0..10).map(|i| (i as f32) + 100.0).collect(),
        );

        // total=10, workers=3 => ranges: 0..4, 4..7, 7..10
        let shard = ShardSpec::new(1, NonZeroUsize::new(3).unwrap()); // 4..7
        let mut dl = DataLoader::new(ds, shard, 2);

        assert_eq!(dl.shard_range(), 4..7);

        assert_eq!(dl.next_spec(), Some(BatchSpec { start: 4, end: 6 }));
        assert_eq!(dl.next_spec(), Some(BatchSpec { start: 6, end: 7 }));
        assert_eq!(dl.next_spec(), None);

        dl.reset();
        assert_eq!(dl.next_spec(), Some(BatchSpec { start: 4, end: 6 }));
    }

    #[test]
    fn dataloader_next_batch_matches_expected_slices() {
        let ds = InMemoryDataset::new(
            (0..6).map(|i| i as f32).collect(),
            (0..6).map(|i| (i as f32) + 10.0).collect(),
        );

        let shard = ShardSpec::new(0, NonZeroUsize::new(1).unwrap()); // 0..6
        let mut dl = DataLoader::new(ds, shard, 4);

        let b1 = dl.next_batch().unwrap();
        assert_eq!(b1.xs, &[0., 1., 2., 3.]);
        assert_eq!(b1.ys, &[10., 11., 12., 13.]);

        let b2 = dl.next_batch().unwrap();
        assert_eq!(b2.xs, &[4., 5.]);
        assert_eq!(b2.ys, &[14., 15.]);

        assert!(dl.next_batch().is_none());
    }

    #[test]
    fn next_batch_is_consistent_with_next_spec() {
        let ds = InMemoryDataset::new(
            (0..7).map(|i| i as f32).collect(),
            (0..7).map(|i| (i as f32) + 50.0).collect(),
        );

        let shard = ShardSpec::new(0, NonZeroUsize::new(1).unwrap()); // 0..7

        let mut dl1 = DataLoader::new(ds.clone(), shard, 3);
        let mut dl2 = DataLoader::new(ds, shard, 3);

        loop {
            let s = dl1.next_spec();
            let b = dl2.next_batch();

            match (s, b) {
                (None, None) => break,
                (Some(spec), Some(batch)) => {
                    let xs = dl1.dataset().xs();
                    let ys = dl1.dataset().ys();
                    assert_eq!(batch.xs, &xs[spec.start..spec.end]);
                    assert_eq!(batch.ys, &ys[spec.start..spec.end]);
                }
                other => panic!("mismatch between next_spec and next_batch: {other:?}"),
            }
        }
    }
}
