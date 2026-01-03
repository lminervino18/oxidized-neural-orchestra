use std::num::NonZeroUsize;
use std::ops::Range;

/// Splits `total` samples among `num_workers` and returns the shard for `worker_id`.
///
/// Properties:
/// - Ranges are contiguous, disjoint and cover `[0..total)`.
/// - Sizes differ by at most 1 (balanced partition).
pub fn shard_range(total: usize, worker_id: usize, num_workers: usize) -> Range<usize> {
    assert!(num_workers > 0);
    assert!(worker_id < num_workers);

    let base = total / num_workers;
    let rem = total % num_workers;

    let start = worker_id * base + worker_id.min(rem);
    let extra = if worker_id < rem { 1 } else { 0 };
    let end = start + base + extra;

    start..end
}

/// Shard specification for a worker (stable API).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSpec {
    pub worker_id: usize,
    pub num_workers: NonZeroUsize,
}

impl ShardSpec {
    pub fn new(worker_id: usize, num_workers: NonZeroUsize) -> Self {
        assert!(worker_id < num_workers.get(), "worker_id out of range");
        Self { worker_id, num_workers }
    }

    #[inline]
    pub fn range(self, total: usize) -> Range<usize> {
        shard_range(total, self.worker_id, self.num_workers.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_range_balanced() {
        // total 10, workers 3 => sizes 4,3,3
        assert_eq!(shard_range(10, 0, 3), 0..4);
        assert_eq!(shard_range(10, 1, 3), 4..7);
        assert_eq!(shard_range(10, 2, 3), 7..10);
    }

    #[test]
    fn shard_spec_range_matches_function() {
        let spec = ShardSpec::new(1, NonZeroUsize::new(3).unwrap());
        assert_eq!(spec.range(10), 4..7);
    }
}
