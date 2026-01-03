use std::ops::Range;

/// Splits `total` samples among `num_workers` and returns the shard for `worker_id`.
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
}
