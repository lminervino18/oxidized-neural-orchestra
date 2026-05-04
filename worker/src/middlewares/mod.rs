mod server_cluster;
mod worker_ring;

use std::{mem, num::NonZeroUsize};

pub use server_cluster::ServerClusterManager;
pub use worker_ring::WorkerRingManager;

/// Helper trait for spliting a gradient into `n`
/// chunks minimizing their difference in length.
trait SplitIntoChunksMut<T> {
    fn split_chunks_mut(&mut self, n: NonZeroUsize) -> SplitChunks<'_, T>;
}

impl<T> SplitIntoChunksMut<T> for [T] {
    /// Creates a new `Split` iterator.
    ///
    /// # Args
    /// * `n` - The amount of chunks to split the slice into.
    ///
    /// # Returns
    /// A new `Split` iterator instance.
    fn split_chunks_mut(&mut self, n: NonZeroUsize) -> SplitChunks<'_, T> {
        SplitChunks {
            len: self.len() / n.get(),
            rem: self.len() % n.get(),
            slice: self,
        }
    }
}

/// The splitted slice iterator.
struct SplitChunks<'a, T> {
    slice: &'a mut [T],
    len: usize,
    rem: usize,
}

impl<'a, T> Iterator for SplitChunks<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        let mut len = self.len;
        if self.rem > 0 {
            len += 1;
            self.rem -= 1;
        }

        let slice = mem::take(&mut self.slice);
        let (chunk, rest) = slice.split_at_mut(len);
        self.slice = rest;

        Some(chunk)
    }
}
