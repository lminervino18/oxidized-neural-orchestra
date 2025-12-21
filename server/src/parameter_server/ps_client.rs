use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use atomic_float::AtomicF32;
use rayon::prelude::*;

/// A thread-safe handle for workers to interact with the Parameter Server.
///
/// `PSClient` is a reference-counted handle to the underlying `SharedData`. It is designed to be cloned and moved into
/// multiple worker threads, allowing for concurrent gradient accumulation without explicit locking.
pub type PSClient = Arc<SharedData>;

/// The core synchronization and storage primitive for gradient accumulation.
///
/// `SharedData` manages two identical buffers of atomic floating-point numbers. At any given time, one buffer is
/// "active" (receiving updates from workers) while the other is "offline" (being processed by the server).
#[derive(Debug)]
pub struct SharedData {
    grads: [Vec<AtomicF32>; 2],
    idx: AtomicUsize,
}

impl SharedData {
    /// Creates a new `SharedData` instance with all gradients initialized to zero.
    ///
    /// # Arguments
    /// * `n` - The number of parameters in the model.
    pub fn new(n: usize) -> Self {
        Self {
            grads: [
                (0..n).map(|_| AtomicF32::new(0.)).collect(),
                (0..n).map(|_| AtomicF32::new(0.)).collect(),
            ],
            idx: AtomicUsize::new(0),
        }
    }

    /// Accumulates a local gradient into the currently active global buffer.
    ///
    /// # Arguments
    /// * `gradient` - A slice of gradient computed locally by a worker.
    pub fn accumulate(&self, gradient: &[f32]) {
        let idx = self.idx.load(Ordering::Acquire);

        self.grads[idx]
            .par_iter()
            .zip(gradient.par_iter())
            .for_each(|(acc, &delta)| {
                acc.fetch_add(delta, Ordering::Relaxed);
            });
    }

    /// Swaps the active buffer index and returns a reference to the previously active buffer.
    ///
    /// This is typically called by the `PSServer` during a weight update.
    ///
    /// # Returns
    /// A reference to the buffer containing the accumulated gradients for the current epoch.
    pub fn swap_grad(&self) -> &[AtomicF32] {
        let idx = self.idx.fetch_xor(1, Ordering::Release);
        &self.grads[idx]
    }
}
