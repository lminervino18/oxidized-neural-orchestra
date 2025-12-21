use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
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
    /// Two buffers of atomic floats to support double-buffered accumulation.
    grads: [Vec<AtomicF32>; 2],
    /// The index (0 or 1) of the buffer currently designated for accumulation.
    idx: AtomicU8,
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
            idx: AtomicU8::new(0),
        }
    }

    /// Accumulates a local gradient into the currently active global buffer.
    ///
    /// # Arguments
    /// * `gradient` - A slice of gradient computed by a worker.
    pub fn accumulate(&self, gradient: &[f32]) {
        let idx = self.idx.load(Ordering::Acquire) as usize;

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
        let idx = self.idx.fetch_xor(1, Ordering::Release) as usize;
        &self.grads[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_grad(pc: &PSClient, i: usize) -> Vec<f32> {
        pc.grads[i]
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect()
    }

    fn get_idx(pc: &PSClient) -> usize {
        pc.idx.load(Ordering::Relaxed) as usize
    }

    #[test]
    fn write_to_first_gradient_first() {
        let pc = PSClient::new(SharedData::new(3));
        let gradient = [1., 2., 3.];
        pc.accumulate(&gradient);

        assert_eq!(get_grad(&pc, 0), gradient);
    }

    #[test]
    fn swap_gradients() {
        let pc = PSClient::new(SharedData::new(3));
        pc.swap_grad();
        assert_eq!(get_idx(&pc), 1);
    }

    #[test]
    fn write_to_second_gradient_second() {
        let pc = PSClient::new(SharedData::new(3));

        let gradient = [1., 2., 3.];
        pc.accumulate(&gradient);
        assert_eq!(get_grad(&pc, 0), gradient);
        assert_eq!(get_grad(&pc, 1), [0.].repeat(3));

        pc.swap_grad();

        pc.accumulate(&gradient);
        assert_eq!(get_grad(&pc, 0), gradient);
        assert_eq!(get_grad(&pc, 1), gradient);
    }
}
