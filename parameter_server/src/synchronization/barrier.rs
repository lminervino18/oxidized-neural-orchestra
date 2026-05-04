use std::sync::{Arc, Mutex};

use tokio::sync::watch;

use super::Synchronizer;
use crate::storage::{Result, Store, StoreHandle};

struct BarrierState {
    /// Workers still expected to contribute to each barrier step.
    connected: usize,
    /// Workers currently blocked inside `wait`.
    arrived: usize,
    /// Monotone counter; incremented each time the barrier releases.
    generation: u64,
}

/// A barrier that releases early when participating workers disconnect.
///
/// Each time a worker exits without contributing to the current step, it calls
/// [`DrainableBarrier::drain`]. If all remaining connected workers are already
/// waiting, the barrier releases immediately so they are not stuck indefinitely.
struct DrainableBarrier {
    state: Arc<Mutex<BarrierState>>,
    release_tx: Arc<watch::Sender<u64>>,
    release_rx: watch::Receiver<u64>,
}

impl DrainableBarrier {
    fn new(n: usize) -> Self {
        let (release_tx, release_rx) = watch::channel(0u64);
        Self {
            state: Arc::new(Mutex::new(BarrierState {
                connected: n,
                arrived: 0,
                generation: 0,
            })),
            release_tx: Arc::new(release_tx),
            release_rx,
        }
    }

    /// Signals that one worker will no longer contribute to future barrier steps.
    ///
    /// If all remaining connected workers are already waiting, releases the barrier
    /// immediately so they can proceed without the disconnected worker.
    fn drain(&self) {
        let new_gen = {
            let mut state = self.state.lock().unwrap();
            if state.connected == 0 {
                return;
            }
            state.connected -= 1;
            // If every remaining connected worker is already waiting, release now.
            if state.connected > 0 && state.arrived >= state.connected {
                state.generation += 1;
                state.arrived = 0;
                Some(state.generation)
            } else {
                None
            }
        };

        if let Some(released_at) = new_gen {
            let _ = self.release_tx.send(released_at);
        }
    }

    /// Waits until all connected workers have arrived at this barrier step.
    ///
    /// Returns `true` if this caller is the designated leader and should
    /// perform the parameter update, `false` otherwise.
    async fn wait(&self) -> bool {
        let (target_generation, is_leader, maybe_release) = {
            let mut state = self.state.lock().unwrap();
            state.arrived += 1;
            let target_generation = state.generation;
            let is_leader = state.arrived >= state.connected;

            let maybe_release = if is_leader {
                state.generation += 1;
                state.arrived = 0;
                Some(state.generation)
            } else {
                None
            };

            (target_generation, is_leader, maybe_release)
        };

        if let Some(released_at) = maybe_release {
            let _ = self.release_tx.send(released_at);
            return is_leader;
        }

        // Wait until the watch value moves past the generation we started in.
        let mut rx = self.release_rx.clone();
        let _ = rx.wait_for(|&v| v > target_generation).await;
        false
    }
}

impl Clone for DrainableBarrier {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            release_tx: self.release_tx.clone(),
            release_rx: self.release_rx.clone(),
        }
    }
}

/// Synchronizes parameter updates across multiple workers using a drainable barrier.
///
/// When all N connected workers push their gradients, the barrier releases, one
/// leader applies the accumulated gradients, and all workers receive the updated
/// parameters before the next epoch. If a worker disconnects mid-training, calling
/// [`BarrierSync::drain`] decrements the expected count so remaining workers are
/// never left waiting indefinitely.
pub struct BarrierSync {
    barrier: DrainableBarrier,
}

impl BarrierSync {
    /// Creates a new `BarrierSync` synchronizer.
    ///
    /// # Args
    /// * `barrier_size` - The number of workers expected to synchronize each epoch.
    ///
    /// # Returns
    /// A new `BarrierSync` instance.
    pub fn new(barrier_size: usize) -> Self {
        Self {
            barrier: DrainableBarrier::new(barrier_size),
        }
    }
}

impl Clone for BarrierSync {
    fn clone(&self) -> Self {
        Self {
            barrier: self.barrier.clone(),
        }
    }
}

impl Synchronizer for BarrierSync {
    fn drain(&self) {
        self.barrier.drain();
    }

    async fn step<S>(&self, handle: &StoreHandle<S>, grad: &[f32], params: &mut [f32]) -> Result<()>
    where
        S: Store + Send + Sync,
    {
        handle.accumulate(grad).await?;

        if self.barrier.wait().await {
            handle.update_params().await;
        }

        self.barrier.wait().await;
        handle.pull_params(params).await?;
        Ok(())
    }
}
