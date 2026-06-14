use std::num::NonZeroUsize;

use parking_lot::{Condvar, Mutex, MutexGuard};

/// Tracks the uses of a `DynBarrier` instance.
struct BarrierSharedState {
    generation: usize,
    leader_gen: usize,
    remaining: usize,
    size: usize,
}

/// A barrier implementation with dynamic amount of waiting threads to trigger.
pub struct DynBarrier {
    state: Mutex<BarrierSharedState>,
    cvar: Condvar,
}

impl DynBarrier {
    /// Creates a new `DynBarrier`.
    ///
    /// # Args
    /// * `size` - The amount of waiting threads needed to trigger the barrier.
    ///
    /// # Returns
    /// A new `DynBarrier` instance.
    pub fn new(n: NonZeroUsize) -> Self {
        let shared_state = BarrierSharedState {
            generation: 0,
            leader_gen: 0,
            remaining: n.get(),
            size: n.get(),
        };

        Self {
            state: Mutex::new(shared_state),
            cvar: Condvar::new(),
        }
    }

    /// Locks the current thread until all required threads wait for the barrier to trigger.
    ///
    /// The leader guard will trigger the barrier once it's dropped.
    ///
    /// # Returns
    /// Either the leader guard or nothing.
    pub fn wait_with<F>(&self, mut leader_fn: F)
    where
        F: FnMut(),
    {
        let mut state = self.state.lock();
        state.remaining -= 1;

        let is_leader = if state.remaining > 0 {
            self.wait_until_new_gen(&mut state);
            state.leader_gen < state.generation
        } else {
            true
        };

        if is_leader {
            leader_fn();
            state.leader_gen += 1;

            if state.leader_gen > state.generation {
                self.advance(&mut state);
            }
        }
    }

    /// Acquires a waiting thread's place by decreasing the amount of needed threads to trigger the barrier.
    pub fn acquire(&self) {
        let mut state = self.state.lock();

        state.remaining -= 1;
        state.size -= 1;

        if state.remaining == 0 {
            self.advance(&mut state);
        }
    }

    /// Advances the barrier to the next generation, updates tracking counters,
    /// and wakes up all threads registered to the condition variable.
    ///
    /// # Args
    /// * `state` - The state to mutate by advancing the counters.
    fn advance(&self, guard: &mut MutexGuard<BarrierSharedState>) {
        guard.generation += 1;
        guard.remaining = guard.size;

        if guard.remaining > 1 {
            self.cvar.notify_all();
        }
    }

    /// Blocks the current thread until the current generation changes.
    ///
    /// # Args
    /// * `guard` - The barrier's mutable state guard.
    fn wait_until_new_gen(&self, guard: &mut MutexGuard<BarrierSharedState>) {
        let local_gen = guard.generation;

        self.cvar
            .wait_while(guard, |state| state.generation == local_gen);
    }
}
