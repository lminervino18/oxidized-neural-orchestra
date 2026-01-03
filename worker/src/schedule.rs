use std::num::NonZeroUsize;

/// Defines when to send gradients.
#[derive(Debug, Clone)]
pub struct Schedule {
    pub microbatch_k: NonZeroUsize,
}

impl Schedule {
    pub fn new(microbatch_k: NonZeroUsize) -> Self {
        Self { microbatch_k }
    }

    /// Returns true if this step ends a microbatch window.
    #[inline]
    pub fn should_send(&self, local_batch_index: usize) -> bool {
        let k = self.microbatch_k.get();
        (local_batch_index + 1) % k == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn microbatch_schedule() {
        let s = Schedule::new(NonZeroUsize::new(3).unwrap());
        assert!(!s.should_send(0));
        assert!(!s.should_send(1));
        assert!(s.should_send(2));
        assert!(!s.should_send(3));
        assert!(!s.should_send(4));
        assert!(s.should_send(5));
    }
}
