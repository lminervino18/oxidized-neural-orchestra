/// Persistent buffers reused across steps to avoid per-iteration allocations.
#[derive(Debug)]
pub struct WorkerState {
    pub step: u64,

    /// Local snapshot of weights (flat).
    pub weights: Vec<f32>,

    /// Gradient buffer (flat).
    pub grads: Vec<f32>,
}

impl WorkerState {
    pub fn new(num_params: usize) -> Self {
        Self {
            step: 0,
            weights: vec![0.0; num_params],
            grads: vec![0.0; num_params],
        }
    }

    #[inline]
    pub fn zero_grads(&mut self) {
        self.grads.fill(0.0);
    }

    #[inline]
    pub fn inc_step(&mut self) {
        self.step += 1;
    }
}
