use comms::specs::worker::StrategySpec;
use machine_learning::{MlError, StepStats, TrainStrategy};

/// Worker-local training strategy selected at runtime from `StrategySpec`.
pub enum Strategy {
    Noop(NoopStrategy),
    Mock(MockStrategy),
}

impl Strategy {
    /// Builds a worker strategy from a wire-level `StrategySpec`.
    ///
    /// # Args
    /// * `spec` - Strategy selection received from the orchestrator.
    ///
    /// # Returns
    /// A concrete worker strategy implementation.
    pub fn from_spec(spec: &StrategySpec) -> Self {
        match spec {
            StrategySpec::Noop => Self::Noop(NoopStrategy),
            StrategySpec::Mock => Self::Mock(MockStrategy),
        }
    }

    /// Returns a stable identifier for the strategy kind.
    ///
    /// # Returns
    /// A static string identifying the strategy kind.
    pub fn kind(&self) -> &'static str {
        match self {
            Strategy::Noop(_) => "noop",
            Strategy::Mock(_) => "mock",
        }
    }
}

impl TrainStrategy for Strategy {
    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        match self {
            Strategy::Noop(s) => s.step(weights, grads),
            Strategy::Mock(s) => s.step(weights, grads),
        }
    }
}

/// Strategy that performs no-op training.
pub struct NoopStrategy;

impl TrainStrategy for NoopStrategy {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

/// Strategy used for testing the training loop.
pub struct MockStrategy;

impl TrainStrategy for MockStrategy {
    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        if weights.len() != grads.len() {
            return Err(MlError::ShapeMismatch {
                what: "params",
                got: weights.len(),
                expected: grads.len(),
            });
        }

        for (g, w) in grads.iter_mut().zip(weights.iter()) {
            *g = 2.0 * *w;
        }

        Ok(StepStats::new(1, 0))
    }
}
