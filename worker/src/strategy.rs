use std::io;

use comms::specs::worker::StrategySpec;
use ml_core::{MlError, StepStats, TrainStrategy};

/// Worker-local training strategy.
///
/// This type provides a runtime-selectable implementation of `TrainStrategy`
/// derived from a wire-level `StrategySpec`.
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
    ///
    /// # Errors
    /// Returns `io::Error` if the requested strategy cannot be constructed.
    ///
    /// # Panics
    /// Never panics.
    pub fn from_spec(spec: &StrategySpec) -> io::Result<Self> {
        Ok(match spec {
            StrategySpec::Noop => Self::Noop(NoopStrategy),
            StrategySpec::Mock => Self::Mock(MockStrategy),
        })
    }

    /// Returns a stable identifier for the strategy kind.
    ///
    /// # Returns
    /// A static string identifying the strategy kind.
    ///
    /// # Panics
    /// Never panics.
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
///
/// This strategy always returns a successful step and leaves gradients as-is.
pub struct NoopStrategy;

impl TrainStrategy for NoopStrategy {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

/// Strategy used for testing the training loop.
///
/// This strategy computes `grad[i] = 2 * weight[i]`.
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
