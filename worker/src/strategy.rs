use std::io;

use comms::specs::worker::{StrategySpec, WorkerSpec};
use ml_core::{MlError, StepStats, TrainStrategy};

/// Training strategy selected from a `WorkerSpec`.
#[derive(Debug)]
pub enum Strategy {
    Noop(Noop),
    Mock(Mock),
}

impl Strategy {
    /// Builds a `Strategy` from a wire-level `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    ///
    /// # Returns
    /// A concrete `Strategy` implementation.
    ///
    /// # Errors
    /// Returns `io::Error` if the spec is invalid for the selected strategy.
    pub fn from_spec(spec: &WorkerSpec) -> io::Result<Self> {
        Ok(match spec.strategy {
            StrategySpec::Noop => Self::Noop(Noop),
            StrategySpec::Mock => Self::Mock(Mock),
        })
    }

    /// Returns a stable string label for logging.
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

#[derive(Debug)]
pub struct Noop;

impl TrainStrategy for Noop {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

#[derive(Debug)]
pub struct Mock;

impl TrainStrategy for Mock {
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
