use std::io;

use comms::specs::worker::StrategySpec;
use ml_core::{MlError, StepStats, TrainStrategy};

pub fn from_spec(spec: &StrategySpec) -> io::Result<Strategy> {
    Ok(match spec {
        StrategySpec::Noop => Strategy::Noop(NoopStrategy),
        StrategySpec::Mock => Strategy::Mock(MockStrategy),
    })
}


pub enum Strategy {
    Noop(NoopStrategy),
    Mock(MockStrategy),
}

impl TrainStrategy for Strategy {
    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        match self {
            Strategy::Noop(s) => s.step(weights, grads),
            Strategy::Mock(s) => s.step(weights, grads),
        }
    }
}

impl Strategy {
    pub fn kind(&self) -> &'static str {
        match self {
            Strategy::Noop(_) => "noop",
            Strategy::Mock(_) => "mock",
        }
    }
}


pub struct NoopStrategy;

impl TrainStrategy for NoopStrategy {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

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
