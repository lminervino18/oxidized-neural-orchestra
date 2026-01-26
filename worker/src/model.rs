use comms::specs::worker::ModelSpec;
use machine_learning::{MlError, StepStats, TrainStrategy};

/// Worker-local model selected at runtime from `ModelSpec`.
pub enum Model {
    Noop(NoopModel),
    Mock(MockModel),
    FeedForward(UnsupportedModel),
}

impl Model {
    /// Builds a worker model from a wire-level `ModelSpec`.
    ///
    /// # Args
    /// * `spec` - Model selection received from the orchestrator.
    ///
    /// # Returns
    /// A concrete worker model implementation.
    pub fn from_spec(spec: &ModelSpec) -> Self {
        match spec {
            ModelSpec::Noop => Self::Noop(NoopModel),
            ModelSpec::Mock => Self::Mock(MockModel),
            ModelSpec::FeedForward { .. } => Self::FeedForward(UnsupportedModel),
        }
    }

    /// Returns a stable identifier for the model kind.
    ///
    /// # Returns
    /// A static string identifying the model kind.
    pub fn kind(&self) -> &'static str {
        match self {
            Model::Noop(_) => "noop",
            Model::Mock(_) => "mock",
            Model::FeedForward(_) => "feed_forward",
        }
    }
}

impl TrainStrategy for Model {
    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        match self {
            Model::Noop(m) => m.step(weights, grads),
            Model::Mock(m) => m.step(weights, grads),
            Model::FeedForward(m) => m.step(weights, grads),
        }
    }
}

/// No-op model used for protocol tests.
pub struct NoopModel;

impl TrainStrategy for NoopModel {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

/// Mock model used for end-to-end tests.
pub struct MockModel;

impl TrainStrategy for MockModel {
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

/// Placeholder for models not yet supported by the runtime.
pub struct UnsupportedModel;

impl TrainStrategy for UnsupportedModel {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Err(MlError::InvalidInput("model is not supported yet"))
    }
}
