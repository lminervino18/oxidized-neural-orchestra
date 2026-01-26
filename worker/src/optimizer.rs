use comms::specs::worker::{ModelSpec, TrainingSpec};
use machine_learning::{MlError, StepStats};

/// Worker-local optimization interface.
pub trait Optimizer: Send {
    /// Computes gradients for the given weights.
    ///
    /// # Args
    /// * `weights` - Current model parameters.
    /// * `grads` - Output buffer for gradients.
    ///
    /// # Returns
    /// Step-level statistics on success.
    ///
    /// # Errors
    /// Returns `MlError` when inputs violate shape invariants or are invalid.
    fn gradient(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError>;

    /// Applies a local update to the weights.
    ///
    /// # Args
    /// * `weights` - Mutable model parameters to update.
    /// * `grads` - Gradients to apply.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// Returns `MlError` when inputs violate shape invariants or are invalid.
    fn update_weights(&mut self, weights: &mut [f32], grads: &[f32]) -> Result<(), MlError>;
}

/// Builds a worker-local optimizer from wire-level specs.
pub struct OptimizerBuilder;

impl OptimizerBuilder {
    /// Builds an optimizer from model and training specifications.
    ///
    /// # Args
    /// * `model` - Model specification.
    /// * `training` - Training specification.
    ///
    /// # Returns
    /// A boxed optimizer implementation.
    pub fn build(model: &ModelSpec, training: &TrainingSpec) -> Box<dyn Optimizer> {
        match model {
            ModelSpec::Noop => Box::new(NoopOptimizer::new(training.offline_steps)),
            ModelSpec::Mock => Box::new(MockOptimizer::new(training.offline_steps)),
            ModelSpec::FeedForward { .. } => Box::new(UnsupportedOptimizer),
        }
    }
}

struct NoopOptimizer {
    offline_steps: usize,
}

impl NoopOptimizer {
    fn new(offline_steps: usize) -> Self {
        Self { offline_steps }
    }
}

impl Optimizer for NoopOptimizer {
    fn gradient(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(self.offline_steps.max(1), 0))
    }

    fn update_weights(&mut self, _weights: &mut [f32], _grads: &[f32]) -> Result<(), MlError> {
        Ok(())
    }
}

struct MockOptimizer {
    offline_steps: usize,
}

impl MockOptimizer {
    fn new(offline_steps: usize) -> Self {
        Self { offline_steps }
    }
}

impl Optimizer for MockOptimizer {
    fn gradient(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
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

        Ok(StepStats::new(self.offline_steps.max(1), 0))
    }

    fn update_weights(&mut self, _weights: &mut [f32], _grads: &[f32]) -> Result<(), MlError> {
        Ok(())
    }
}

struct UnsupportedOptimizer;

impl Optimizer for UnsupportedOptimizer {
    fn gradient(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Err(MlError::InvalidInput("model is not supported yet"))
    }

    fn update_weights(&mut self, _weights: &mut [f32], _grads: &[f32]) -> Result<(), MlError> {
        Err(MlError::InvalidInput("model is not supported yet"))
    }
}

impl<T: Optimizer + ?Sized> Optimizer for Box<T> {
    fn gradient(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        (**self).gradient(weights, grads)
    }

    fn update_weights(&mut self, weights: &mut [f32], grads: &[f32]) -> Result<(), MlError> {
        (**self).update_weights(weights, grads)
    }
}
