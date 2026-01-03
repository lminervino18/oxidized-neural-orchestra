pub mod layout;
pub mod ops;
pub mod spec;
pub mod view;

use layout::ParameterLayout;
use spec::ModelSpec;

/// Stable model wrapper used by the worker loop.
#[derive(Debug, Clone)]
pub struct Model {
    spec: ModelSpec,
    layout: ParameterLayout,
}

impl Model {
    pub fn new(spec: ModelSpec) -> Self {
        let layout = ParameterLayout::new(spec);
        Self { spec, layout }
    }

    #[inline]
    pub fn spec(&self) -> ModelSpec {
        self.spec
    }

    #[inline]
    pub fn num_params(&self) -> usize {
        self.spec.num_params()
    }

    #[inline]
    pub fn layout(&self) -> &ParameterLayout {
        &self.layout
    }

    /// Compute gradients for a full batch.
    /// Writes into `grads` (flat buffer) using the internal layout.
    pub fn grad_batch(&self, weights: &[f32], grads: &mut [f32], xs: &[f32], ys: &[f32]) {
        match self.spec {
            ModelSpec::LinearRegression1D => {
                ops::linreg_mse_grad_batch(&self.layout, weights, grads, xs, ys)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::spec::ModelSpec;

    #[test]
    fn model_reports_num_params() {
        let m = Model::new(ModelSpec::LinearRegression1D);
        assert_eq!(m.num_params(), 2);
    }
}
