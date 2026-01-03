use super::layout::ParameterLayout;

/// A read-only view over a flat weights buffer.
///
/// The view *does not own* weights. It interprets them via `ParameterLayout`.
#[derive(Debug, Clone, Copy)]
pub struct LinearRegressionView<'a> {
    weights: &'a [f32],
    layout: &'a ParameterLayout,
}

impl<'a> LinearRegressionView<'a> {
    pub fn new(weights: &'a [f32], layout: &'a ParameterLayout) -> Self {
        // For this baseline model, expect 2 params total.
        debug_assert!(weights.len() >= layout.b.end);
        Self { weights, layout }
    }

    #[inline]
    pub fn w(&self) -> f32 {
        self.weights[self.layout.w.start]
    }

    #[inline]
    pub fn b(&self) -> f32 {
        self.weights[self.layout.b.start]
    }

    /// y = w*x + b
    #[inline]
    pub fn predict(&self, x: f32) -> f32 {
        self.w() * x + self.b()
    }
}
