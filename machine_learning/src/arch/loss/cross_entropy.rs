use ndarray::{Array2, ArrayView2, ArrayViewMut2, azip};

use super::LossFn;
use crate::arch::InplaceReshape;

/// Cross entropy loss function.
#[derive(Default, Clone)]
pub struct CrossEntropy {
    delta: Array2<f32>,
}

impl CrossEntropy {
    /// Creates a new `CrossEntropy` loss function.
    ///
    /// # Returns
    /// A new `CrossEntropy` instance.
    pub fn new() -> Self {
        Self {
            delta: Array2::zeros((1, 1)),
        }
    }
}

impl LossFn for CrossEntropy {
    fn loss_prime<'a>(
        &'a mut self,
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
    ) -> (f32, ArrayViewMut2<'a, f32>) {
        self.delta.reshape_inplace(y_pred.raw_dim());

        let n = y_pred.len() as f32;
        let one_over_n = 1.0 / n;
        let mut total_loss = 0.0;

        azip!((grad in &mut self.delta, &yp in &y_pred, &yt in &y) {
            let yp_safe = yp.max(f32::EPSILON);
            total_loss -= yt * yp_safe.ln();
            *grad = - (yt / yp_safe) * one_over_n;
        });

        (total_loss / n, self.delta.view_mut())
    }
}
