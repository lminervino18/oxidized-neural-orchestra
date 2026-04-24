use ndarray::{ArrayD, ArrayView, ArrayViewMut, Dimension, IxDyn, azip};

use super::LossFn;
use crate::arch::InplaceReshape;

/// Cross entropy loss function.
#[derive(Default, Clone)]
pub struct CrossEntropy {
    delta: ArrayD<f32>,
}

impl CrossEntropy {
    /// Creates a new `CrossEntropy` loss function.
    ///
    /// # Returns
    /// A new `CrossEntropy` instance.
    pub fn new() -> Self {
        Self {
            delta: ArrayD::zeros(IxDyn(&[1])),
        }
    }
}

impl LossFn for CrossEntropy {
    fn loss_prime<D>(
        &mut self,
        y_pred: ArrayView<f32, D>,
        y: ArrayView<f32, D>,
    ) -> (f32, ArrayViewMut<'_, f32, D>)
    where
        D: Dimension,
    {
        self.delta.reshape_inplace(y.raw_dim().into_dyn());

        let n = y_pred.len() as f32;
        let one_over_n = 1.0 / n;
        let mut total_loss = 0.0;

        let mut delta_view = self.delta.view_mut().into_dimensionality::<D>().unwrap();

        azip!((grad in &mut delta_view, &yp in &y_pred, &yt in &y) {
            let yp_safe = yp.max(f32::EPSILON);
            total_loss -= yt * yp_safe.ln();
            *grad = - (yt / yp_safe) * one_over_n;
        });

        (total_loss / n, delta_view)
    }
}
