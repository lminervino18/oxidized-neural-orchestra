use ndarray::{Array2, ArrayView2, ArrayViewMut2, azip};

use super::LossFn;
use crate::arch::InplaceReshape;

/// Mean squared error loss function.
#[derive(Default, Clone)]
pub struct Mse {
    delta: Array2<f32>,
}

impl Mse {
    /// Creates a new `Mse` loss function.
    ///
    /// # Returns
    /// A new `Mse` instance.
    pub fn new() -> Self {
        Self {
            delta: Array2::zeros((1, 1)),
        }
    }
}

impl LossFn for Mse {
    fn loss_prime<'a>(
        &'a mut self,
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
    ) -> (f32, ArrayViewMut2<'a, f32>) {
        self.delta.reshape_inplace(y_pred.raw_dim());

        let n = y_pred.len() as f32;
        let two_over_n = 2.0 / n;
        let mut total_loss = 0.0;

        azip!((grad in &mut self.delta, &yp in &y_pred, &yt in &y) {
            let diff = yp - yt;
            total_loss += diff.powi(2);
            *grad = diff * two_over_n;
        });

        (total_loss / n, self.delta.view_mut())
    }
}
