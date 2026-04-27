use ndarray::{ArrayD, ArrayView, ArrayViewMut, Dimension, IxDyn, azip};

use super::LossFn;
use crate::arch::InplaceReshape;

/// Mean squared error loss function.
#[derive(Default, Clone)]
pub struct Mse {
    delta: ArrayD<f32>,
}

impl Mse {
    /// Creates a new `Mse` loss function.
    ///
    /// # Returns
    /// A new `Mse` instance.
    pub fn new() -> Self {
        Self {
            delta: ArrayD::zeros(IxDyn(&[1])),
        }
    }
}

impl LossFn for Mse {
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
        let two_over_n = 2.0 / n;
        let mut total_loss = 0.0;

        let mut delta_view = self.delta.view_mut().into_dimensionality::<D>().unwrap();

        azip!((grad in &mut delta_view, &yp in &y_pred, &yt in &y) {
            let diff = yp - yt;
            total_loss += diff.powi(2);
            *grad = diff * two_over_n;
        });

        (total_loss / n, delta_view)
    }
}
