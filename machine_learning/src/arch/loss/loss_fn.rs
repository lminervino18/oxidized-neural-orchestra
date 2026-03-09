use ndarray::{ArrayView2, ArrayViewMut2};

/// This trait represent the *function* that is to be used to compute the difference between a
/// given output and the expected one.
pub trait LossFn {
    /// Calculates a model's current loss and it's derivative vector in one call.
    ///
    /// # Arguments
    /// * `y_pred` - The models' computed output.
    /// * `y` - The expected *real* output.
    ///
    /// # Returns
    /// The calculated loss and it's derivative vector.
    fn loss_prime<'a>(
        &'a mut self,
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
    ) -> (f32, ArrayViewMut2<'a, f32>);

    /// Calculates the model's current loss.
    ///
    /// # Arguments
    /// * `y_pred` - The models' computed output.
    /// * `y` - The expected *real* output.
    ///
    /// # Returns
    /// The calculated loss.
    fn loss(&mut self, y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32 {
        self.loss_prime(y_pred, y).0
    }
}
