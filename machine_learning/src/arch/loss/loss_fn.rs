use ndarray::{ArrayView, ArrayViewMut, Dimension};

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
    fn loss_prime<D>(
        &mut self,
        y_pred: ArrayView<f32, D>,
        y: ArrayView<f32, D>,
    ) -> (f32, ArrayViewMut<'_, f32, D>)
    where
        D: Dimension;

    /// Calculates the model's current loss.
    ///
    /// # Arguments
    /// * `y_pred` - The models' computed output.
    /// * `y` - The expected *real* output.
    ///
    /// # Returns
    /// The calculated loss.
    fn loss<D>(&mut self, y_pred: ArrayView<f32, D>, y: ArrayView<f32, D>) -> f32
    where
        D: Dimension,
    {
        self.loss_prime(y_pred, y).0
    }
}
