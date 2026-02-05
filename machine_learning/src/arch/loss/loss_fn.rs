use ndarray::{Array2, ArrayView2};

/// This trait represent the *function* that is to be used to compute the difference between a
/// given output and the expected one.
pub trait LossFn {
    /// Returns the loss of the predicted output `y_pred` with respect to the expected one `y`.
    ///
    /// # Arguments
    /// * `y_pred` - The computed output
    /// * `y` - The expected *real* output
    fn loss(&self, y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32;
    /// Returns the derivative of the loss of the predicted output `y_pred` with respect to the expected one `y`.
    ///
    /// # Arguments
    /// * `y_pred` - The computed output
    /// * `y` - The expected *real* output
    fn loss_prime(&self, y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> Array2<f32>;
}
