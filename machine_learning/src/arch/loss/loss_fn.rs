use ndarray::{Array2, ArrayView2};

pub trait LossFn {
    fn loss(&self, y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32;
    fn loss_prime(&self, y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> Array2<f32>;
}
