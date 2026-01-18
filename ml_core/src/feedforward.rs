use super::model::Model;
use ndarray::Array1;

pub trait Feedforward: Model {
    fn forward(&mut self, x: Array1<f32>) -> Array1<f32>;
}
