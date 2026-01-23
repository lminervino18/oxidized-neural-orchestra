use crate::model::layer::Layer;
use ndarray::{s, Array2, ArrayView2};

struct Sequential {
    layers: Vec<Layer>,
}

trait LossFn {
    fn loss(y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32;
    fn loss_prime(y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32;
}

struct Mse;
impl LossFn for Mse {
    fn loss(y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32 {
        todo!()
    }

    fn loss_prime(y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32 {
        todo!()
    }
}

impl Sequential {
    pub fn forward(&mut self, mut params: &[f32], x: ArrayView2<f32>) -> Array2<f32> {
        // let mut y = self.layers[0].forward(params, x);

        for l in &mut self.layers[1..] {
            // y = l.forward(&mut params, y.view());
        }

        // y
        todo!()
    }

    // retornar grad_params (grad_w y grad_b)
    pub fn backward<L: LossFn>(
        &mut self,
        params: &[f32],
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
        loss: &L,
    ) {
        // let mut d = loss.loss_prime(y_pred, y);

        for l in &self.layers {
            // d = l.backward(d);
        }
    }
}
