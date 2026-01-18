use super::{dataset::Dataset, neural_net::NeuralNet, optimizer::Optimizer};
use crate::feedforward::Feedforward;
use ndarray::{Array1, Array2, ArrayView1};

pub struct SGD {
    eta: f32,
}

fn cost(y_pred: ArrayView1<f32>, y: ArrayView1<f32>) -> Array1<f32> {
    (y - y_pred).pow2()
}

fn cost_prime(y_pred: ArrayView1<f32>, y: ArrayView1<f32>) -> Array1<f32> {
    todo!()
}

impl SGD {
pub fn outer_product1(v: ArrayView1<f32>, w: ArrayView1<f32>) -> Array2<f32> {
    let v_reshaped = v.to_shape((1, v.dim())).unwrap();
    let w_reshaped = w.to_shape((1, w.dim())).unwrap();
    v_reshaped.dot(&w_reshaped.t())
}

// from https://github.com/rust-ndarray/ndarray/issues/1148
// pub fn outer_product2(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array4<f32> {
//     let (m, n) = a.dim();
//     let (p, q) = b.dim();
//
//     let a_reshaped = a.to_shape((m, n, 1, 1)).unwrap();
//     let b_reshaped = b.to_shape((1, 1, p, q)).unwrap();
//
//     let prod = &a_reshaped * &b_reshaped;
//
//     prod.to_owned()
// }
    fn backprop(
        &self,
        model: &NeuralNet,
        y_pred: Array1<f32>,
        y: Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let grad_a_cost = cost_prime(y_pred.view(), y.view());
        let delta_output = grad_a_cost
            * model
                .weighted_sums
                .last()
                .unwrap()
                .clone()
                .mapv_into(|z| (model.sigmoid_prime)(z));

        (0..model.weights.len()).rev().into_iter().for_each(|idx| {
            /*
             * copiado de neural_net
             * let z = &mut model.weighted_sums[idx];
            // let input = &mut model.activations[idx];
            // let output = &mut model.activations[idx + 1];
            let w = &model.weights[idx];
            let b = &model.biases[idx];
            let sigmoid = &(model.sigmoid);

            *z = w.dot(&model.activations[idx]) + b;
            model.activations[idx + 1] = z.clone().mapv_into(sigmoid); */
        });

        todo!()
    }
}

impl Optimizer for SGD {
    type ModelT = NeuralNet;

    fn train(&self, model: &mut NeuralNet, x_train: Vec<Array1<f32>>, y_train: Vec<Array1<f32>>) {
        let mut grad_w: Vec<_> = model
            .weights
            .iter()
            .map(|w| Array2::zeros(w.dim()))
            .collect();

        let mut grad_b: Vec<_> = model
            .biases
            .iter()
            .map(|b| Array1::zeros(b.dim()))
            .collect();

        x_train.iter().zip(y_train).for_each(|(x, y)| {
            let y_pred = model.forward(x.clone());
            let (delta_gw, delta_gb) = self.backprop(y_pred, y);

            grad_w
                .iter_mut()
                .zip(delta_gw)
                .for_each(|(gw, dgw)| *gw += &dgw);

            grad_b
                .iter_mut()
                .zip(delta_gb)
                .for_each(|(gb, dgb)| *gb += &dgb);
        });

        model
            .weights
            .iter_mut()
            .zip(grad_w)
            .for_each(|(w, gw)| w.scaled_add(-self.eta / x_train.len() as f32, &gw));

        model
            .biases
            .iter_mut()
            .zip(grad_b)
            .for_each(|(b, gb)| b.scaled_add(-self.eta / x_train.len() as f32, &gb));
    }
}
