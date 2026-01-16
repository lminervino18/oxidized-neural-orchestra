use super::{dataset::Dataset, neural_net::NeuralNet, optimizer::Optimizer};
use crate::feedforward::Feedforward;
use ndarray::{Array1, Array2};

pub struct SGD {
    eta: f32,
}

impl SGD {
    fn backprop(
        &self,
        y_pred: Array1<f32>,
        y: Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
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
            .for_each(|(w, gw)| w.scaled_add(-self.eta, &gw));

        model
            .biases
            .iter_mut()
            .zip(grad_b)
            .for_each(|(b, gb)| b.scaled_add(-self.eta, &gb));
    }
}
