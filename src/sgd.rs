use super::{neural_net::NeuralNet, optimizer::Optimizer};
use crate::feedforward::Feedforward;
use ndarray::{Array1, Array2, ArrayView1};

pub struct Sgd {
    eta: f32,
}

fn cost(y_pred: ArrayView1<f32>, y: ArrayView1<f32>) -> Array1<f32> {
    (y.to_owned() - y_pred).pow2()
}

fn cost_prime(y_pred: ArrayView1<f32>, y: ArrayView1<f32>) -> Array1<f32> {
    let mut cost_prime = y.to_owned() - y_pred;
    cost_prime.mapv_inplace(|x| x * 2.);
    cost_prime
}

pub fn outer_product1(v: ArrayView1<f32>, w: ArrayView1<f32>) -> Array2<f32> {
    let v_reshaped = v.to_shape((1, v.dim())).unwrap();
    let w_reshaped = w.to_shape((1, w.dim())).unwrap();
    v_reshaped.t().dot(&w_reshaped)
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

impl Sgd {
    fn backprop(
        &self,
        model: &NeuralNet,
        y_pred: Array1<f32>,
        y: Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let grad_a_cost = cost_prime(y_pred.view(), y.view());
        let mut delta = grad_a_cost
            * model
                .weighted_sums
                .last()
                .unwrap()
                .clone()
                .mapv_into(|z| (model.sigmoid_prime)(z));

        let (grad_w, grad_b): (Vec<_>, Vec<_>) = (0..model.weights.len() - 1)
            .rev()
            .map(|idx| {
                let z = &model.weighted_sums[idx];
                let input = &model.activations[idx];
                let w = &model.weights[idx + 1];
                let sigmoid_prime = &(model.sigmoid_prime);

                delta = w.t().dot(&delta) * z.clone().mapv_into(sigmoid_prime);
                let grad_w = outer_product1(delta.view(), input.view());
                let grad_b = delta.clone();

                (grad_w, grad_b)
            })
            .collect();

        (grad_w, grad_b)
    }
}

impl Optimizer for Sgd {
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
            let (delta_gw, delta_gb) = self.backprop(model, y_pred, y);

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

#[cfg(test)]
mod test {
    use super::*;

    fn sigmoid(z: f32) -> f32 {
        1. / (1. - std::f32::consts::E.powf(z))
    }

    fn sigmoid_prime(z: f32) -> f32 {
        let sigmoid = sigmoid(z);
        sigmoid * (1. - sigmoid)
    }

    #[test]
    fn test_outer_product1() {
        let a = Array1::<f32>::from_vec(vec![1., 2., 3.]);
        let expected =
            Array2::<f32>::from_shape_vec((3, 3), vec![1., 2., 3., 2., 4., 6., 3., 6., 9.])
                .unwrap();
        let outer = outer_product1(a.view(), a.view());
        assert_eq!(outer, expected);
    }

    #[test]
    fn test00() {
        let mut net = NeuralNet::new(&[2, 3, 1], sigmoid, sigmoid_prime);
        let x_train = [
            Array1::<f32>::from_vec(vec![0., 0.]),
            Array1::<f32>::from_vec(vec![0., 1.]),
            Array1::<f32>::from_vec(vec![1., 0.]),
            Array1::<f32>::from_vec(vec![1., 1.]),
        ];
        let y_train = [
            Array1::<f32>::from_elem(1, 0.),
            Array1::<f32>::from_elem(1, 0.),
            Array1::<f32>::from_elem(1, 0.),
            Array1::<f32>::from_elem(1, 1.),
        ];

        let sgd = Sgd { eta: 1. };
        sgd.train(&mut net, x_train.to_vec(), y_train.to_vec());

        let y_pred = x_train.clone().map(|x| net.forward(x));
        assert_eq!(y_pred, y_train);
    }
}
