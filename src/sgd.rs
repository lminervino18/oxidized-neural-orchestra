use super::{neural_net::Mlp, optimizer::Optimizer};
use crate::feedforward::Feedforward;
use ndarray::{Array1, Array2, ArrayView1};

pub struct Sgd {
    eta: f32,
}

fn cost(y_pred: ArrayView1<f32>, y: ArrayView1<f32>) -> f32 {
    (y_pred.to_owned() - y).pow2().sum().sqrt()
}

fn cost_prime(y_pred: ArrayView1<f32>, y: ArrayView1<f32>) -> Array1<f32> {
    (y_pred.to_owned() - y).mapv(|x| x * 2.)
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
        model: &Mlp,
        y_pred: Array1<f32>,
        y: Array1<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        // init grads
        let mut grad_w: Vec<_> = model
            .weights
            .iter()
            .map(|w| Array2::<f32>::zeros(w.dim()))
            .collect();
        let mut grad_b: Vec<_> = model
            .biases
            .iter()
            .map(|b| Array1::<f32>::zeros(b.dim()))
            .collect();

        // compute last layer's delta (cannot be inside the loop since it requires a different math
        // expression) and use it to compute model's last layers grads
        let mut delta = cost_prime(y_pred.view(), y.view())
            * model
                .weighted_sums
                .last()
                .unwrap()
                .mapv(|z| (model.sigmoid_prime)(z));

        *grad_w.last_mut().unwrap() = outer_product1(delta.view(), y_pred.view());
        *grad_b.last_mut().unwrap() = delta.clone();

        // loop through all layers and compute each delta and use it to compute the grads of the layer
        (0..model.weights.len() - 1).rev().for_each(|idx| {
            let z = &model.weighted_sums[idx];
            let a = &model.activations[idx];
            let w_next = &model.weights[idx + 1];
            let sigmoid_prime = &(model.sigmoid_prime);

            delta = w_next.t().dot(&delta) * z.mapv(sigmoid_prime);
            grad_w[idx] = outer_product1(delta.view(), a.view());
            grad_b[idx] = delta.clone();
        });

        // nashe
        grad_w.reverse();
        grad_b.reverse();

        (grad_w, grad_b)
    }
}

impl Optimizer for Sgd {
    type ModelT = Mlp;

    fn train(
        &self,
        model: &mut Mlp,
        x_train: Vec<Array1<f32>>,
        y_train: Vec<Array1<f32>>,
        n_iters: usize,
    ) {
        for _ in 0..n_iters {
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

            x_train.iter().zip(y_train.clone()).for_each(|(x, y)| {
                let y_pred = model.forward(x.clone());
                let (delta_gw, delta_gb) = self.backprop(model, y_pred, y);

                grad_w
                    .iter_mut()
                    .zip(delta_gw.iter().rev())
                    .for_each(|(gw, dgw)| *gw += dgw);

                grad_b
                    .iter_mut()
                    .zip(delta_gb.iter().rev())
                    .for_each(|(gb, dgb)| *gb += dgb);
            });

            // dbg!(grad_w);
            // panic!();
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
}

#[cfg(test)]
mod test {
    use super::*;

    fn sigmoid(z: f32) -> f32 {
        1. / (1. + std::f32::consts::E.powf(-z))
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
        let mut net = Mlp::new(&[2, 3, 1], sigmoid, sigmoid_prime);
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

        let sgd = Sgd { eta: 3. };
        sgd.train(&mut net, x_train.to_vec(), y_train.to_vec(), 10000);

        let y_pred = x_train.clone().map(|x| net.forward(x));

        let error = y_pred
            .into_iter()
            .zip(&y_train)
            .map(|(yp, y)| cost(yp.view(), y.view()))
            .collect::<Vec<_>>()
            .into_iter()
            .sum::<f32>()
            / y_train.len() as f32;

        dbg!(error);
        assert!(error < 0.01);
    }
}
