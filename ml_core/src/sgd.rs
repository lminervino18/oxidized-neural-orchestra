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

impl Sgd {
    fn backprop(
        &self,
        model: &Mlp,
        y_pred: ArrayView1<f32>,
        y: ArrayView1<f32>,
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

        *grad_w.last_mut().unwrap() = outer_product1(
            delta.view(),
            model.activations[model.activations.len() - 2].view(),
        );

        *grad_b.last_mut().unwrap() = delta.clone();

        // loop through all layers, compute each delta and use it to compute the grads of the layer
        let n_layers = model.weights.len();
        (0..n_layers - 1).rev().for_each(|idx| {
            let z = &model.weighted_sums[idx];
            let a = &model.activations[idx];
            let w_next = &model.weights[idx + 1];
            let sigmoid_prime = &(model.sigmoid_prime);

            delta = w_next.t().dot(&delta) * z.mapv(sigmoid_prime);
            grad_w[idx] = outer_product1(delta.view(), a.view());
            grad_b[idx] = delta.clone();
        });

        (grad_w, grad_b)
    }
}

impl Optimizer for Sgd {
    type ModelT = Mlp;

    fn optimize(
        &self,
        model: &mut Mlp,
        x_train: &[Array1<f32>],
        y_train: &[Array1<f32>],
        n_iters: usize,
    ) {
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

        for _ in 0..n_iters {
            grad_w.iter_mut().for_each(|g| g.fill(0.));
            grad_b.iter_mut().for_each(|g| g.fill(0.));

            x_train.iter().zip(y_train).for_each(|(x, y)| {
                let y_pred = model.forward(x.view());
                let (delta_gw, delta_gb) = self.backprop(model, y_pred.view(), y.view());

                grad_w
                    .iter_mut()
                    .zip(&delta_gw)
                    .for_each(|(gw, dgw)| *gw += dgw);

                grad_b
                    .iter_mut()
                    .zip(&delta_gb)
                    .for_each(|(gb, dgb)| *gb += dgb);
            });

            model
                .weights
                .iter_mut()
                .zip(&grad_w)
                .for_each(|(w, gw)| w.scaled_add(-self.eta / x_train.len() as f32, gw));

            model
                .biases
                .iter_mut()
                .zip(&grad_b)
                .for_each(|(b, gb)| b.scaled_add(-self.eta / x_train.len() as f32, gb));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn get_accuracy(y_pred: &[Array1<f32>], y_test: &[Array1<f32>]) -> f32 {
        y_pred
            .iter()
            .zip(y_test)
            .map(|(yp, y)| {
                if cost(yp.view(), y.view()) < 0.1 {
                    1.
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .sum::<f32>()
            / y_test.len() as f32
    }

    fn sigmoid(z: f32) -> f32 {
        1. / (1. + std::f32::consts::E.powf(-z))
    }

    fn sigmoid_prime(z: f32) -> f32 {
        let sigmoid = sigmoid(z);
        sigmoid * (1. - sigmoid)
    }

    fn atanh_prime(x: f32) -> f32 {
        1. / (1. - x.powi(2))
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
    fn test_converges_on_and2() {
        let mut net = Mlp::new(&[2, 5, 9, 1], sigmoid, sigmoid_prime);
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
        sgd.optimize(&mut net, &x_train, &y_train, 1000);

        let y_pred = x_train.map(|x| net.forward(x.view()));

        let accuracy = y_pred
            .iter()
            .zip(&y_train)
            .map(|(yp, y)| {
                if cost(yp.view(), y.view()) < 0.1 {
                    1.
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .sum::<f32>()
            / y_train.len() as f32;

        assert_eq!(accuracy, 1., "got: {}% accuracy", accuracy * 100.);
    }

    #[test]
    fn test_converges_on_xor2() {
        let mut net = Mlp::new(&[2, 5, 9, 1], sigmoid, sigmoid_prime);
        let x_train = [
            Array1::<f32>::from_vec(vec![0., 0.]),
            Array1::<f32>::from_vec(vec![0., 1.]),
            Array1::<f32>::from_vec(vec![1., 0.]),
            Array1::<f32>::from_vec(vec![1., 1.]),
        ];

        let y_train = [
            Array1::<f32>::from_elem(1, 0.),
            Array1::<f32>::from_elem(1, 1.),
            Array1::<f32>::from_elem(1, 1.),
            Array1::<f32>::from_elem(1, 0.),
        ];

        let sgd = Sgd { eta: 3. };
        sgd.optimize(&mut net, &x_train, &y_train, 1000);
        let y_pred = x_train.map(|x| net.forward(x.view()));
        let accuracy = get_accuracy(&y_pred, &y_train);

        assert_eq!(accuracy, 1., "got: {}% accuracy", accuracy * 100.);
    }

    #[ignore = "la sigmoid de la última capa tiene imagen [0..1] xd, después le pongo tanh"]
    #[test]
    fn test_converges_with_non_linear_function() {
        // let func = |x| 2. * f32::cos(x).powi(2);
        let func = |x| 2. * x;
        let x: Vec<_> = (0..100)
            .map(|x| Array1::<f32>::from_elem(1, x as f32))
            .collect();

        let y: Vec<_> = x.iter().map(|x| x.mapv(func)).collect();

        let train_portion = (0.8 * x.len() as f32) as usize;
        let x_train = &x[..train_portion];
        let y_train = &y[..train_portion];
        let x_test = &x[train_portion..];
        let y_test = &y[train_portion..];

        let mut net = Mlp::new(&[1, 9, 1], sigmoid, sigmoid_prime);
        let sgd = Sgd { eta: 0.1 };
        let n_iters = 10000;
        sgd.optimize(&mut net, x_train, y_train, n_iters);
        let y_pred: Vec<_> = x_test.iter().map(|x| net.forward(x.view())).collect();

        let accuracy = get_accuracy(&y_pred, y_test);

        assert!(
            accuracy >= 0.9,
            "got: {}% accuracy,\n--- first 10 y_test:\n{:?},\n--- first 10 y_pred:\n{:?}",
            accuracy * 100.,
            &y_test[..10],
            &y_pred[..10]
        );
    }
}
