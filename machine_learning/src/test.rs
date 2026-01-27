#![cfg(test)]
use rand::Rng;

use crate::{
    arch::{activations::ActFn, layers::Layer, loss::Mse, Sequential},
    dataset::Dataset,
};

#[test]
fn test_ml_and3_gate_convergence() {
    let and3 = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ];

    let dataset = Dataset::new(and3.into(), 3, 1);

    let mut rng = rand::rng();
    let mut params: Vec<f32> = (0..3).map(|_| rng.random()).collect();
    let mut grad: Vec<f32> = (0..3).map(|_| rng.random()).collect();

    let mut net = Sequential::new([Layer::dense((3, 1), ActFn::sigmoid(1, 1.))]);

    // net.backprop(&mut params, &mut grad, &Mse {}, dataset);
}
