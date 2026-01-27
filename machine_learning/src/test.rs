#![cfg(test)]

use ndarray::ArrayView2;
use rand::Rng;

use crate::{
    arch::{activations::ActFn, layers::Layer, loss::Mse, Model, Sequential},
    dataset::Dataset,
    optimization::GradientDescent,
    training::Trainer,
};

#[test]
fn test_ml_and3_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let and3 = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ];

    let dataset = Dataset::new(and3.into(), 3, 1);
    let mut rng = rand::rng();

    let model = Sequential::new([Layer::dense((3, 1), ActFn::sigmoid(1.))]);
    let mut params: Vec<f32> = (0..model.size()).map(|_| rng.random()).collect();
    let optimizer = GradientDescent::new(0.1);
    let mut trainer = Trainer::new(model, optimizer, dataset, 20, 8, Mse, rng);
    trainer.train(&mut params);

    // 2

    let mut model = Sequential::new([Layer::dense((3, 1), ActFn::sigmoid(1.))]);

    let data = ArrayView2::from_shape((8, 4), &and3).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 2);

    let y_pred = model.forward(&params, x);

    println!("{y:#?}\n\n\n{y_pred:#?}");
}
