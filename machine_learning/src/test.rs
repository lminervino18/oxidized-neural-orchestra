#![cfg(test)]

use ndarray::ArrayView2;

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
        0.0, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, // 3
        1.0, 0.0, 0.0, // 5
        1.0, 1.0, 1.0, // 8
    ];

    let dataset = Dataset::new(and3.into(), 2, 1);
    let model = Sequential::new([Layer::dense((2, 1), ActFn::sigmoid(1.))]);
    let mut params: Vec<f32> = vec![0.; model.size()];
    let optimizer = GradientDescent::new(2.);
    let mut trainer = Trainer::new(model, optimizer, dataset, 2000, 4, Mse, rand::rng());
    trainer.train(&mut params);

    // 2

    let mut model = Sequential::new([Layer::dense((2, 1), ActFn::sigmoid(1.))]);
    let data = ArrayView2::from_shape((4, 3), &and3).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 2);
    let y_pred = model.forward(&params, x);

    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("params: {params:?}");
}
