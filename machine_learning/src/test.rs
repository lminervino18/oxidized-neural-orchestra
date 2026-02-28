#![cfg(test)]

use std::num::NonZeroUsize;

use ndarray::ArrayView2;
use rand::Rng;

use crate::{
    arch::{
        Model, Sequential,
        activations::ActFn,
        layers::Layer,
        loss::{LossFn, Mse},
    },
    dataset::Dataset,
    middleware::{ParamManager, ServerParamsMetadata},
    optimization::GradientDescent,
    training::ModelTrainer,
};

fn gen_params_grads(server_sizes: &[usize]) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut rng = rand::rng();
    server_sizes
        .iter()
        .map(|&size| {
            (
                (0..size).map(|_| rng.random_range(-0.5..0.5)).collect(),
                vec![0.0; size],
                vec![0.0; size],
            )
        })
        .collect()
}

#[test]
fn test_ml_and2_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let and2 = [
        0.0, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, // 3
        1.0, 0.0, 0.0, // 5
        1.0, 1.0, 1.0, // 8
    ];

    let mut model = Sequential::new([
        Layer::dense((2, 2), ActFn::sigmoid(1.0)),
        Layer::dense((2, 1), ActFn::sigmoid(1.0)),
    ]);

    let nparams = model.size();
    let optimizer = GradientDescent::new(1.0);
    let dataset = Dataset::new(and2.into(), 2, 1);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let loss_fn = Mse::new();
    let rng = rand::rng();

    let mut trainer = ModelTrainer::new(
        model.clone(),
        vec![optimizer],
        dataset,
        offline_epochs,
        max_epochs,
        batch_size,
        loss_fn,
        rng,
    );

    let ordering = [0, 0];
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let data = ArrayView2::from_shape((4, 3), &and2).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 2);
    let y_pred = model.forward(&mut param_manager, x).unwrap();

    println!("{y:#?}\n\n\n{y_pred:#?}");
    let loss = loss_fn.loss(y_pred, y);
    println!("err: {loss}");
}

// #[test]
// fn test_ml_and3_gate_convergence() {
//     unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

//     let and3 = [
//         0.0, 0.0, 0.0, 0.0, // 1
//         0.0, 0.0, 1.0, 0.0, // 1
//         0.0, 1.0, 0.0, 0.0, // 1
//         0.0, 1.0, 1.0, 0.0, // 1
//         1.0, 0.0, 0.0, 0.0, // 1
//         1.0, 0.0, 1.0, 0.0, // 1
//         1.0, 1.0, 0.0, 0.0, // 1
//         1.0, 1.0, 1.0, 1.0, // 1
//     ];

//     let dataset = Dataset::new(and3.into(), 3, 1);
//     // let model = Sequential::new([Layer::dense((3, 1), ActFn::step(1., 0., 0.5))]);
//     let model = Sequential::new([
//         Layer::dense((3, 2), ActFn::sigmoid(1.0)),
//         Layer::dense((2, 1), ActFn::sigmoid(1.0)),
//     ]);
//     let mut params: Vec<f32> = vec![0.; model.size()];
//     let optimizer = GradientDescent::new(1.);
//     let mut trainer = ModelTrainer::new(
//         model,
//         optimizer,
//         dataset,
//         10000 - 1,
//         NonZeroUsize::new(8).unwrap(),
//         Mse,
//         rand::rng(),
//     );
//     trainer.train(&mut params);

//     // 2

//     let mut model = Sequential::new([
//         Layer::dense((3, 2), ActFn::sigmoid(1.0)),
//         Layer::dense((2, 1), ActFn::sigmoid(1.0)),
//     ]);
//     let data = ArrayView2::from_shape((8, 4), &and3).unwrap();
//     let (x, y) = data.split_at(ndarray::Axis(1), 3);
//     let y_pred = model.forward(&params, x);

//     println!("{y:#?}\n\n\n{y_pred:#?}");
//     println!("params: {params:?}");
//     let err = (&y_pred - &y)
//         .mapv(|x| x.powi(2))
//         .mean()
//         .unwrap_or_default();

//     println!("err: {}", err);
//     // assert_eq!(err, 0.0);
// }

// #[test]
// fn test_ml_xor2_gate_convergence() {
//     unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

//     let and2 = [
//         0.0, 0.0, 0.0, // 1
//         0.0, 1.0, 1.0, // 3
//         1.0, 0.0, 1.0, // 5
//         1.0, 1.0, 0.0, // 8
//     ];

//     let dataset = Dataset::new(and2.into(), 2, 1);
//     let model = Sequential::new([
//         Layer::dense((2, 2), ActFn::sigmoid(1.)),
//         Layer::dense((2, 1), ActFn::sigmoid(1.)),
//     ]);
//     let mut rng = rand::rng();
//     let mut params: Vec<f32> = (0..model.size())
//         .map(|_| (rng.random::<f32>() - 0.5) * 2.)
//         .collect();
//     let optimizer = GradientDescent::new(1.0);
//     let mut trainer = ModelTrainer::new(
//         model,
//         optimizer,
//         dataset,
//         5000 - 1,
//         NonZeroUsize::new(4).unwrap(),
//         Mse,
//         rand::rng(),
//     );
//     trainer.train(&mut params);

//     // 2

//     let mut model = Sequential::new([
//         Layer::dense((2, 2), ActFn::sigmoid(1.)),
//         Layer::dense((2, 1), ActFn::sigmoid(1.)),
//     ]);
//     let data = ArrayView2::from_shape((4, 3), &and2).unwrap();
//     let (x, y) = data.split_at(ndarray::Axis(1), 2);
//     let y_pred = model.forward(&params, x);

//     println!("{y:#?}\n\n\n{y_pred:#?}");
//     println!("params: {params:?}");
//     let _err = (&y_pred - &y)
//         .mapv(|x| x.powi(2))
//         .mean()
//         .unwrap_or_default();

//     // assert_eq!(err, 0.0);
// }

// #[test]
// fn test_ml_xor4_gate_convergence() {
//     unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

//     // crear dataset
//     let xor4 = [
//         0.0, 0.0, 0.0, 0.0, 0.0, // 1
//         0.0, 0.0, 0.0, 1.0, 1.0, // 1
//         0.0, 0.0, 1.0, 0.0, 1.0, // 1
//         0.0, 0.0, 1.0, 1.0, 0.0, // 1
//         0.0, 1.0, 0.0, 0.0, 1.0, // 1
//         0.0, 1.0, 0.0, 1.0, 0.0, // 1
//         0.0, 1.0, 1.0, 0.0, 0.0, // 1
//         0.0, 1.0, 1.0, 1.0, 1.0, // 1
//         1.0, 0.0, 0.0, 0.0, 1.0, // 1
//         1.0, 0.0, 0.0, 1.0, 0.0, // 1
//         1.0, 0.0, 1.0, 0.0, 0.0, // 1
//         1.0, 0.0, 1.0, 1.0, 1.0, // 1
//         1.0, 1.0, 0.0, 0.0, 0.0, // 1
//         1.0, 1.0, 0.0, 1.0, 1.0, // 1
//         1.0, 1.0, 1.0, 0.0, 1.0, // 1
//         1.0, 1.0, 1.0, 1.0, 0.0, // 1
//     ];
//     let dataset = Dataset::new(xor4.into(), 4, 1);

//     // params
//     let mut model = Sequential::new([
//         Layer::dense((4, 8), ActFn::sigmoid(1.0)),
//         Layer::dense((8, 3), ActFn::sigmoid(1.0)),
//         Layer::dense((3, 1), ActFn::sigmoid(1.0)),
//     ]);

//     let mut rng = rand::rng();
//     let mut params: Vec<f32> = (0..model.size())
//         .map(|_| (rng.random::<f32>() - 0.5) * 5.)
//         .collect();

//     // training
//     let learning_rate = 1.;
//     let optimizer = GradientDescent::new(learning_rate);
//     let mut trainer = ModelTrainer::new(
//         model.clone(),
//         optimizer,
//         dataset,
//         5000 - 1,
//         NonZeroUsize::new(16).unwrap(),
//         Mse,
//         rand::rng(),
//     );
//     trainer.train(&mut params);

//     // pred
//     let data = ArrayView2::from_shape((16, 5), &xor4).unwrap();
//     let (x, y) = data.split_at(ndarray::Axis(1), 4);
//     let y_pred = model.forward(&params, x);

//     println!("{y:#?}\n\n\n{y_pred:#?}");
//     println!("params: {params:?}");
//     let err = Mse.loss(y_pred, y);

//     println!("err: {}", err);
//     // assert_eq!(err, 0.0);
// }
