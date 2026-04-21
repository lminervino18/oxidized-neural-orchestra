#![cfg(test)]

use std::num::NonZeroUsize;

use ndarray::ArrayView2;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    arch::{
        Sequential,
        layers::Layer,
        loss::{LossFn, Mse},
    },
    dataset::{Dataset, DatasetSrc},
    optimization::GradientDescent,
    param_manager::{ParamManager, ServerParamsMetadata},
    training::{BackpropTrainer, Trainer},
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
fn test_ml_lineal_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [0., 1., 2., 3.];
    let y = [1., 2., 3., 4.];

    let mut model = Sequential::new(vec![Layer::dense((1, 1))]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(1).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(100).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(0.1);
    let mut loss_fn = Mse::new();
    let rng = StdRng::from_os_rng();

    let mut trainer = BackpropTrainer::new(
        model.clone(),
        vec![optimizer],
        dataset,
        loss_fn.clone(),
        offline_epochs,
        max_epochs,
        batch_size,
        rng,
    );

    let ordering = [0];
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, residual)| ServerParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, x_size.get()), &x).unwrap();
    let y = ArrayView2::from_shape((4, y_size.get()), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x).unwrap();

    let loss = loss_fn.loss(y_pred, y);
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_and2_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y = [0.0, 0.0, 0.0, 1.0];

    let mut model = Sequential::new(vec![
        Layer::dense((2, 2)),
        Layer::sigmoid(1.0),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(2).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = StdRng::from_os_rng();

    let mut trainer = BackpropTrainer::new(
        model.clone(),
        vec![optimizer],
        dataset,
        loss_fn.clone(),
        offline_epochs,
        max_epochs,
        batch_size,
        rng,
    );

    let ordering = [0, 0];
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, residual)| ServerParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, x_size.get()), &x).unwrap();
    let y = ArrayView2::from_shape((4, y_size.get()), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x).unwrap();

    let loss = loss_fn.loss(y_pred, y);
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_and3_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [
        0.0, 0.0, 0.0, // 1
        0.0, 0.0, 1.0, // 1
        0.0, 1.0, 0.0, // 1
        0.0, 1.0, 1.0, // 1
        1.0, 0.0, 0.0, // 1
        1.0, 0.0, 1.0, // 1
        1.0, 1.0, 0.0, // 1
        1.0, 1.0, 1.0, // 1
    ];
    let y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let mut model = Sequential::new(vec![
        Layer::dense((3, 2)),
        Layer::sigmoid(1.0),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(3).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(2000).unwrap();
    let batch_size = NonZeroUsize::new(8).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = StdRng::from_os_rng();

    let mut trainer = BackpropTrainer::new(
        model.clone(),
        vec![optimizer],
        dataset,
        loss_fn.clone(),
        offline_epochs,
        max_epochs,
        batch_size,
        rng,
    );

    let ordering = [0, 0];
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, residual)| ServerParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((8, 3), &x).unwrap();
    let y = ArrayView2::from_shape((8, 1), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x).unwrap();

    let loss = loss_fn.loss(y_pred, y);
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_xor2_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [
        0.0, 0.0, // 1
        0.0, 1.0, // 3
        1.0, 0.0, // 5
        1.0, 1.0, // 8
    ];
    let y = [0.0, 1.0, 1.0, 0.0];

    let mut model = Sequential::new(vec![
        Layer::dense((2, 2)),
        Layer::sigmoid(1.0),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(2).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = StdRng::from_os_rng();

    let mut trainer = BackpropTrainer::new(
        model.clone(),
        vec![optimizer],
        dataset,
        loss_fn.clone(),
        offline_epochs,
        max_epochs,
        batch_size,
        rng,
    );

    let ordering = [0, 0];
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, residual)| ServerParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, 2), &x).unwrap();
    let y = ArrayView2::from_shape((4, 1), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x).unwrap();

    let loss = loss_fn.loss(y_pred, y);
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

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

//     let loss = Mse.loss(y_pred, y);
//     println!("{y:#?}\n\n\n{y_pred:#?}");
//     println!("loss: {loss}");
// }
