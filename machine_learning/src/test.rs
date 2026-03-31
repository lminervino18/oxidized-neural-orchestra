#![cfg(test)]

use std::num::NonZeroUsize;

use ndarray::ArrayView2;
use rand::Rng;

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
fn test_ml_linear_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let linear = [
        0.0, 1.0, // 2
        1.0, 2.0, // 4
        2.0, 3.0, // 6
        3.0, 4.0, // 8
    ];

    let mut model = Sequential::new(vec![Layer::dense((1, 1))]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(linear.into()), x_size, x_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(100).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(0.1);
    let mut loss_fn = Mse::new();
    let rng = rand::rng();

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
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let data = ArrayView2::from_shape((4, 2), &linear).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 1);
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
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

    let mut model = Sequential::new(vec![
        Layer::dense((2, 2)),
        Layer::sigmoid(1.0),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(2).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(and2.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = rand::rng();

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
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let data = ArrayView2::from_shape((4, 3), &and2).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 2);
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_and3_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let and3 = [
        0.0, 0.0, 0.0, 0.0, // 1
        0.0, 0.0, 1.0, 0.0, // 1
        0.0, 1.0, 0.0, 0.0, // 1
        0.0, 1.0, 1.0, 0.0, // 1
        1.0, 0.0, 0.0, 0.0, // 1
        1.0, 0.0, 1.0, 0.0, // 1
        1.0, 1.0, 0.0, 0.0, // 1
        1.0, 1.0, 1.0, 1.0, // 1
    ];

    let mut model = Sequential::new(vec![
        Layer::dense((3, 2)),
        Layer::sigmoid(1.0),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(3).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(and3.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(2000).unwrap();
    let batch_size = NonZeroUsize::new(8).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = rand::rng();

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
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let data = ArrayView2::from_shape((8, 4), &and3).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 3);
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_xor2_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let xor2 = [
        0.0, 0.0, 0.0, // 1
        0.0, 1.0, 1.0, // 3
        1.0, 0.0, 1.0, // 5
        1.0, 1.0, 0.0, // 8
    ];

    let mut model = Sequential::new(vec![
        Layer::dense((2, 2)),
        Layer::sigmoid(1.0),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(2).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(xor2.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = rand::rng();

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
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let data = ArrayView2::from_shape((4, 3), &xor2).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 2);
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_xor4_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let xor4 = [
        0.0, 0.0, 0.0, 0.0, 0.0, // 1
        0.0, 0.0, 0.0, 1.0, 1.0, // 2
        0.0, 0.0, 1.0, 0.0, 1.0, // 3
        0.0, 0.0, 1.0, 1.0, 0.0, // 4
        0.0, 1.0, 0.0, 0.0, 1.0, // 5
        0.0, 1.0, 0.0, 1.0, 0.0, // 6
        0.0, 1.0, 1.0, 0.0, 0.0, // 7
        0.0, 1.0, 1.0, 1.0, 1.0, // 8
        1.0, 0.0, 0.0, 0.0, 1.0, // 9
        1.0, 0.0, 0.0, 1.0, 0.0, // 10
        1.0, 0.0, 1.0, 0.0, 0.0, // 11
        1.0, 0.0, 1.0, 1.0, 1.0, // 12
        1.0, 1.0, 0.0, 0.0, 0.0, // 13
        1.0, 1.0, 0.0, 1.0, 1.0, // 14
        1.0, 1.0, 1.0, 0.0, 1.0, // 15
        1.0, 1.0, 1.0, 1.0, 0.0, // 16
    ];

    let mut model = Sequential::new(vec![
        Layer::dense((4, 8)),
        Layer::sigmoid(1.0),
        Layer::dense((8, 3)),
        Layer::sigmoid(1.0),
        Layer::dense((3, 1)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(4).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(xor4.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(16).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = rand::rng();

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

    let ordering = [0, 0, 0];
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2
    let data = ArrayView2::from_shape((16, 5), &xor4).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 4);
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_3by3_symbols_convergence_with_convolutional() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let symbols = [
        0.0, 1.0, 0.0, //
        1.0, 1.0, 1.0, //
        0.0, 1.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, // plus sign
        0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, // dot
        1.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, //
        1.0, 0.0, 1.0, //
        0.0, 0.0, 1.0, 0.0, // cross
        1.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, //
        1.0, 1.0, 1.0, //
        0.0, 0.0, 0.0, 1.0,
        // box
    ];

    let mut model = Sequential::new(vec![
        Layer::two_d_to4d(1, 3, 3),
        Layer::conv2d(1, 1, (2, 2), 1, 0),
        Layer::four_d_to2d(1, 2, 2),
        Layer::dense((4, 4)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(9).unwrap();
    let y_size = NonZeroUsize::new(4).unwrap();
    let dataset = Dataset::new(DatasetSrc::inmem(symbols.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(1.0);
    let mut loss_fn = Mse::new();
    let rng = rand::rng();

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
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::new(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2
    let data = ArrayView2::from_shape((4, 9), &symbols).unwrap();
    let (x, y) = data.split_at(ndarray::Axis(1), 9);
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}
