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
fn test_ml_linear_convergence() {
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

    let mut param_manager = ParamManager::for_servers(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, x_size.get()), &x).unwrap();
    let y = ArrayView2::from_shape((4, y_size.get()), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
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

    let mut param_manager = ParamManager::for_servers(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, x_size.get()), &x).unwrap();
    let y = ArrayView2::from_shape((4, y_size.get()), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
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

    let mut param_manager = ParamManager::for_servers(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((8, 3), &x).unwrap();
    let y = ArrayView2::from_shape((8, 1), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
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

    let mut param_manager = ParamManager::for_servers(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, 2), &x).unwrap();
    let y = ArrayView2::from_shape((4, 1), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("{y:#?}\n\n\n{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_xor4_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [
        0.0, 0.0, 0.0, 0.0, // 1
        0.0, 0.0, 0.0, 1.0, // 2
        0.0, 0.0, 1.0, 0.0, // 3
        0.0, 0.0, 1.0, 1.0, // 4
        0.0, 1.0, 0.0, 0.0, // 5
        0.0, 1.0, 0.0, 1.0, // 6
        0.0, 1.0, 1.0, 0.0, // 7
        0.0, 1.0, 1.0, 1.0, // 8
        1.0, 0.0, 0.0, 0.0, // 9
        1.0, 0.0, 0.0, 1.0, // 10
        1.0, 0.0, 1.0, 0.0, // 11
        1.0, 0.0, 1.0, 1.0, // 12
        1.0, 1.0, 0.0, 0.0, // 13
        1.0, 1.0, 0.0, 1.0, // 14
        1.0, 1.0, 1.0, 0.0, // 15
        1.0, 1.0, 1.0, 1.0, // 16
    ];

    let y = [
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
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
    let dataset = Dataset::new(DatasetSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(16).unwrap();
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

    let ordering = [0, 0, 0];
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::for_servers(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2
    let x = ArrayView2::from_shape((16, 4), &x).unwrap();
    let y = ArrayView2::from_shape((16, 1), &y).unwrap();
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
        0.0, 1.0, 0.0, // plus sign
        0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, // dot
        1.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, //
        1.0, 0.0, 1.0, // cross
        1.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, //
        1.0, 1.0, 1.0, // box
    ];

    let labels = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];

    let mut model = Sequential::new(vec![
        Layer::two_d_to4d(1, 3, 3),
        Layer::conv2d(1, 1, 2, 1, 0),
        Layer::four_d_to2d(1, 2, 2),
        Layer::dense((4, 4)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(9).unwrap();
    let y_size = NonZeroUsize::new(4).unwrap();
    let dataset = Dataset::new(
        DatasetSrc::inmem(symbols.into(), labels.into()),
        x_size,
        y_size,
    );
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
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::for_servers(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    let x = ArrayView2::from_shape((4, 9), &symbols).unwrap();
    let y = ArrayView2::from_shape((4, 4), &labels).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("y:{y:#?}\n\n\ny_pred:{y_pred:#?}");
    println!("loss: {loss}");
}

#[test]
fn test_ml_3by3by2_symbols_convergence_with_convolutional3filters() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    // original symbol in the first channel and then inverted
    let symbols = [
        0.0, 1.0, 0.0, //
        1.0, 1.0, 1.0, //
        0.0, 1.0, 0.0, //
        //
        1.0, 0.0, 1.0, //
        0.0, 0.0, 0.0, //
        1.0, 0.0, 1.0, // plus sign
        //
        0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, //
        //
        1.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, //
        1.0, 1.0, 1.0, // dot
        //
        1.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, //
        1.0, 0.0, 1.0, //
        //
        0.0, 1.0, 0.0, //
        1.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, // cross
        //
        1.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, //
        1.0, 1.0, 1.0, //
        //
        0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, // box
    ];

    let labels = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];

    let in_channels = 2;
    let input_height = 3;
    let input_width = 3;

    let filters = 3;
    let kernel_size = 2;
    let stride = 1;
    let padding = 1;

    let output_height = 2;
    let output_width = 2;

    let mut model = Sequential::new(vec![
        Layer::two_d_to4d(in_channels, input_height, input_width),
        Layer::conv2d(filters, in_channels, kernel_size, stride, padding),
        Layer::four_d_to2d(filters, output_height, output_width),
        Layer::dense((filters * output_height * output_width, 4)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(input_height * input_width * in_channels).unwrap();
    let y_size = NonZeroUsize::new(4).unwrap();
    let dataset = Dataset::new(
        DatasetSrc::inmem(symbols.into(), labels.into()),
        x_size,
        y_size,
    );
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
        .map(|(params, grad, acc_grad_buf)| ServerParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::for_servers(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    let x = ArrayView2::from_shape((batch_size.get(), x_size.get()), &symbols).unwrap();
    let y = ArrayView2::from_shape((batch_size.get(), y_size.get()), &labels).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    println!("y:{y:#?}\n\n\ny_pred:{y_pred:#?}");
    println!("loss: {loss}");
}
