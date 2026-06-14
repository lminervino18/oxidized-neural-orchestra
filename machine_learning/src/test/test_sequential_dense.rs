use std::num::NonZeroUsize;

use comms::floats::FloatPositive;
use ndarray::ArrayView2;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    arch::{
        Sequential,
        layers::Layer,
        loss::{LossFn, Mse},
    },
    datasets::{DataSrc, Dataset},
    optimization::GradientDescent,
    param_manager::{ParamManager, ParamsMetadata},
    test::gen_params_grads,
    training::{BackpropTrainer, Trainer},
};

#[test]
fn test_machine_learning00_linear_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [0., 1., 2., 3.];
    let y = [1., 2., 3., 4.];

    let mut model = Sequential::new(vec![Layer::dense((1, 1))]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(1).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::loaded(DataSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(100).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(0.1).unwrap());
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
        .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, x_size.get()), &x).unwrap();
    let y = ArrayView2::from_shape((4, y_size.get()), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let _loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    // println!("{y:#?}\n\n\n{y_pred:#?}");
    // println!("loss: {loss}");
}

#[test]
fn test_machine_learning01_and2_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [0., 0., 0., 1., 1., 0., 1., 1.];
    let y = [0., 0., 0., 1.];

    let mut model = Sequential::new(vec![
        Layer::dense((2, 2)),
        Layer::sigmoid(1.),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(2).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::loaded(DataSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(1.).unwrap());
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
        .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, x_size.get()), &x).unwrap();
    let y = ArrayView2::from_shape((4, y_size.get()), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let _loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    // println!("{y:#?}\n\n\n{y_pred:#?}");
    // println!("loss: {loss}");
}

#[test]
fn test_machine_learning02_and3_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [
        0., 0., 0., // 1
        0., 0., 1., // 1
        0., 1., 0., // 1
        0., 1., 1., // 1
        1., 0., 0., // 1
        1., 0., 1., // 1
        1., 1., 0., // 1
        1., 1., 1., // 1
    ];
    let y = [0., 0., 0., 0., 0., 0., 0., 1.];

    let mut model = Sequential::new(vec![
        Layer::dense((3, 2)),
        Layer::sigmoid(1.),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(3).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::loaded(DataSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(2000).unwrap();
    let batch_size = NonZeroUsize::new(8).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(1.).unwrap());
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
        .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((8, 3), &x).unwrap();
    let y = ArrayView2::from_shape((8, 1), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let _loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    // println!("{y:#?}\n\n\n{y_pred:#?}");
    // println!("loss: {loss}");
}

#[test]
fn test_machine_learning03_xor2_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [
        0., 0., // 1
        0., 1., // 3
        1., 0., // 5
        1., 1., // 8
    ];
    let y = [0., 1., 1., 0.];

    let mut model = Sequential::new(vec![
        Layer::dense((2, 2)),
        Layer::sigmoid(1.),
        Layer::dense((2, 1)),
        Layer::sigmoid(1.),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(2).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::loaded(DataSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(1.).unwrap());
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
        .map(|(params, grad, residual)| ParamsMetadata::new(params, grad, residual))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2

    let x = ArrayView2::from_shape((4, 2), &x).unwrap();
    let y = ArrayView2::from_shape((4, 1), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let _loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    // println!("{y:#?}\n\n\n{y_pred:#?}");
    // println!("loss: {loss}");
}

#[test]
fn test_machine_learning04_xor4_gate_convergence() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let x = [
        0., 0., 0., 0., // 1
        0., 0., 0., 1., // 2
        0., 0., 1., 0., // 3
        0., 0., 1., 1., // 4
        0., 1., 0., 0., // 5
        0., 1., 0., 1., // 6
        0., 1., 1., 0., // 7
        0., 1., 1., 1., // 8
        1., 0., 0., 0., // 9
        1., 0., 0., 1., // 10
        1., 0., 1., 0., // 11
        1., 0., 1., 1., // 12
        1., 1., 0., 0., // 13
        1., 1., 0., 1., // 14
        1., 1., 1., 0., // 15
        1., 1., 1., 1., // 16
    ];

    let y = [
        0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0.,
    ];

    let mut model = Sequential::new(vec![
        Layer::dense((4, 8)),
        Layer::sigmoid(1.),
        Layer::dense((8, 3)),
        Layer::sigmoid(1.),
        Layer::dense((3, 1)),
        Layer::sigmoid(1.),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(4).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let dataset = Dataset::loaded(DataSrc::inmem(x.into(), y.into()), x_size, y_size);
    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(1000).unwrap();
    let batch_size = NonZeroUsize::new(16).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(1.).unwrap());
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
        .map(|(params, grad, acc_grad_buf)| ParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    // 2
    let x = ArrayView2::from_shape((16, 4), &x).unwrap();
    let y = ArrayView2::from_shape((16, 1), &y).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let _loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    // println!("{y:#?}\n\n\n{y_pred:#?}");
    // println!("loss: {loss}");
}
