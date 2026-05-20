use std::num::NonZeroUsize;

use comms::floats::FloatPositive;
use ndarray::ArrayView2;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    arch::{
        Sequential,
        layers::Layer,
        loss::{CrossEntropy, LossFn, Mse},
    },
    dataset::{Dataset, DatasetSrc},
    optimization::GradientDescent,
    param_manager::{ParamManager, ParamsMetadata},
    test::gen_params_grads,
    training::{BackpropTrainer, Trainer},
};

#[allow(clippy::too_many_arguments)]
fn test_conv_dense(
    filters: usize,
    in_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    symbols: &[f32],
    input_height: usize,
    input_width: usize,
    labels: &[f32],
    y_size: usize,
) {
    let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    let mut model = Sequential::new(vec![
        Layer::two_d_to4d(in_channels, input_height, input_width),
        Layer::conv2d(filters, in_channels, kernel_size, stride, padding),
        Layer::four_d_to2d(filters, output_height, output_width),
        Layer::dense((filters * output_height * output_width, y_size)),
        Layer::softmax(),
    ]);
    let nparams = model.size();

    let x_size = NonZeroUsize::new(input_height * input_width * in_channels).unwrap();
    let y_size = NonZeroUsize::new(y_size).unwrap();
    let dataset = Dataset::new(
        DatasetSrc::inmem(symbols.into(), labels.into()),
        x_size,
        y_size,
    );

    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(300).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(1.).unwrap());
    let mut loss_fn = CrossEntropy::new();
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
        .map(|(params, grad, acc_grad_buf)| ParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    while !trainer.train(&mut param_manager).unwrap().was_last {}

    let x = ArrayView2::from_shape((y_size.get(), x_size.get()), symbols).unwrap();
    let y = ArrayView2::from_shape((y_size.get(), y_size.get()), labels).unwrap();
    let y_pred = model.forward(&mut param_manager, x.into_dyn()).unwrap();

    let loss = loss_fn.loss(y_pred.view(), y.into_dyn());
    // println!("y:{y:#?}\n\n\ny_pred:{y_pred:#?}");
    // println!("loss: {loss}");
    assert!(loss < 0.001);
}

const NSYMBOLS: usize = 4;
const SYMBOLS_IN_CHANNELS: usize = 1;
const SYMBOLS_HEIGHT: usize = 3;
const SYMBOLS_WIDTH: usize = 3;
const SYMBOLS: [f32; SYMBOLS_HEIGHT * SYMBOLS_WIDTH * SYMBOLS_IN_CHANNELS * NSYMBOLS] = [
    0., 1., 0., //
    1., 1., 1., //
    0., 1., 0., // plus sign
    0., 0., 0., //
    0., 1., 0., //
    0., 0., 0., // dot
    1., 0., 1., //
    0., 1., 0., //
    1., 0., 1., // cross
    1., 1., 1., //
    1., 0., 1., //
    1., 1., 1., // box
];
const SYMBOL_LABELS_SIZE: usize = 4;
const SYMBOL_LABELS: [f32; SYMBOL_LABELS_SIZE * NSYMBOLS] = [
    1., 0., 0., 0., //
    0., 1., 0., 0., //
    0., 0., 1., 0., //
    0., 0., 0., 1., //
];

#[test]
fn test_machine_learning05_3by3_symbols_convergence_with_convolutional() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let filters = 1;
    let kernel_size = 2;
    let stride = 1;
    let padding = 0;

    test_conv_dense(
        filters,
        SYMBOLS_IN_CHANNELS,
        kernel_size,
        stride,
        padding,
        &SYMBOLS,
        SYMBOLS_HEIGHT,
        SYMBOLS_WIDTH,
        &SYMBOL_LABELS,
        SYMBOL_LABELS_SIZE,
    );
}

const SYMBOLS_2_CHANN_IN_CHANNELS: usize = 2;
const SYMBOLS_2_CHANN: [f32; SYMBOLS_HEIGHT
    * SYMBOLS_WIDTH
    * NSYMBOLS
    * SYMBOLS_2_CHANN_IN_CHANNELS] = [
    0., 1., 0., //
    1., 1., 1., //
    0., 1., 0., //
    //
    1., 0., 1., //
    0., 0., 0., //
    1., 0., 1., // plus sign
    //
    0., 0., 0., //
    0., 1., 0., //
    0., 0., 0., //
    //
    1., 1., 1., //
    1., 0., 1., //
    1., 1., 1., // dot
    //
    1., 0., 1., //
    0., 1., 0., //
    1., 0., 1., //
    //
    0., 1., 0., //
    1., 0., 1., //
    0., 1., 0., // cross
    //
    1., 1., 1., //
    1., 0., 1., //
    1., 1., 1., //
    //
    0., 0., 0., //
    0., 1., 0., //
    0., 0., 0., // box
];

#[test]
fn test_machine_learning06_003by3by2_symbols_convergence_with_convolutional3filters() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let filters = 1;
    let kernel_size = 2;
    let stride = 1;
    let padding = 0;

    test_conv_dense(
        filters,
        SYMBOLS_2_CHANN_IN_CHANNELS,
        kernel_size,
        stride,
        padding,
        &SYMBOLS_2_CHANN,
        SYMBOLS_HEIGHT,
        SYMBOLS_WIDTH,
        &SYMBOL_LABELS,
        SYMBOL_LABELS_SIZE,
    );
}

#[test]
fn test_machine_learning07_3by3by2_symbols_convergence_with_convolutional3filters_and_padding1() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let filters = 3;
    let kernel_size = 2;
    let stride = 1;
    let padding = 1;

    test_conv_dense(
        filters,
        SYMBOLS_2_CHANN_IN_CHANNELS,
        kernel_size,
        stride,
        padding,
        &SYMBOLS_2_CHANN,
        SYMBOLS_HEIGHT,
        SYMBOLS_WIDTH,
        &SYMBOL_LABELS,
        SYMBOL_LABELS_SIZE,
    );
}

#[test]
fn test_machine_learning08_3by3by2_filters1_kernel_size3_stride1_padding1() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let filters = 1;
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;

    test_conv_dense(
        filters,
        SYMBOLS_IN_CHANNELS,
        kernel_size,
        stride,
        padding,
        &SYMBOLS,
        SYMBOLS_HEIGHT,
        SYMBOLS_WIDTH,
        &SYMBOL_LABELS,
        SYMBOL_LABELS_SIZE,
    );
}

#[test]
fn test_machine_learning_dimensionality_correctness() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let nsamples = 10;

    let in_channels = 1;
    let input_height = 28;
    let input_width = 28;

    let x_size = in_channels * input_height * input_width;
    let y_size = 10;

    let samples = vec![1.; nsamples * x_size];
    let labels = vec![1.; nsamples * y_size];

    let filters = 4;
    let kernel_size = 3;
    let stride = 2;
    let padding = 1;

    let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    let model = Sequential::new(vec![
        Layer::two_d_to4d(in_channels, input_height, input_width),
        Layer::conv2d(filters, in_channels, kernel_size, stride, padding),
        Layer::four_d_to2d(filters, output_height, output_width),
        Layer::dense((filters * output_height * output_width, y_size)),
        Layer::sigmoid(1.0),
    ]);
    let nparams = model.size();
    let nreal_layers = 2;
    let ordering = vec![0; nreal_layers];

    let dataset = Dataset::new(
        DatasetSrc::inmem(samples, labels),
        NonZeroUsize::new(x_size).unwrap(),
        NonZeroUsize::new(y_size).unwrap(),
    );

    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(10).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(1.).unwrap());
    let loss_fn = Mse::new();
    let rng = StdRng::from_os_rng();

    let mut trainer = BackpropTrainer::new(
        model,
        vec![optimizer],
        dataset,
        loss_fn,
        offline_epochs,
        max_epochs,
        batch_size,
        rng,
    );

    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, acc_grad_buf)| ParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    assert!(trainer.train(&mut param_manager).is_ok())
}

#[test]
fn test_machine_learning_dimensionality_correctness_with_max_pooling() {
    unsafe { std::env::set_var("RUST_BACKTRACE", "1") };

    let nsamples = 10;

    let in_channels = 1;
    let input_height = 28;
    let input_width = 28;

    let x_size = in_channels * input_height * input_width;
    let y_size = 10;

    let samples = vec![1.; nsamples * x_size];
    let labels = vec![1.; nsamples * y_size];

    let filters = 4;
    let kernel_size = 3;
    let stride = 2;
    let padding = 1;

    let pooling_filter_size = 3;
    let pooling_stride = 3;
    let pooling_padding = 0;

    let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    let pooling_output_height =
        (output_height + 2 * pooling_padding - pooling_filter_size) / pooling_stride + 1;
    let pooling_output_width =
        (output_width + 2 * pooling_padding - pooling_filter_size) / pooling_stride + 1;

    let model = Sequential::new(vec![
        Layer::two_d_to4d(in_channels, input_height, input_width),
        Layer::conv2d(filters, in_channels, kernel_size, stride, padding),
        Layer::max_pooling(pooling_filter_size, pooling_stride, pooling_padding),
        Layer::four_d_to2d(filters, pooling_output_height, pooling_output_width),
        Layer::dense((
            filters * pooling_output_height * pooling_output_width,
            y_size,
        )),
        Layer::sigmoid(1.),
    ]);
    let nparams = model.size();
    let nreal_layers = 2;
    let ordering = vec![0; nreal_layers];

    let dataset = Dataset::new(
        DatasetSrc::inmem(samples, labels),
        NonZeroUsize::new(x_size).unwrap(),
        NonZeroUsize::new(y_size).unwrap(),
    );

    let offline_epochs = 0;
    let max_epochs = NonZeroUsize::new(10).unwrap();
    let batch_size = NonZeroUsize::new(4).unwrap();
    let optimizer = GradientDescent::new(FloatPositive::new(1.).unwrap());
    let loss_fn = Mse::new();
    let rng = StdRng::from_os_rng();

    let mut trainer = BackpropTrainer::new(
        model,
        vec![optimizer],
        dataset,
        loss_fn,
        offline_epochs,
        max_epochs,
        batch_size,
        rng,
    );

    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, acc_grad_buf)| ParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);
    assert!(trainer.train(&mut param_manager).is_ok())
}
