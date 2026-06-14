use std::{env, fs::File, io::Read, num::NonZeroUsize, path::PathBuf};

use comms::floats::FloatPositive;
use machine_learning::{
    arch::{Sequential, layers::Layer, loss::CrossEntropy},
    dataset::{Dataset, DatasetSrc},
    models::{make_nielsen_mnist_model, some_other_mnist_model},
    optimization::GradientDescent,
    param_manager::{ParamManager, ParamsMetadata},
    training::{BackpropTrainer, TrainResult, Trainer},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

const MNIST_DIR_STR: &str = "datasets/mnist/";
const TRAIN_SAMPLES_STR: &str = "mnist_train_samples.bin";
const TRAIN_LABELS_STR: &str = "mnist_train_labels.bin";
const TEST_SAMPLES_STR: &str = "mnist_test_samples.bin";
const TEST_LABELS_STR: &str = "mnist_test_labels.bin";

const IN_CHANNELS: usize = 1;
const INPUT_HEIGHT: usize = 28;
const INPUT_WIDTH: usize = 28;
const SAMPLE_SIZE: usize = IN_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
const LABEL_SIZE: usize = 10;

fn main() {
    unsafe {
        env::set_var("RUST_BACKTRACE", "1");
    }

    let mut model = make_nielsen_mnist_model();

    let train_size = None; // whole dataset
    let train_dataset = make_mnist_dataset(train_size, TRAIN_SAMPLES_STR, TRAIN_LABELS_STR);

    let learning_rate = FloatPositive::new(0.1).unwrap();
    let optimizer = GradientDescent::new(learning_rate);
    let loss_fn = CrossEntropy::new();
    let epochs = NonZeroUsize::new(60).unwrap();
    let batch_size = NonZeroUsize::new(10).unwrap();
    let rng = StdRng::from_os_rng();

    let mut trainer = BackpropTrainer::new(
        model.clone(),
        vec![optimizer],
        train_dataset,
        loss_fn,
        0,
        epochs,
        batch_size,
        rng,
    );

    let nparams = model.size();
    let mut params_grads = gen_params_grads(&[nparams]);
    let servers: Vec<_> = params_grads
        .iter_mut()
        .map(|(params, grad, acc_grad_buf)| ParamsMetadata::new(params, grad, acc_grad_buf))
        .collect();

    let ordering = [0, 0, 0];
    let mut param_manager = ParamManager::for_parameter_server(servers, &ordering);

    let mut epoch = 0;
    let epochs_until_log = 1;
    loop {
        let TrainResult { losses, was_last } = trainer.train(&mut param_manager).unwrap();
        let loss = losses.last().unwrap();

        if epoch % epochs_until_log == 0 {
            println!("epoch: {epoch}, loss: {loss}")
        }
        epoch += 1;

        if was_last {
            break;
        }
    }

    let test_size = None; // whole dataset
    let test_dataset = make_mnist_dataset(test_size, TEST_SAMPLES_STR, TEST_LABELS_STR); // whole
    let test_batches = test_dataset.batches(NonZeroUsize::new(1).unwrap());

    let mut got_right = 0;
    for (x, y) in test_batches {
        let mut pred = model
            .forward(&mut param_manager, x.into_dyn())
            .unwrap()
            .to_owned();

        let (max_idx, _) = pred
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        pred.iter_mut().enumerate().for_each(|(idx, y)| {
            if idx == max_idx {
                *y = 1.;
            } else {
                *y = 0.;
            }
        });

        if pred == y.into_dyn() {
            got_right += 1;
        }
    }

    // let mut params_iter = param_manager.front();
    // let params = params_iter.next(nparams);
    // println!("params: {params:#?}");

    let acc_percentage = (got_right as f32 / test_dataset.rows() as f32) * 100.;
    println!("accuracy: {acc_percentage}%");
}

fn gen_params_grads(server_sizes: &[usize]) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut rng = rand::rng();
    server_sizes
        .iter()
        .map(|&size| {
            (
                (0..size).map(|_| rng.random_range(-0.5..0.5)).collect(),
                vec![0.; size],
                vec![0.; size],
            )
        })
        .collect()
}

fn make_mnist_dataset(size: Option<usize>, samples_path: &str, labels_path: &str) -> Dataset {
    let dir_path = PathBuf::from(MNIST_DIR_STR);
    let mut samples_file = File::open(dir_path.join(samples_path)).unwrap();
    let mut labels_file = File::open(dir_path.join(labels_path)).unwrap();

    let mut samples_buf = vec![];
    let mut labels_buf = vec![];

    if let Some(size) = size {
        samples_buf = vec![0; size * SAMPLE_SIZE * size_of::<f32>()];
        labels_buf = vec![0; size * LABEL_SIZE * size_of::<f32>()];

        samples_file.read_exact(&mut samples_buf).unwrap();
        labels_file.read_exact(&mut labels_buf).unwrap();
    } else {
        samples_file.read_to_end(&mut samples_buf).unwrap();
        labels_file.read_to_end(&mut labels_buf).unwrap();
    };

    let samples: Vec<f32> = samples_buf
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let labels: Vec<f32> = labels_buf
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();

    Dataset::new(
        DatasetSrc::inmem(samples, labels),
        NonZeroUsize::new(SAMPLE_SIZE).unwrap(),
        NonZeroUsize::new(LABEL_SIZE).unwrap(),
    )
}
