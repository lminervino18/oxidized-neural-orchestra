use comms::specs::machine_learning::{
    ActFnSpec, DatasetSpec, LayerSpec, LossFnSpec, OptimizerSpec, TrainerSpec,
};
use rand::{SeedableRng, rngs::StdRng};

use super::{BackpropTrainer, Trainer};
use crate::{
    arch::{
        Sequential,
        layers::{Inner, Layer},
        loss::{CrossEntropy, LossFn, Mse},
    },
    datasets::Dataset,
    optimization::{Adam, GradientDescent, GradientDescentWithMomentum, Optimizer},
};

/// Builds `Trainer`s given a specification.
#[derive(Default)]
pub struct TrainerBuilder;

impl TrainerBuilder {
    /// Creates a new `TrainerBuilder`.
    pub fn new() -> Self {
        Self
    }

    /// Builds a new `Trainer` following a spec.
    ///
    /// # Args
    /// * `spec` - The specification for the trainer.
    /// * `server_sizes` - The mount of parameters per server.
    ///
    /// # Returns
    /// A new `Trainer`.
    pub fn build(&self, spec: TrainerSpec, server_sizes: &[usize]) -> Box<dyn Trainer> {
        self.resolve_optimizers(spec, server_sizes)
    }

    /// Resolves the `Optimizer`s for this trainer.
    ///
    /// # Args
    /// * `spec` - The specification of the trainer.
    /// * `server_sizes` - The amount of parameters per server.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_optimizers(&self, spec: TrainerSpec, server_sizes: &[usize]) -> Box<dyn Trainer> {
        match spec.optimizer {
            OptimizerSpec::GradientDescent { learning_rate } => {
                let optimizers: Vec<_> = server_sizes
                    .iter()
                    .map(|_| GradientDescent::new(learning_rate))
                    .collect();

                self.resolve_layers(spec, optimizers)
            }
            OptimizerSpec::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                let optimizers: Vec<_> = server_sizes
                    .iter()
                    .map(|&len| Adam::new(len, learning_rate, beta1, beta2, epsilon))
                    .collect();

                self.resolve_layers(spec, optimizers)
            }
            OptimizerSpec::GradientDescentWithMomentum {
                learning_rate,
                momentum,
            } => {
                let optimizers: Vec<_> = server_sizes
                    .iter()
                    .map(|&len| GradientDescentWithMomentum::new(len, learning_rate, momentum))
                    .collect();

                self.resolve_layers(spec, optimizers)
            }
        }
    }

    /// Resolves the the `Layer`s for a `Sequential` model.
    ///
    /// # Args
    /// * `spec` - The specification of the trainer.
    /// * `optimizers` - A list of resolved optimizers.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_layers<O>(&self, spec: TrainerSpec, optimizers: Vec<O>) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
    {
        let mut layers = vec![];

        let mut last = None;
        for spec in &spec.layers {
            self.resolve_layer_into(*spec, last, &mut layers);
            last = Some(*spec);
        }

        self.resolve_loss_fn(spec, optimizers, layers)
    }

    /// Resolves a `Layer` for a `Sequential` model.
    ///
    /// # Args
    /// * `spec` - The specification of a certain layer.
    ///
    /// # Returns
    /// A new `Layer`.
    fn resolve_layer_into(
        &self,
        spec: LayerSpec,
        last: Option<LayerSpec>,
        layers: &mut Vec<Layer>,
    ) {
        use Inner::*;

        let act_fn = match spec {
            LayerSpec::Dense { dim, act_fn } => {
                if matches!(layers.last(), Some(Layer(Conv2d(_) | MaxPooling(_)))) {
                    let last = last.unwrap();
                    let (out_c, out_h, out_w) = match last {
                        LayerSpec::Conv {
                            input_dim,
                            kernel_dim,
                            stride,
                            padding,
                            ..
                        } => self.spatial_size(
                            input_dim,
                            kernel_dim.2,
                            stride,
                            padding,
                            kernel_dim.0,
                        ),
                        LayerSpec::MaxPooling {
                            input_dim,
                            filter_size,
                            stride,
                            padding,
                            ..
                        } => {
                            self.spatial_size(input_dim, filter_size, stride, padding, input_dim.2)
                        }
                        _ => unreachable!(),
                    };
                    layers.push(Layer::four_d_to2d(out_c, out_h, out_w))
                }

                layers.push(Layer::dense(dim));
                act_fn
            }
            LayerSpec::Conv {
                input_dim,
                kernel_dim,
                stride,
                padding,
                act_fn,
            } => {
                if matches!(layers.last(), Some(Layer(Conv2d(_) | MaxPooling(_)))) {
                    layers.push(Layer::two_d_to4d(input_dim.0, input_dim.1, input_dim.2))
                }

                layers.push(Layer::conv2d(
                    kernel_dim.0,
                    kernel_dim.1,
                    kernel_dim.2,
                    stride,
                    padding,
                ));

                if act_fn.is_some() {
                    let out_h = (input_dim.1 + 2 * padding - kernel_dim.2) / stride + 1;
                    let out_w = (input_dim.2 + 2 * padding - kernel_dim.2) / stride + 1;
                    layers.push(Layer::four_d_to2d(kernel_dim.0, out_h, out_w))
                }

                act_fn
            }
            LayerSpec::MaxPooling {
                input_dim,
                filter_size,
                stride,
                padding,
                act_fn,
            } => {
                if !matches!(layers.last(), Some(Layer(Conv2d(_) | MaxPooling(_)))) {
                    layers.push(Layer::two_d_to4d(input_dim.0, input_dim.1, input_dim.2))
                }

                layers.push(Layer::max_pooling(filter_size, stride, padding));

                if act_fn.is_some() {
                    let out_h = (input_dim.1 + 2 * padding - filter_size) / stride + 1;
                    let out_w = (input_dim.2 + 2 * padding - filter_size) / stride + 1;
                    layers.push(Layer::four_d_to2d(input_dim.0, out_h, out_w))
                }

                act_fn
            }
        };

        if let Some(spec) = act_fn {
            if matches!(layers.last(), Some(Layer(Conv2d(_) | MaxPooling(_)))) {
                let last = last.unwrap();
                let (out_c, out_h, out_w) = match last {
                    LayerSpec::Conv {
                        input_dim,
                        kernel_dim,
                        stride,
                        padding,
                        ..
                    } => self.spatial_size(input_dim, kernel_dim.2, stride, padding, kernel_dim.0),
                    LayerSpec::MaxPooling {
                        input_dim,
                        filter_size,
                        stride,
                        padding,
                        ..
                    } => self.spatial_size(input_dim, filter_size, stride, padding, input_dim.2),
                    _ => unreachable!(),
                };
                layers.push(Layer::four_d_to2d(out_c, out_h, out_w))
            }
            layers.push(self.resolve_act_fn(spec));
        };
    }

    fn spatial_size(
        &self,
        input_dim: (usize, usize, usize),
        square_filter_size: usize,
        stride: usize,
        padding: usize,
        out_channels: usize,
    ) -> (usize, usize, usize) {
        let calc_dim = |in_dim: usize| (in_dim + 2 * padding) - (square_filter_size) / stride + 1;

        let output_height = calc_dim(input_dim.1);
        let output_width = calc_dim(input_dim.2);

        (output_height, output_width, out_channels)
    }

    /// Resolves the `ActFn` for a specific layer.
    ///
    /// # Args
    /// * `spec` - An optional specification for an `ActFn`.
    ///
    /// # Returns
    /// A new `Layer`.
    fn resolve_act_fn(&self, spec: ActFnSpec) -> Layer {
        match spec {
            ActFnSpec::Sigmoid { amp } => Layer::sigmoid(amp),
            ActFnSpec::Softmax => Layer::softmax(),
            ActFnSpec::Tanh { amp } => Layer::tanh(amp),
            ActFnSpec::ReLU { slope } => Layer::relu(slope),
        }
    }

    /// Resolves the `LossFn` for this trainer.
    ///
    /// # Args
    /// * `spec` - The specification for this trainer.
    /// * `optimizers` - A list of resolved optimizers.
    /// * `layers` - A list of resolved layers.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn resolve_loss_fn<O>(
        &self,
        spec: TrainerSpec,
        optimizers: Vec<O>,
        layers: Vec<Layer>,
    ) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
    {
        match spec.loss_fn {
            LossFnSpec::Mse => {
                let loss_fn = Mse::new();
                self.terminate_build(spec, optimizers, layers, loss_fn)
            }
            LossFnSpec::CrossEntropy => {
                let loss_fn = CrossEntropy::new();
                self.terminate_build(spec, optimizers, layers, loss_fn)
            }
        }
    }

    /// Terminates the entire build for this trainer and instanciates the final entity.
    ///
    /// # Args
    /// * `spec` - The specification for this trainer.
    /// * `optimizers` - A list of optimizers, one per server.
    /// * `layers` - A list of resolved layers.
    /// * `loss_fn` - A resolved loss function.
    ///
    /// # Returns
    /// A new `Trainer`.
    fn terminate_build<O, L>(
        &self,
        spec: TrainerSpec,
        optimizers: Vec<O>,
        layers: Vec<Layer>,
        loss_fn: L,
    ) -> Box<dyn Trainer>
    where
        O: Optimizer + Send + 'static,
        L: LossFn + Send + 'static,
    {
        let model = Sequential::new(layers);
        let DatasetSpec { x_size, y_size } = spec.dataset;
        let dataset = Dataset::new(x_size, y_size);
        let trainer = BackpropTrainer::new(
            model,
            optimizers,
            dataset,
            loss_fn,
            spec.offline_epochs,
            spec.max_epochs,
            spec.batch_size,
            self.generate_rng(spec.seed),
        );

        Box::new(trainer)
    }

    /// Generates a random number generator given (or not) a seed.
    ///
    /// # Args
    /// * `seed` - An optional seed for the rng.
    ///
    /// # Returns
    /// A new rng.
    fn generate_rng(&self, seed: Option<u64>) -> StdRng {
        match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        }
    }
}
