use super::{dataset::Dataset, neural_net::NeuralNet, optimizer::Optimizer};

pub struct SGD {
    eta: f32,
}

impl Optimizer for SGD {
    type ModelT = NeuralNet;

    fn train(model: NeuralNet, dataset: &Dataset) {}
}
