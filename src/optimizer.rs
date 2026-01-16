use super::{dataset::Dataset, model::Model};

pub trait Optimizer {
    type ModelT: Model;

    fn train(model: Self::ModelT, dataset: &Dataset);
}
