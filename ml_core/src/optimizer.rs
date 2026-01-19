use super::{dataset::Dataset, model::Model};
use ndarray::Array1;

pub trait Optimizer {
    type ModelT: Model;

    // fn train(model: Self::ModelT, dataset: &Dataset);
    fn optimize(
        &self,
        model: &mut Self::ModelT,
        x_train: &[Array1<f32>],
        y_train: &[Array1<f32>],
        n_iters: usize, // se podría generalizar la condición de corte
    );
}
