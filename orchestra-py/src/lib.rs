mod activations;
mod arch;
mod datasets;
mod initialization;
mod loss_fns;
mod optimizers;
mod session;
mod store;
mod sync;
mod training;

use pyo3::prelude::*;

#[pymodule]
fn _orchestra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // model
    m.add_class::<arch::Sequential>()?;
    m.add_class::<arch::Dense>()?;

    // activations
    m.add_class::<activations::Sigmoid>()?;

    // initialization
    m.add_class::<initialization::Kaiming>()?;
    m.add_class::<initialization::Xavier>()?;
    m.add_class::<initialization::Lecun>()?;
    m.add_class::<initialization::XavierUniform>()?;
    m.add_class::<initialization::LecunUniform>()?;
    m.add_class::<initialization::Const>()?;
    m.add_class::<initialization::Uniform>()?;
    m.add_class::<initialization::UniformInclusive>()?;
    m.add_class::<initialization::Normal>()?;

    // datasets
    m.add_class::<datasets::InlineDataset>()?;
    m.add_class::<datasets::LocalDataset>()?;

    // optimizers
    m.add_class::<optimizers::GradientDescent>()?;

    // loss functions
    m.add_class::<loss_fns::Mse>()?;
    m.add_class::<loss_fns::CrossEntropy>()?;

    // sync
    m.add_class::<sync::BarrierSync>()?;
    m.add_class::<sync::NonBlockingSync>()?;

    // store
    m.add_class::<store::BlockingStore>()?;
    m.add_class::<store::WildStore>()?;

    // session
    m.add_class::<session::Session>()?;
    m.add_class::<session::TrainedModel>()?;

    // training
    m.add_class::<training::PyTrainingConfig>()?;
    m.add_function(wrap_pyfunction!(training::parameter_server, m)?)?;
    m.add_function(wrap_pyfunction!(training::orchestrate, m)?)?;

    Ok(())
}
