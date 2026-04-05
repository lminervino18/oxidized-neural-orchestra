mod activations;
mod arch;
mod datasets;
mod initialization;
mod loss_fns;
mod optimizers;
mod serializer;
mod session;
mod store;
mod sync;
mod training;

use pyo3::prelude::*;

#[pymodule(gil_used = true)]
fn _orchestra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<arch::Sequential>()?;
    m.add_class::<arch::Dense>()?;

    m.add_class::<activations::Sigmoid>()?;

    m.add_class::<initialization::Kaiming>()?;
    m.add_class::<initialization::Xavier>()?;
    m.add_class::<initialization::Lecun>()?;
    m.add_class::<initialization::XavierUniform>()?;
    m.add_class::<initialization::LecunUniform>()?;
    m.add_class::<initialization::Const>()?;
    m.add_class::<initialization::Uniform>()?;
    m.add_class::<initialization::UniformInclusive>()?;
    m.add_class::<initialization::Normal>()?;

    m.add_class::<datasets::InlineDataset>()?;
    m.add_class::<datasets::LocalDataset>()?;

    m.add_class::<optimizers::GradientDescent>()?;

    m.add_class::<loss_fns::Mse>()?;
    m.add_class::<loss_fns::CrossEntropy>()?;

    m.add_class::<serializer::BaseSerializer>()?;
    m.add_class::<serializer::SparseSerializer>()?;

    m.add_class::<sync::BarrierSync>()?;
    m.add_class::<sync::NonBlockingSync>()?;

    m.add_class::<store::BlockingStore>()?;
    m.add_class::<store::WildStore>()?;

    m.add_class::<session::Session>()?;
    m.add_class::<session::TrainedModel>()?;

    m.add_class::<training::PyTrainingConfig>()?;
    m.add_function(wrap_pyfunction!(training::parameter_server, m)?)?;
    m.add_function(wrap_pyfunction!(training::orchestrate, m)?)?;

    Ok(())
}
