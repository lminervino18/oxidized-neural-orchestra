use comms::specs::{machine_learning::LayerSpec, worker::WorkerSpec};
use machine_learning::{dataset::DatasetBuilder, training::TrainerBuilder};

use super::worker::Worker;

#[derive(Default)]
pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Creates a new `WorkerBuilder`.
    ///
    /// # Returns
    /// A new `WorkerBuilder` instance.
    pub fn new() -> Self {
        Self
    }

    /// Builds a parameter-server worker from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - The specification for a worker.
    /// * `server_sizes` - The amount of parameters held by each server.
    /// * `dataset_raw` - The raw dataset partition assigned to this worker.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build_parameter_server(
        &self,
        spec: WorkerSpec,
        server_sizes: &[usize],
        dataset_raw: Vec<f32>,
    ) -> Worker {
        self.build(spec, server_sizes, dataset_raw)
    }

    /// Builds a ring all-reduce worker from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - The specification for a worker.
    /// * `dataset_raw` - The raw dataset partition assigned to this worker.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build_ring_all_reduce(&self, spec: WorkerSpec, dataset_raw: Vec<f32>) -> Worker {
        let model_size = self.model_size(&spec);
        self.build(spec, &[model_size], dataset_raw)
    }

    fn build(&self, spec: WorkerSpec, server_sizes: &[usize], dataset_raw: Vec<f32>) -> Worker {
        let dataset_builder = DatasetBuilder::new();
        let dataset = dataset_builder.build_inmem(spec.dataset, dataset_raw);

        let trainer_builder = TrainerBuilder::new();
        let trainer = trainer_builder.build(spec.trainer, server_sizes, dataset);

        Worker::new(trainer)
    }

    fn model_size(&self, spec: &WorkerSpec) -> usize {
        spec.trainer
            .layers
            .iter()
            .map(|layer| match layer {
                LayerSpec::Dense { dim: (input, output), .. } => input * output + output,
            })
            .sum()
    }
}