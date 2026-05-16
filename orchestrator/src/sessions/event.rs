use crate::{OrchErr, StopReason, TrainedModel};

/// An event produced during a training session.
#[derive(Debug)]
pub enum TrainingEvent {
    PublishedLosses {
        worker_id: usize,
        losses: Vec<f64>,
    },
    WorkerDone(usize),
    TrainingComplete {
        model: TrainedModel,
        reason: StopReason,
    },
    Params(Vec<f32>),
    Error(OrchErr),
}
