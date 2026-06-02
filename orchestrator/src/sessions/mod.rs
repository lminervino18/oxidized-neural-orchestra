mod cancel_handle;
mod convergence_tracker;
mod event_listener;
mod greater_than_one_usize;
mod loss_recorder;
mod session;
mod switch_tracker;
mod trained_model;
mod worker_listener;

use comms::{
    NetRtp, ParamServerHandle,
    specs::{machine_learning::TrainerSpec, server::ServerSpec},
};

pub use cancel_handle::CancelHandle;
pub use convergence_tracker::ConvergenceTracker;
pub use event_listener::EventListener;
pub use greater_than_one_usize::GreaterThanOneUsize;
pub use loss_recorder::LossRecorder;
pub use session::Session;
pub use switch_tracker::SwitchTracker;
pub use trained_model::TrainedModel;
pub use worker_listener::WorkerListener;

use crate::OrchErr;

/// Requests that the orchestrator can make to a worker handler task.
#[derive(Debug, Clone)]
pub enum WorkerRequest {
    PullParams,
    Disconnect,
    Switch {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
        trainer_spec: Box<TrainerSpec>,
    },
    Upgrade {
        spec: Box<ServerSpec>,
        ranges: Vec<(usize, usize)>,
    },
    Stop,
}

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
        stop_reason: StopReason,
    },
    Params(Vec<f32>),
    Disconnect {
        worker_id: usize,
    },
    Upgrade {
        server_handle: Box<ParamServerHandle<NetRtp>>,
    },
    SwitchedToServer {
        worker_id: usize,
    },
    Error(OrchErr),
}

/// Why a training session ended.
#[derive(Debug, Clone, Copy, Default)]
pub enum StopReason {
    #[default]
    MaxEpochsReached,
    EarlyStopping,
    ManualStop,
}
